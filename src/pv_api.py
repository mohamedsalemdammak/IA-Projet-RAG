"""REST APIs for meeting audio, minutes generation, and RAG indexing.

Start from the repository root with:

    uvicorn src.pv_api:app --reload

Examples:

    curl -X POST -F "audio=@reunion.mp3" http://localhost:8000/audio-to-text
    curl -X POST http://localhost:8000/summarize-pv \
      -H "Content-Type: application/json" \
      -d "{\"transcription\":\"Alice présente le budget.\"}"
    curl -X POST http://localhost:8000/pv-to-embedding \
      -H "Content-Type: application/json" \
      -d "{\"pv\":\"Décision: valider le budget.\",\"metadata\":{\"date\":\"2026-10-19\"}}"
    curl -X POST http://localhost:8000/report-to-embedding \
      -H "Content-Type: application/json" \
      -d "{\"report\":\"Contenu du rapport.\",\"title\":\"Rapport annuel 2026\"}"

Example responses:

    {"transcription": "Alice présente le budget."}
    {"participants": ["Alice"], "subjects_discussed": ["budget"],
     "decisions": [], "actions": []}
    {"pv_id": "...", "chunks_count": 1, "status": "indexed"}

The module deliberately imports the existing RAG components instead of creating
another vector store.  The indexing endpoint updates the same FAISS index and
chunks file used by ``main.py``.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import tempfile
from pathlib import Path
from threading import Lock
from typing import Any, Annotated

import faiss
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field, field_validator

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKS_FILE,
    EMBEDDINGS_FILE,
    FAISS_FILE,
    GROQ_API_KEY_ENV,
    GROQ_MODEL,
)
from src.chunking import chunk_text
from src.embeddings import E5Embedder
from src.llm import GroqLLM


logger = logging.getLogger(__name__)

MAX_AUDIO_BYTES = int(os.getenv("PV_MAX_AUDIO_BYTES", str(25 * 1024 * 1024)))
MAX_TRANSCRIPTION_CHARS = int(
    os.getenv("PV_MAX_TRANSCRIPTION_CHARS", "200000")
)
MAX_PV_CHARS = int(os.getenv("PV_MAX_PV_CHARS", "200000"))
MAX_REPORT_CHARS = int(os.getenv("PV_MAX_REPORT_CHARS", "200000"))
WHISPER_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")
WHISPER_LANGUAGE = os.getenv("PV_AUDIO_LANGUAGE", "ar")
WHISPER_PROMPT = os.getenv(
    "PV_WHISPER_PROMPT",
    "اجتماع باللهجة التونسية. أسماء الأشخاص والمؤسسات التونسية "
    "والدارجة التونسية. حافظ على الكلمات كما نُطقت.",
)
ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a"}

_index_lock = Lock()
_embedder: E5Embedder | None = None
_chat_pipeline: RAGPipeline | None = None
_chat_store_signature: tuple[int, int, int, int] | None = None


class ActionItem(BaseModel):
    action: str = Field(description="Action explicitly stated in the transcription")
    responsible: str | None = Field(
        default=None, description="Responsible person, only when stated"
    )
    deadline: str | None = Field(
        default=None, description="Deadline, only when stated"
    )


class Minutes(BaseModel):
    participants: list[str] = Field(default_factory=list)
    subjects_discussed: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    actions: list[ActionItem] = Field(default_factory=list)


class SummarizeRequest(BaseModel):
    transcription: str = Field(
        min_length=1,
        max_length=MAX_TRANSCRIPTION_CHARS,
        examples=["La réunion commence à 10h. Alice présente le budget."],
    )

    @field_validator("transcription")
    @classmethod
    def validate_transcription(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("La transcription ne peut pas être vide.")
        return value


class IndexRequest(BaseModel):
    pv: str = Field(
        min_length=1,
        max_length=MAX_PV_CHARS,
        examples=["Participants: Alice. Décision: valider le budget."],
    )
    pv_id: str | None = Field(
        default=None,
        max_length=200,
        description="Identifiant stable facultatif; sinon SHA-256 du PV.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Métadonnées conservées avec chaque chunk.",
    )

    @field_validator("pv")
    @classmethod
    def validate_pv(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Le PV ne peut pas être vide.")
        return value

    @field_validator("pv_id")
    @classmethod
    def validate_pv_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("pv_id ne peut pas être vide.")
        return value


class ReportIndexRequest(BaseModel):
    report: str = Field(
        examples=["Le rapport annuel présente les activités réalisées en 2026."],
    )
    title: str | None = Field(default=None, max_length=500)
    document_id: str | None = Field(default=None, max_length=200)

    @field_validator("report")
    @classmethod
    def validate_report(cls, value: str) -> str:
        return value.strip()

    @field_validator("title", "document_id")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("Le titre et l'identifiant ne peuvent pas être vides.")
        return value


class IndexResponse(BaseModel):
    pv_id: str
    chunks_count: int
    status: str


class ReportIndexResponse(BaseModel):
    document_id: str
    chunks_count: int
    status: str


class ChatRequest(BaseModel):
    conversation_id: str = Field(max_length=100)
    question: str

    @field_validator("conversation_id")
    @classmethod
    def validate_conversation_id(cls, value: str) -> str:
        return value.strip()

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        return value.strip()


class ChatResponse(BaseModel):
    conversation_id: str
    question: str
    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    out_of_scope: bool = False
    optimized_search_query: str | None = None


app = FastAPI(
    title="PV APIs",
    description="Transcription Whisper, synthèse Qwen et indexation dans le FAISS du pipeline RAG.",
    version="1.0.0",
)


def _raise_api_error(message: str, status_code: int = 502) -> None:
    raise HTTPException(status_code=status_code, detail=message)


def _audio_extension(filename: str | None) -> str:
    return Path(filename or "").suffix.lower()


def _read_audio(upload: UploadFile) -> tuple[str, bytes]:
    extension = _audio_extension(upload.filename)
    if extension not in ALLOWED_AUDIO_EXTENSIONS:
        _raise_api_error(
            "Format audio non supporté. Utilisez MP3, WAV ou M4A.",
            status.HTTP_400_BAD_REQUEST,
        )

    content = upload.file.read(MAX_AUDIO_BYTES + 1)
    if not content:
        _raise_api_error("Le fichier audio est vide.", status.HTTP_400_BAD_REQUEST)
    if len(content) > MAX_AUDIO_BYTES:
        _raise_api_error(
            f"Le fichier audio dépasse la limite de {MAX_AUDIO_BYTES} octets.",
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        )
    return upload.filename or f"audio{extension}", content


def _transcribe_audio(filename: str, content: bytes) -> str:
    try:
        from groq import Groq

        api_key = os.getenv(GROQ_API_KEY_ENV)
        if not api_key:
            _raise_api_error(
                f"La variable d'environnement {GROQ_API_KEY_ENV} est requise.",
                status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        response = Groq(api_key=api_key).audio.transcriptions.create(
            file=(filename, io.BytesIO(content)),
            model=WHISPER_MODEL,
            language=WHISPER_LANGUAGE,
            prompt=WHISPER_PROMPT,
            response_format="json",
        )
    except HTTPException:
        raise
    except (OSError, ValueError) as exc:
        logger.exception("Erreur lors de la préparation de la transcription")
        _raise_api_error(f"Impossible de préparer la transcription: {exc}")
    except Exception as exc:
        logger.exception("Erreur Groq Whisper")
        _raise_api_error(f"Erreur de l'API Whisper: {exc}")

    text = str(getattr(response, "text", "") or "").strip()
    if not text:
        _raise_api_error("Whisper a retourné une transcription vide.")
    if len(text) > MAX_TRANSCRIPTION_CHARS:
        _raise_api_error(
            "La transcription dépasse la taille maximale autorisée.",
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        )
    return text


def _minutes_prompt(transcription: str) -> tuple[str, str]:
    system = (
        "Tu rédiges un procès-verbal fidèle en français. Retourne uniquement un "
        "objet JSON avec les clés participants, subjects_discussed, decisions et "
        "actions. Chaque action est un objet avec action, responsible et deadline. "
        "N'invente jamais une information: utilise une liste vide ou null lorsque "
        "l'information n'est pas explicitement présente dans la transcription."
    )
    user = (
        "Transforme la transcription suivante en PV structuré. Conserve les "
        "incertitudes et ne déduis pas les participants, responsables ou échéances.\n\n"
        f"TRANSCRIPTION:\n{transcription}"
    )
    return system, user


def _summarize(transcription: str) -> Minutes:
    try:
        system, user = _minutes_prompt(transcription)
        result = GroqLLM(model=os.getenv("GROQ_MODEL", GROQ_MODEL)).complete_json(
            system, user, max_tokens=1200
        )
        return Minutes.model_validate(result)
    except HTTPException:
        raise
    except (RuntimeError, ValueError) as exc:
        logger.exception("Réponse Qwen invalide")
        _raise_api_error(f"Impossible de générer un PV structuré: {exc}")
    except Exception as exc:
        logger.exception("Erreur Groq Qwen")
        _raise_api_error(f"Erreur de l'API Qwen: {exc}")


def _get_embedder() -> E5Embedder:
    global _embedder
    if _embedder is None:
        _embedder = E5Embedder()
    return _embedder


def _stable_pv_id(pv: str, requested_id: str | None) -> str:
    if requested_id:
        return requested_id.strip()
    return hashlib.sha256(pv.encode("utf-8")).hexdigest()


def _load_chunks() -> list[dict[str, Any]]:
    if not Path(CHUNKS_FILE).exists():
        return []
    value = json.loads(Path(CHUNKS_FILE).read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("Le fichier de chunks doit contenir une liste JSON.")
    return value


def _write_json_atomically(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary:
        json.dump(value, temporary, ensure_ascii=False, indent=2)
        temporary.flush()
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def _index_document(
    text: str,
    document_id: str,
    metadata: dict[str, Any],
    identifier_key: str,
) -> tuple[int, bool]:
    with _index_lock:
        try:
            chunks = _load_chunks()
            existing = [
                chunk
                for chunk in chunks
                if chunk.get("metadata", {}).get(identifier_key) == document_id
            ]
            if existing:
                return len(existing), True

            pieces = chunk_text(
                text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP
            )
            if not pieces:
                _raise_api_error(
                    "Le document ne contient aucun texte indexable.",
                    status.HTTP_400_BAD_REQUEST,
                )

            new_chunks = [
                {
                    "text": text,
                    "metadata": {
                        **metadata,
                        "chunk_index": index,
                        "chunk_id": f"{document_id}:{index}",
                        "token_count": token_count,
                    },
                }
                for index, (text, token_count) in enumerate(pieces)
            ]
            embeddings = np.asarray(
                _get_embedder().embed_documents(
                    [chunk["text"] for chunk in new_chunks], batch_size=32
                ),
                dtype="float32",
            )
            if embeddings.ndim != 2 or embeddings.shape[0] != len(new_chunks):
                raise ValueError("Le modèle n'a pas retourné un embedding par chunk.")

            index_path = Path(FAISS_FILE)
            if index_path.exists():
                vector_index = faiss.read_index(str(index_path))
                if vector_index.ntotal != len(chunks):
                    raise ValueError(
                        "L'index FAISS existant et les chunks sont désynchronisés."
                    )
                if vector_index.d != embeddings.shape[1]:
                    raise ValueError("La dimension des embeddings diffère de l'index FAISS.")
            else:
                vector_index = faiss.IndexFlatIP(embeddings.shape[1])
                if chunks:
                    raise ValueError(
                        "Les chunks existent mais l'index FAISS est absent."
                    )
            vector_index.add(embeddings)

            updated_chunks = chunks + new_chunks
            if vector_index.ntotal != len(updated_chunks):
                raise ValueError("L'index FAISS et le fichier de chunks sont désynchronisés.")

            faiss.write_index(vector_index, str(index_path))
            _write_json_atomically(Path(CHUNKS_FILE), updated_chunks)

            embeddings_path = Path(EMBEDDINGS_FILE)
            if embeddings_path.exists():
                previous = np.asarray(np.load(embeddings_path), dtype="float32")
                combined = np.concatenate((previous, embeddings), axis=0)
            else:
                combined = embeddings
            if combined.shape[0] != len(updated_chunks):
                raise ValueError("Le fichier d'embeddings et les chunks sont désynchronisés.")
            embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(embeddings_path, combined)
            return len(new_chunks), False
        except HTTPException:
            raise
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.exception("Erreur d'indexation du document")
            _raise_api_error(f"Indexation impossible: {exc}")
        except Exception as exc:
            logger.exception("Erreur inattendue d'indexation du document")
            _raise_api_error(f"Indexation impossible: {exc}")


def _index_pv(request: IndexRequest) -> IndexResponse:
    pv_id = _stable_pv_id(request.pv, request.pv_id)
    metadata = {
        **request.metadata,
        "pv_id": pv_id,
        "source": request.metadata.get("source", f"pv:{pv_id}"),
        "document_name": request.metadata.get("document_name", pv_id),
        "type": request.metadata.get("type", "pv"),
    }
    chunks_count, already_indexed = _index_document(
        request.pv, pv_id, metadata, "pv_id"
    )
    return IndexResponse(
        pv_id=pv_id,
        chunks_count=chunks_count,
        status="already_indexed" if already_indexed else "indexed",
    )


def _stable_report_id(report: str, requested_id: str | None) -> str:
    if requested_id:
        return requested_id.strip()
    return hashlib.sha256(report.encode("utf-8")).hexdigest()


def _index_report(request: ReportIndexRequest) -> ReportIndexResponse:
    if not request.report:
        _raise_api_error(
            "Le rapport ne peut pas être vide.", status.HTTP_400_BAD_REQUEST
        )
    if len(request.report) > MAX_REPORT_CHARS:
        _raise_api_error(
            f"Le rapport dépasse la limite de {MAX_REPORT_CHARS} caractères.",
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        )
    document_id = _stable_report_id(request.report, request.document_id)
    metadata = {
        "source": "rapport",
        "type": "rapport",
        "title": request.title or document_id,
        "document_id": document_id,
    }
    chunks_count, already_indexed = _index_document(
        request.report, document_id, metadata, "document_id"
    )
    return ReportIndexResponse(
        document_id=document_id,
        chunks_count=chunks_count,
        status="already_indexed" if already_indexed else "indexed",
    )


def _store_signature() -> tuple[int, int, int, int]:
    index_path = Path(FAISS_FILE)
    chunks_path = Path(CHUNKS_FILE)
    return (
        index_path.stat().st_mtime_ns,
        index_path.stat().st_size,
        chunks_path.stat().st_mtime_ns,
        chunks_path.stat().st_size,
    )


def _get_chat_pipeline() -> RAGPipeline:
    global _chat_pipeline, _chat_store_signature
    from src.pipeline import RAGPipeline
    from src.retrieval import HybridRetriever
    from src.vector_store import load_store

    signature = _store_signature()
    if _chat_pipeline is None or _chat_store_signature != signature:
        index, chunks = load_store(FAISS_FILE, CHUNKS_FILE)
        _chat_pipeline = RAGPipeline(HybridRetriever(index, chunks))
        _chat_store_signature = signature
    return _chat_pipeline


@app.post(
    "/audio-to-text",
    summary="Transcrire un fichier audio avec Whisper via Groq",
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "transcription": "توة نبدأو الاجتماع ونحكيو على الميزانية."
                    }
                }
            }
        },
        400: {"description": "Format invalide ou fichier vide"},
        413: {"description": "Fichier trop volumineux"},
        502: {"description": "Erreur de l'API Groq"},
    },
)
async def audio_to_text(
    audio: Annotated[UploadFile, File(description="Fichier MP3, WAV ou M4A")],
) -> dict[str, str]:
    filename, content = _read_audio(audio)
    return {"transcription": _transcribe_audio(filename, content)}


@app.post(
    "/summarize-pv",
    response_model=Minutes,
    summary="Générer un procès-verbal structuré avec Qwen via Groq",
    responses={
        200: {
            "description": "PV structuré sans ajout d'informations absentes",
            "content": {
                "application/json": {
                    "example": {
                        "participants": ["Alice"],
                        "subjects_discussed": ["Budget"],
                        "decisions": ["Valider le budget"],
                        "actions": [
                            {"action": "Envoyer le budget", "responsible": "Alice", "deadline": None}
                        ],
                    }
                }
            },
        }
    },
)
async def summarize_pv(request: SummarizeRequest) -> Minutes:
    return _summarize(request.transcription)


@app.post(
    "/pv-to-embedding",
    response_model=IndexResponse,
    summary="Indexer un PV dans le Vector Store FAISS existant",
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "pv_id": "a1b2c3",
                        "chunks_count": 2,
                        "status": "indexed",
                    }
                }
            }
        },
        409: {"description": "PV déjà indexé"},
    },
)
async def pv_to_embedding(request: IndexRequest) -> IndexResponse:
    return _index_pv(request)


@app.post(
    "/report-to-embedding",
    response_model=ReportIndexResponse,
    summary="Indexer un rapport dans le Vector Store FAISS existant",
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "document_id": "a1b2c3",
                        "chunks_count": 5,
                        "status": "indexed",
                    }
                }
            }
        },
        400: {"description": "Rapport vide ou sans texte indexable"},
        413: {"description": "Rapport trop volumineux"},
        502: {"description": "Erreur d'embedding, FAISS ou sauvegarde"},
    },
)
async def report_to_embedding(
    request: ReportIndexRequest,
) -> ReportIndexResponse:
    return _index_report(request)


@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Répondre à une question avec le pipeline conversationnel RAG",
    responses={
        400: {"description": "Question ou identifiant de conversation invalide"},
        502: {"description": "Erreur du pipeline RAG ou du LLM"},
        503: {"description": "Vector Store ou service requis indisponible"},
    },
)
async def chat(request: ChatRequest) -> ChatResponse:
    if not request.conversation_id:
        _raise_api_error(
            "conversation_id ne peut pas être vide.", status.HTTP_400_BAD_REQUEST
        )
    if not request.question:
        _raise_api_error(
            "La question ne peut pas être vide.", status.HTTP_400_BAD_REQUEST
        )

    try:
        result = _get_chat_pipeline().run(
            request.question,
            conversation_id=request.conversation_id,
        )
        search_result = result.get("search_result")
        sources = (
            search_result.get("results", [])
            if isinstance(search_result, dict)
            else []
        )
        return ChatResponse(
            conversation_id=request.conversation_id,
            question=request.question,
            answer=str(result["answer"]),
            sources=sources,
            out_of_scope=bool(result.get("out_of_scope", False)),
            optimized_search_query=result.get("enhanced_query"),
        )
    except HTTPException:
        raise
    except ValueError as exc:
        logger.warning("Requête chat invalide: %s", exc)
        _raise_api_error(str(exc), status.HTTP_400_BAD_REQUEST)
    except (OSError, RuntimeError) as exc:
        logger.exception("Erreur du pipeline chat")
        _raise_api_error(
            "Le pipeline conversationnel est temporairement indisponible.",
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    except Exception:
        logger.exception("Erreur inattendue du pipeline chat")
        _raise_api_error(
            "Impossible de traiter la conversation.",
            status.HTTP_502_BAD_GATEWAY,
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.pv_api:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))