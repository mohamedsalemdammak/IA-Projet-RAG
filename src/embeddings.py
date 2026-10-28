import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME


class E5Embedder:
    def __init__(self, model_name=EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts, batch_size=32):
        prepared = [f'passage: {text}' for text in texts]
        return self.model.encode(prepared, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)

    def embed_query(self, query):
        return self.model.encode(f'query: {query}', normalize_embeddings=True)


def build_embeddings(chunks_file, embeddings_file, batch_size=32):
    chunks = json.loads(Path(chunks_file).read_text(encoding='utf-8'))
    if not isinstance(chunks, list) or not chunks:
        raise ValueError('The chunks file must contain a non-empty JSON list')
    if batch_size <= 0:
        raise ValueError('batch_size must be greater than zero')

    texts = []
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict) or not isinstance(chunk.get('text'), str):
            raise ValueError(f'Chunk {index} must contain text as a string')
        text = chunk['text'].strip()
        if not text:
            raise ValueError(f'Chunk {index} has empty text')
        texts.append(text)

    model = E5Embedder()
    embeddings = np.asarray(
        model.embed_documents(texts, batch_size=batch_size),
        dtype='float32',
    )
    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise ValueError(
            f'Expected one embedding per chunk ({len(chunks)}), '
            f'got shape {embeddings.shape}'
        )

    Path(embeddings_file).parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_file, embeddings)
    return embeddings.shape
