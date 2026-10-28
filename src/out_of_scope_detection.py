"""Détection de périmètre avec le modèle décisionnel open source Laya."""

import argparse
import importlib
import logging

from config import (
    DOMAIN_DESCRIPTION,
    LAYA_MAX_LENGTH,
    LAYA_MODEL,
    SCOPE_DETECTION_FALLBACK,
)


logger = logging.getLogger(__name__)
DECISIONS = {"IN_SCOPE", "OUT_OF_SCOPE", "UNCERTAIN"}
MAX_HISTORY_MESSAGES = 5

SCOPE_SYSTEM_CONTEXT = """Tu es un classificateur de pertinence pour un système RAG.
Détermine uniquement si la demande appartient au domaine documentaire autorisé.
Ne réponds jamais à la question et n'effectue aucune recherche documentaire.
Utilise l'historique, le résumé et les mots-clés pour comprendre les références
implicites. Une question ambiguë doit être classée UNCERTAIN.
"""


class ScopeDetectionError(RuntimeError):
    """Erreur de chargement, d'inférence ou de parsing de Laya."""


def _format_history(messages):
    lines = []
    for message in list(messages or [])[-MAX_HISTORY_MESSAGES:]:
        if isinstance(message, dict):
            content = str(message.get("content", message.get("text", ""))).strip()
            role = str(message.get("role", "message")).strip() or "message"
            if content:
                lines.append(f"{role}: {content}")
        elif str(message).strip():
            lines.append(str(message).strip())
    return "\n".join(lines) or "Aucun message."


def _build_state(user_question, last_5_messages, conversation_summary,
                 keywords, domain_description):
    keyword_text = (
        keywords if isinstance(keywords, str)
        else ", ".join(str(value).strip() for value in (keywords or []) if str(value).strip())
    )
    return f"""{SCOPE_SYSTEM_CONTEXT}

DOMAINE DOCUMENTAIRE AUTORISÉ
{domain_description}

QUESTION ORIGINALE
{user_question}

HISTORIQUE (5 DERNIERS MESSAGES)
{_format_history(last_5_messages)}

RÉSUMÉ
{conversation_summary or 'Aucun résumé.'}

MOTS-CLÉS
{keyword_text or 'Aucun mot-clé.'}
"""


def _extract_result(raw_result):
    if not isinstance(raw_result, dict):
        raise ScopeDetectionError("Laya a retourné un format inattendu.")
    value = raw_result.get("scope", raw_result.get("answers", raw_result))
    if isinstance(value, dict) and "scope" in value:
        value = value["scope"]
    if isinstance(value, dict):
        decision = value.get(
            "choice",
            value.get("answer", value.get("value", value.get("decision"))),
        )
        probabilities = value.get("probabilities", {})
        confidence = value.get(
            "answerConfidence",
            value.get("probability", probabilities.get(decision, 0.0)
                      if isinstance(probabilities, dict) else 0.0),
        )
        reason = value.get("reason", "")
    else:
        decision, confidence, reason = value, 0.0, ""
    decision = str(decision or "").strip().upper()
    if decision not in DECISIONS:
        raise ScopeDetectionError(f"Décision Laya invalide: {decision!r}.")
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError) as error:
        raise ScopeDetectionError("La confiance Laya doit être numérique.") from error
    return {
        "decision": decision,
        "confidence": confidence,
        "reason": str(reason or "").strip(),
    }


class LayaScopeDetector:
    """Charge Router une seule fois et expose une classification stable."""

    def __init__(self, router=None, model=LAYA_MODEL, max_len=LAYA_MAX_LENGTH):
        self._router = router
        self.model = model
        self.max_len = max_len

    def _get_router(self):
        if self._router is None:
            try:
                Router = importlib.import_module("laya").Router
            except ImportError as error:
                raise ScopeDetectionError(
                    "Laya n'est pas installé. Installez le package officiel avec: pip install laya"
                ) from error
            try:
                self._router = Router()
            except Exception as error:
                raise ScopeDetectionError("Impossible de charger le modèle Laya.") from error
        return self._router

    def detect(self, user_question, last_5_messages, conversation_summary,
               keywords, domain_description):
        state = _build_state(
            user_question, last_5_messages, conversation_summary,
            keywords, domain_description,
        )
        questions = {
            "scope": {
                "type": "choice",
                "instructions": "La demande appartient-elle au domaine autorisé ?",
                "criteria": {
                    "IN_SCOPE": "La demande concerne directement le domaine documentaire.",
                    "OUT_OF_SCOPE": "La demande est clairement sans rapport avec le domaine.",
                    "UNCERTAIN": "La demande est ambiguë ou le contexte ne permet pas de décider.",
                },
            }
        }
        try:
            raw_result = self._get_router().predict(
                state,
                questions,
                model=self.model,
                max_len=self.max_len,
            )
            return _extract_result(raw_result)
        except ScopeDetectionError:
            raise
        except Exception as error:
            raise ScopeDetectionError("Erreur d'inférence Laya.") from error


def detect_out_of_scope(
    user_question,
    last_5_messages,
    conversation_summary,
    keywords,
    domain_description,
    detector=None,
):
    """Retourne une décision Laya; les erreurs techniques deviennent UNCERTAIN."""
    detector = detector or LayaScopeDetector()
    try:
        return detector.detect(
            user_question, last_5_messages, conversation_summary,
            keywords, domain_description,
        )
    except ScopeDetectionError as error:
        logger.warning("Détection Laya indisponible: %s", error)
        if SCOPE_DETECTION_FALLBACK != "uncertain":
            raise
        return {
            "decision": "UNCERTAIN",
            "confidence": 0.0,
            "reason": "Le modèle de détection est indisponible.",
        }


def main():
    parser = argparse.ArgumentParser(
        description="Teste la détection de périmètre avec le modèle Laya."
    )
    parser.add_argument(
        "question",
        nargs="?",
        default="Et qui est responsable ?",
        help="Question à classifier.",
    )
    parser.add_argument(
        "--summary",
        default="La conversation porte sur les AGP et leurs activités.",
        help="Résumé de la conversation.",
    )
    parser.add_argument(
        "--keyword",
        action="append",
        dest="keywords",
        default=None,
        help="Mot-clé à ajouter. Peut être répété.",
    )
    parser.add_argument(
        "--message",
        action="append",
        dest="messages",
        default=None,
        help="Message récent à ajouter. Peut être répété.",
    )
    parser.add_argument(
        "--domain",
        default=(
            "Documents de l'association concernant les AGP, projets, activités, "
            "réunions, formations, rapports et statistiques."
        ),
        help="Description du domaine documentaire.",
    )
    args = parser.parse_args()

    messages = args.messages or [
        "Nous parlons des activités des AGP.",
        "Je veux comprendre qui est responsable.",
    ]
    keywords = args.keywords or ["AGP", "activité", "responsable"]
    result = detect_out_of_scope(
        user_question=args.question,
        last_5_messages=[
            {"role": "user", "content": message} for message in messages
        ],
        conversation_summary=args.summary,
        keywords=keywords,
        domain_description=args.domain,
    )

    print("Question:")
    print(args.question)
    print("\nDécision Laya:")
    print(result["decision"])
    print(f"Confiance: {result['confidence']:.2f}")
    print(f"Raison: {result['reason'] or 'Non fournie.'}")


if __name__ == "__main__":
    main()