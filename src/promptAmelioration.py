"""Amélioration LLM des requêtes avant la recherche RAG."""

import argparse

from src.llm import GroqLLM


MAX_HISTORY_MESSAGES = 5

SEARCH_QUERY_SYSTEM_PROMPT = """Tu es un module de reformulation pour une recherche RAG.
Ta tâche est d'améliorer la demande de l'utilisateur avant son passage dans un
modèle d'embedding.

Règles obligatoires :
- conserve exactement l'intention de la demande actuelle ;
- utilise l'historique uniquement pour résoudre les références ambiguës ;
- utilise le résumé et les mots-clés pour préciser le contexte ;
- conserve les noms, dates, lieux, nombres et contraintes importants ;
- n'ajoute aucune information absente des entrées ;
- ne réponds pas à la question ;
- ne produis ni explication, ni liste, ni guillemets, ni préambule ;
- retourne uniquement une requête de recherche claire et autonome ;
- écris la requête dans la langue de la demande utilisateur.
"""


def _format_history(conversation_history):
    messages = list(conversation_history or [])[-MAX_HISTORY_MESSAGES:]
    lines = []
    for message in messages:
        if isinstance(message, dict):
            role = str(message.get("role", "message")).strip() or "message"
            content = str(
                message.get("content", message.get("text", ""))
            ).strip()
            if content:
                lines.append(f"{role}: {content}")
        else:
            content = str(message).strip()
            if content:
                lines.append(content)
    return "\n".join(lines) or "Aucun historique disponible."


def _format_keywords(keywords):
    if isinstance(keywords, str):
        return keywords.strip() or "Aucun mot-clé disponible."
    values = [
        str(keyword).strip()
        for keyword in (keywords or [])
        if keyword is not None and str(keyword).strip()
    ]
    return ", ".join(values) or "Aucun mot-clé disponible."


def build_search_prompt(
    user_prompt,
    conversation_history=None,
    keywords=None,
    summary="",
):
    """Construit le prompt envoyé au LLM pour améliorer une requête RAG."""
    return f"""Demande actuelle de l'utilisateur:
{str(user_prompt or '').strip()}

Résumé de la conversation:
{str(summary or '').strip() or 'Aucun résumé disponible.'}

Mots-clés:
{_format_keywords(keywords)}

Cinq derniers messages de la conversation:
{_format_history(conversation_history)}

Retourne maintenant uniquement la requête de recherche améliorée."""


def improve_user_prompt(
    user_prompt,
    conversation_history=None,
    keywords=None,
    summary="",
    llm=None,
):
    """Améliore la demande utilisateur avec le LLM pour la recherche RAG."""
    if not str(user_prompt or "").strip():
        raise ValueError("La demande utilisateur ne peut pas être vide.")

    model = llm or GroqLLM()
    improved_prompt = model.complete(
        SEARCH_QUERY_SYSTEM_PROMPT,
        build_search_prompt(
            user_prompt,
            conversation_history=conversation_history,
            keywords=keywords,
            summary=summary,
        ),
        temperature=0.0,
    ).strip()
    if not improved_prompt:
        raise RuntimeError("Le LLM a retourné une requête de recherche vide.")
    return improved_prompt


def main():
    parser = argparse.ArgumentParser(
        description="Teste l'amélioration LLM d'une requête pour la recherche RAG."
    )
    parser.add_argument(
        "question",
        nargs="?",
        default="Je n'ai pas compris.",
        help="Question utilisateur à améliorer.",
    )
    parser.add_argument(
        "--summary",
        default="L'utilisateur cherche à comprendre les projets et activités de l'association.",
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
        help="Dernier message de conversation. Peut être répété.",
    )
    args = parser.parse_args()

    history = [
        {"role": "user", "content": message}
        for message in (
            args.messages
            or [
                "Nous parlons des activités de l'association.",
                "Je veux comprendre les projets réalisés.",
            ]
        )
    ]

    print("Question originale:")
    print(args.question)
    print("\nRequête améliorée pour la recherche RAG:")
    print(
        improve_user_prompt(
            user_prompt=args.question,
            conversation_history=history,
            keywords=args.keywords or ["projet", "activité", "réalisation"],
            summary=args.summary,
        )
    )


if __name__ == "__main__":
    main()