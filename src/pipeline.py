import argparse
import json
import re
import unicodedata
import logging

from src.llm import GroqLLM
from src.promptAmelioration import improve_user_prompt
from src.promptbuilding import build_prompt
from src.update_conversation_memory import update_conversation_memory
from src.out_of_scope_detection import detect_out_of_scope
from src.out_of_scope_detection import LayaScopeDetector
from src.conversation_memory import ConversationMemoryStore
from config import DOMAIN_DESCRIPTION, CHUNKS_FILE, FAISS_FILE

logger = logging.getLogger(__name__)

MAX_HISTORY_MESSAGES = 5
MAX_KEYWORDS = 12
DOMAIN_TERMS = (
    'projet', 'projets', 'activité', 'activités', 'activite', 'activites',
    'réunion', 'réunions', 'reunion', 'reunions', 'formation', 'formations',
    'rapport', 'rapports', 'pv', 'statistique', 'statistiques', 'association',
    'programme', 'événement', 'événements', 'evenement', 'evenements',
    'مشروع', 'نشاط', 'اجتماع', 'تقرير', 'احصائيات',
)

SEARCH_PROMPT_TEMPLATE = """Tu es un module de préparation de recherche pour un système RAG.
Améliore la demande de recherche à partir des informations ci-dessous.
Conserve l'intention, les entités, les dates et les contraintes utiles.
Utilise l'historique uniquement pour désambiguïser la question actuelle.
N'invente aucune information et ne réponds pas à la question.
Le texte produit sera envoyé à un modèle d'embedding : il doit rester centré
sur les concepts à rechercher dans les documents.

Demande actuelle:
{question}

Résumé de la conversation:
{summary}

Mots-clés:
{keywords}

Cinq derniers messages:
{history}

Requête améliorée pour la recherche:
"""


def _normalize_scope_text(text):
    text = unicodedata.normalize('NFKC', str(text or '')).lower()
    text = ''.join(
        char for char in unicodedata.normalize('NFD', text)
        if unicodedata.category(char) != 'Mn'
    )
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', text)).strip()


def is_in_scope(question, summary='', conversation_history=None):
    text = _normalize_scope_text(
        ' '.join(
            [
                str(question or ''),
                str(summary or ''),
                ' '.join(str(item.get('content', '')) for item in (conversation_history or [])[-2:] if isinstance(item, dict)),
            ]
        )
    )
    return any(term in text for term in DOMAIN_TERMS)


def build_enhanced_search_prompt(question, conversation_history=None,
                                 keywords=None, summary=''):
    history = conversation_history[-MAX_HISTORY_MESSAGES:] if conversation_history else []
    history_text = '\n'.join(
        f"{item.get('role', 'message')}: {item.get('content', '')}"
        if isinstance(item, dict) else str(item)
        for item in history
    )
    if isinstance(keywords, str):
        keyword_text = keywords.strip()
    else:
        keyword_text = ', '.join(
            str(keyword).strip()
            for keyword in (keywords or [])
            if keyword is not None and str(keyword).strip()
        )
    return SEARCH_PROMPT_TEMPLATE.format(
        question=str(question or '').strip(),
        summary=str(summary or '').strip() or 'Aucun résumé disponible.',
        keywords=keyword_text or 'Aucun mot-clé disponible.',
        history=history_text or 'Aucun historique disponible.',
    )


class RAGPipeline:
    def __init__(self, retriever, llm=None, scope_detector=None,
                 domain_description=DOMAIN_DESCRIPTION, memory_store=None):
        self.retriever = retriever
        self.llm = llm
        self.scope_detector = scope_detector
        self.domain_description = domain_description or DOMAIN_DESCRIPTION
        self.memory_store = memory_store or ConversationMemoryStore()

    def run(self, question, conversation_history=None, summary='', keywords=None,
            conversation_id=None):
        if not str(question or '').strip():
            raise ValueError("La question utilisateur ne peut pas être vide.")
        if conversation_id:
            stored = self.memory_store.load(conversation_id)
            conversation_history = stored['messages']
            summary = stored['summary']
            keywords = stored['keywords']
        history = list(conversation_history or [])
        keywords = list(keywords or [])
        scope = detect_out_of_scope(
            question,
            history,
            summary,
            keywords,
            self.domain_description,
            detector=self.scope_detector,
        )
        if scope['decision'] == 'OUT_OF_SCOPE':
            return {
                'answer': "Cette question ne semble pas concerner le domaine documentaire.",
                'out_of_scope': True,
                'scope_detection': scope,
                'search_result': None,
                'summary': summary,
                'keywords': keywords,
                'memory_updated': False,
            }
        if scope['decision'] == 'UNCERTAIN':
            return {
                'answer': "Pouvez-vous préciser le lien avec les documents de l'association ?",
                'out_of_scope': False,
                'scope_detection': scope,
                'search_result': None,
                'summary': summary,
                'keywords': keywords,
                'memory_updated': False,
            }

        if self.llm is None:
            self.llm = GroqLLM()

        enhanced_query = improve_user_prompt(
            question,
            conversation_history=history,
            keywords=keywords,
            summary=summary,
            llm=self.llm,
        )
        search_result = self.retriever.search(
            question,
            prepared_query=enhanced_query,
        )
        messages = build_prompt(
            user_question=question,
            top_chunks=search_result.get('results', []),
            last_5_messages=history,
            conversation_summary=summary,
            keywords=keywords,
        )
        if hasattr(self.llm, 'complete_messages'):
            answer = self.llm.complete_messages(messages)
        else:
            answer = self.llm.complete(
                messages[0]['content'],
                messages[1]['content'],
            )
        memory = update_conversation_memory(
            summary,
            keywords,
            question,
            answer,
            llm=self.llm,
        )
        updated_history = (history + [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': answer},
        ])[-5:]
        memory_updated = True
        if conversation_id:
            try:
                self.memory_store.save(
                    conversation_id,
                    {
                        'summary': memory['summary'],
                        'keywords': memory['keywords'],
                        'messages': updated_history,
                    },
                )
            except (OSError, RuntimeError, ValueError) as error:
                logger.warning("Mémoire non sauvegardée pour %s: %s", conversation_id, error)
                memory_updated = False
        return {
            'answer': answer,
            'out_of_scope': False,
            'scope_detection': scope,
            'enhanced_query': enhanced_query,
            'search_result': search_result,
            'summary': memory['summary'],
            'keywords': memory['keywords'],
            'memory_updated': memory_updated,
        }


def main():
    """Exécute la pipeline avec Laya, FAISS et Groq réels."""
    parser = argparse.ArgumentParser(
        description='Exécute le pipeline RAG conversationnel réel.'
    )
    parser.add_argument(
        'question',
        nargs='?',
        default="Je n'ai pas compris.",
        help="Question originale de l'utilisateur.",
    )
    parser.add_argument('--conversation-id', default='real-test')
    parser.add_argument('--summary', default='')
    parser.add_argument('--keyword', action='append', default=[])
    parser.add_argument('--message', action='append', default=[])
    parser.add_argument('--domain', default=DOMAIN_DESCRIPTION)
    args = parser.parse_args()

    if not FAISS_FILE.exists() or not CHUNKS_FILE.exists():
        raise RuntimeError(
            'Index RAG introuvable. Exécutez d abord: py main.py pipeline'
        )
    from src.vector_store import load_store
    from src.retrieval import HybridRetriever

    index, chunks = load_store(FAISS_FILE, CHUNKS_FILE)
    pipeline = RAGPipeline(
        HybridRetriever(index, chunks),
        llm=GroqLLM(),
        scope_detector=LayaScopeDetector(),
        domain_description=args.domain,
        memory_store=ConversationMemoryStore(),
    )
    result = pipeline.run(
        args.question,
        conversation_history=[
            {'role': 'user', 'content': message}
            for message in args.message
        ],
        summary=args.summary,
        keywords=args.keyword,
        conversation_id=args.conversation_id,
    )
    print(result['answer'])


if __name__ == '__main__':
    main()
