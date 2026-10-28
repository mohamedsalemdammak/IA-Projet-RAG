# Structured Multilingual RAG

Refactoring du notebook fourni en projet Python modulaire, avec le même pipeline général : ingestion/OCR → chunking → validation → embeddings → FAISS → retrieval hybride (BM25 lexical + FAISS sémantique) → reranking.

## Embedding model

Le modèle est maintenant : `intfloat/multilingual-e5-large`.

- documents : `passage: ...`
- questions : `query: ...`
- embeddings normalisés pour FAISS Inner Product.
- la recherche combine les candidats lexicaux BM25 et les candidats sémantiques FAISS avant le reranking.

## Structure

```text
rag_python_structured/
├── main.py
├── config.py
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   ├── processed/
│   └── index/
├── src/
│   ├── ingestion.py
│   ├── chunking.py
│   ├── validation.py
│   ├── embeddings.py
│   ├── vector_store.py
│   └── retrieval.py
└── notebooks/
```

Chaque fichier représente une étape importante, pas une fonction isolée.

Le chunking conserve le texte de chaque chunk dans `text` et ajoute dans
`metadata` le titre, le type (`pv`, `rm` ou `rf` selon le dossier source), la
date, le lieu, le nom du document et le nombre de tokens. `token_count` concerne
le chunk ; `document_token_count` concerne tout le texte extrait du document.
Les champs date et lieu restent à `null` s’ils ne peuvent pas être repérés dans
le texte. Le comptage des tokens est lexical (mots et ponctuation), sans charger
le tokenizer du modèle d’embedding.

Le fichier `data/processed/ingestion_report.json` contient le texte extrait
complet de chaque document dans `files[].extracted_text`, sans le découper en
chunks. La commande `chunk` produit séparément `data/processed/chunks.json` avec
des chunks de 255 tokens et un chevauchement de 40 tokens, en conservant les
métadonnées du document et en ajoutant `chunk_index`.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Tesseract OCR doit aussi être installé si les documents scannés doivent être traités.

## Utilisation

Mettre les documents dans `data/raw/`, puis :

```bash
python main.py ingest
python main.py chunk
python main.py check
python main.py embed
python main.py index
python main.py search "Quels sont les projets réalisés en 2024 ?"
```

Ou tout exécuter :

```bash
python main.py pipeline
```

### Recherche avec contexte conversationnel

Le retriever peut enrichir la question avant la recherche sémantique, BM25 et le
reranking avec :

- les cinq derniers messages (`conversation_history`) ;
- un résumé (`summary`) ;
- une liste de mots-clés (`keywords`) ;
- la question courante (`query`).

Les messages peuvent être des dictionnaires `{"role": "...", "content": "..."}`
ou de simples chaînes :

```python
from src.retrieval import HybridRetriever

result = retriever.search(
    "Quels projets ont été réalisés ?",
    conversation_history=messages,
    keywords=["projet", "réalisation", "2024"],
    summary="L'utilisateur recherche les projets réalisés en 2024.",
)
```

Le résultat conserve la question originale dans `query` et expose la requête
enrichie utilisée pour la recherche dans `search_query`.

La fiche de prompt dédiée à cette étape est disponible via
`build_enhanced_search_prompt`. Elle encadre explicitement la demande actuelle,
le résumé, les mots-clés et les cinq derniers messages, puis produit une
requête centrée sur les concepts à rechercher par l'embedding :

```python
from src.pipeline import build_enhanced_search_prompt

search_prompt = build_enhanced_search_prompt(
    question="Quels projets ont été réalisés ?",
    conversation_history=messages,
    summary="L'utilisateur cherche les projets réalisés.",
    keywords=["projet", "réalisation", "2024"],
)
```

La version LLM de cette étape est disponible dans
`src/promptAmelioration.py` :

```python
from src.promptAmelioration import improve_user_prompt

search_query = improve_user_prompt(
    user_prompt="Quels projets ont été réalisés ?",
    conversation_history=messages,
    summary="L'utilisateur cherche les projets réalisés.",
    keywords=["projet", "réalisation", "2024"],
)
```

`RAGPipeline.run` utilise cette fonction avant l'appel au retriever. La requête
retournée par le LLM est donc celle qui est transformée en embedding et
envoyée à la recherche RAG.

### Pipeline RAG complète avec Groq

La classe `RAGPipeline` orchestre les étapes suivantes :

1. écarter une question hors sujet ;
2. enrichir la recherche avec les cinq derniers messages, le résumé et les mots-clés ;
3. récupérer les meilleurs résultats et limiter le contexte aux cinq premiers chunks ;
4. construire le prompt avec règles de fiabilité et ton professionnel ;
5. appeler le modèle Qwen via l'API Groq ;
6. mettre à jour le résumé et les mots-clés.

Le modèle par défaut est `qwen/qwen3-32b` et peut être remplacé par la variable
`GROQ_MODEL`. La clé ne doit jamais être écrite dans le code :

```powershell
$env:GROQ_API_KEY = "votre-cle-groq"
python main.py ask "Quels projets ont été réalisés en 2024 ?" `
  --summary "Recherche des projets réalisés" `
  --keyword projet --keyword 2024
```

Pour une intégration applicative :

```python
from src.pipeline import RAGPipeline

pipeline = RAGPipeline(retriever)
result = pipeline.run(
    question,
    conversation_history=messages,
    summary=summary,
    keywords=keywords,
)
```

### Prompt Building Groq

La fonction `build_prompt()` de
[`src/promptbuilding.py`](C:/Users/Faizw/Desktop/rag_python_structured/src/promptbuilding.py)
retourne deux messages compatibles avec `chat.completions.create` :

```python
from src.promptbuilding import build_prompt

messages = build_prompt(
    user_question="Quels projets ont été réalisés en 2024 ?",
    top_chunks=search_result["results"][:5],
    last_5_messages=conversation_history[-5:],
    conversation_summary=summary,
    keywords=keywords,
)
answer = llm.complete_messages(messages)
```

Le contexte documentaire est limité par `MAX_CONTEXT_CHARS` dans
[`config.py`](C:/Users/Faizw/Desktop/rag_python_structured/config.py), et le
module peut être testé sans clé API :

```powershell
py -m src.promptbuilding
```

### Mémoire conversationnelle

Après la réponse du LLM, `RAGPipeline` appelle
[`update_conversation_memory`](C:/Users/Faizw/Desktop/rag_python_structured/src/update_conversation_memory.py)
avec l'ancien résumé, les anciens mots-clés, la question originale et la
réponse générée. Le LLM retourne un objet JSON contenant un résumé consolidé
(150 mots maximum) et au plus 15 mots-clés dédoublonnés. En cas d'erreur Groq,
de JSON invalide ou de données incorrectes, l'ancienne mémoire est conservée
et la réponse principale reste disponible.

### Détection du périmètre avec Laya

La pipeline utilise le modèle décisionnel open source
[`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya).
La documentation officielle indique l'installation `pip install laya` et
l'interface `Router().predict(state, questions, model="multilingual",
max_len=8192)`. Laya est chargé paresseusement une seule fois par
`LayaScopeDetector`.

Le détecteur retourne `IN_SCOPE`, `OUT_OF_SCOPE` ou `UNCERTAIN`. Une erreur
technique, une réponse invalide ou l'absence du package ne devient jamais
`OUT_OF_SCOPE` : avec `SCOPE_DETECTION_FALLBACK = 'uncertain'`, la pipeline
demande une précision sans lancer l'embedding, la recherche ou le LLM.

Pour installer Laya :

```powershell
pip install laya
```

Tester sans télécharger le modèle avec les tests utilisant un faux routeur :

```powershell
pytest tests/test_scope_detector.py
```

### Pipeline conversationnelle persistante

La classe `RAGPipeline` charge et sauvegarde la mémoire par
`conversation_id` dans `data/processed/conversations.json`. Cette mémoire
contient le résumé, les mots-clés et les cinq derniers messages.

Pour traiter une question :

```powershell
python main.py pipeline --question "Quels projets ont été réalisés ?" `
  --conversation-id utilisateur-1
```

Pour ouvrir une session interactive :

```powershell
python main.py chat --conversation-id utilisateur-1
```

La mémoire est isolée par identifiant, et une erreur de sauvegarde ne bloque
pas la réponse générée.

Pour tester le `main` de la pipeline avec les trois appels Groq réels
(amélioration de requête, génération de réponse et mise à jour mémoire) :

```powershell
$env:GROQ_API_KEY = "votre-cle-groq"
py -m src.pipeline "Quels projets ont été réalisés en 2024 ?"
```

Le retriever et le détecteur sont simulés dans ce test afin de vérifier
spécifiquement les appels LLM sans charger FAISS ni Laya. Pour un test sans
réseau ni clé API :

```powershell
py -m src.pipeline --mock
```

## Important

Les paramètres historiques du notebook (chunk size 1500, overlap 300, logique année/intention, reranker et pénalités) sont conservés autant que possible, mais répartis dans les modules pour faciliter la maintenance.
