from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
INDEX_DIR = DATA_DIR / 'index'

CHUNKS_FILE = PROCESSED_DIR / 'chunks.json'
REPORT_FILE = PROCESSED_DIR / 'ingestion_report.json'
EMBEDDINGS_FILE = INDEX_DIR / 'embeddings.npy'
FAISS_FILE = INDEX_DIR / 'faiss.index'

CHUNK_SIZE = 255
CHUNK_OVERLAP = 40
OCR_LANG = 'fra+ara+eng'

EMBEDDING_MODEL_NAME = 'intfloat/multilingual-e5-large'
RERANKER_MODEL_NAME = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'

CANDIDATE_K = 100
RERANK_K = 40
TOP_K = 5
MAX_PER_SOURCE = 3

GROQ_MODEL = 'qwen/qwen3.8-27b'
GROQ_API_KEY_ENV = 'GROQ_API_KEY'
ANSWER_TEMPERATURE = 0.2
ANSWER_MAX_TOKENS = 1200
MEMORY_MAX_TOKENS = 300
MAX_CONTEXT_CHARS = 12000
CONVERSATIONS_FILE = PROCESSED_DIR / 'conversations.json'
DOMAIN_DESCRIPTION = (
    "Documents de l'association concernant les AGP, projets, activités, "
    "réunions, formations, rapports, procès-verbaux et statistiques."
)
LAYA_MODEL = 'multilingual'
LAYA_MAX_LENGTH = 8192
SCOPE_DETECTION_FALLBACK = 'uncertain'

for folder in (RAW_DIR, PROCESSED_DIR, INDEX_DIR):
    folder.mkdir(parents=True, exist_ok=True)
