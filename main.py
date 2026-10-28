import argparse
import json
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from config import RAW_DIR, CHUNKS_FILE, REPORT_FILE, EMBEDDINGS_FILE, FAISS_FILE


def cmd_ingest(args):
    from src.ingestion import ingest_folder
    report=ingest_folder(args.input, REPORT_FILE)
    print(json.dumps(report, ensure_ascii=False, indent=2))

def cmd_chunk(args):
    from src.chunking import build_chunks
    chunks=build_chunks(args.report, args.output)
    print(f'Chunks: {len(chunks)}')

def cmd_check(args):
    from src.validation import validate_dataset
    result=validate_dataset(CHUNKS_FILE, REPORT_FILE)
    print(json.dumps(result, ensure_ascii=False, indent=2))

def cmd_embed(args):
    from src.embeddings import build_embeddings
    print('Embedding model: intfloat/multilingual-e5-large')
    print('Shape:', build_embeddings(CHUNKS_FILE, EMBEDDINGS_FILE))

def cmd_index(args):
    from src.vector_store import build_faiss_index
    print('FAISS:', build_faiss_index(EMBEDDINGS_FILE, FAISS_FILE))

def cmd_search(args):
    from src.vector_store import load_store
    from src.retrieval import HybridRetriever
    index,chunks=load_store(FAISS_FILE,CHUNKS_FILE)
    result=HybridRetriever(index,chunks).search(args.query)
    for i,item in enumerate(result['results'],1):
        print(
            f"\n#{i} score={item['final_score']:.4f} "
            f"| BM25={item['bm25_score']:.4f} "
            f"| semantic={item['semantic_score']:.4f} "
            f"| source={item['source']}"
        )
        print(item['text'][:1200])

def cmd_ask(args):
    from src.vector_store import load_store
    from src.retrieval import HybridRetriever
    from src.pipeline import RAGPipeline
    index, chunks = load_store(FAISS_FILE, CHUNKS_FILE)
    pipeline = RAGPipeline(
        HybridRetriever(index, chunks),
        domain_description=args.domain,
    )
    result = pipeline.run(
        args.query,
        summary=args.summary,
        keywords=args.keyword,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

def cmd_pipeline(args):
    if args.question:
        from src.vector_store import load_store
        from src.retrieval import HybridRetriever
        from src.pipeline import RAGPipeline
        index, chunks = load_store(FAISS_FILE, CHUNKS_FILE)
        result = RAGPipeline(
            HybridRetriever(index, chunks),
            domain_description=args.domain,
        ).run(args.question, conversation_id=args.conversation_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    from src.ingestion import ingest_folder
    from src.chunking import build_chunks
    from src.validation import validate_dataset
    from src.embeddings import build_embeddings
    from src.vector_store import build_faiss_index
    ingest_folder(args.input,REPORT_FILE)
    build_chunks(REPORT_FILE,CHUNKS_FILE)
    print(validate_dataset(CHUNKS_FILE,REPORT_FILE))
    print('Embeddings:',build_embeddings(CHUNKS_FILE,EMBEDDINGS_FILE))
    print('FAISS:',build_faiss_index(EMBEDDINGS_FILE,FAISS_FILE))

def cmd_chat(args):
    from src.vector_store import load_store
    from src.retrieval import HybridRetriever
    from src.pipeline import RAGPipeline

    index, chunks = load_store(FAISS_FILE, CHUNKS_FILE)
    pipeline = RAGPipeline(
        HybridRetriever(index, chunks),
        domain_description=args.domain,
    )
    print("Mode conversationnel. Tapez 'exit' pour quitter.")
    while True:
        try:
            question = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {'exit', 'quit'}:
            break
        if not question:
            print("Assistant: veuillez saisir une question.")
            continue
        result = pipeline.run(question, conversation_id=args.conversation_id)
        print(f"Assistant: {result['answer']}")

parser=argparse.ArgumentParser(description='Structured multilingual RAG pipeline')
sub=parser.add_subparsers(dest='command',required=True)
p=sub.add_parser('ingest');p.add_argument('--input',default=str(RAW_DIR));p.set_defaults(func=cmd_ingest)
p=sub.add_parser('chunk');p.add_argument('--report',default=str(REPORT_FILE));p.add_argument('--output',default=str(CHUNKS_FILE));p.set_defaults(func=cmd_chunk)
for name,func in [('check',cmd_check),('embed',cmd_embed),('index',cmd_index)]:
    p=sub.add_parser(name);p.set_defaults(func=func)
p=sub.add_parser('search');p.add_argument('query');p.set_defaults(func=cmd_search)
p=sub.add_parser('ask');p.add_argument('query')
p.add_argument('--summary', default='')
p.add_argument('--keyword', action='append', default=[])
p.add_argument('--domain', default=None)
p.set_defaults(func=cmd_ask)
p=sub.add_parser('pipeline')
p.add_argument('--input',default=str(RAW_DIR))
p.add_argument('--question')
p.add_argument('--conversation-id', default='default')
p.add_argument('--domain', default=None)
p.set_defaults(func=cmd_pipeline)
p=sub.add_parser('chat')
p.add_argument('--conversation-id', default='default')
p.add_argument('--domain', default=None)
p.set_defaults(func=cmd_chat)
args=parser.parse_args(); args.func(args)
