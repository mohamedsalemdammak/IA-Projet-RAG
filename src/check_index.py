from pathlib import Path
import sys
import json
import numpy as np
import faiss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FAISS_FILE, CHUNKS_FILE, EMBEDDINGS_FILE

print("========== RAG INDEX DIAGNOSTIC ==========")

# 1. FAISS
faiss_path = Path(FAISS_FILE)

if faiss_path.exists():
    index = faiss.read_index(str(faiss_path))
    print(f"FAISS file       : {faiss_path}")
    print(f"FAISS vectors    : {index.ntotal}")
    print(f"FAISS dimension  : {index.d}")
else:
    index = None
    print("FAISS file       : ABSENT")


# 2. CHUNKS
chunks_path = Path(CHUNKS_FILE)

if chunks_path.exists():
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Chunks file      : {chunks_path}")
    print(f"Chunks count     : {len(chunks)}")
else:
    chunks = []
    print("Chunks file      : ABSENT")


# 3. EMBEDDINGS
embeddings_path = Path(EMBEDDINGS_FILE)

if embeddings_path.exists():
    embeddings = np.load(embeddings_path)

    print(f"Embeddings file  : {embeddings_path}")
    print(f"Embeddings shape : {embeddings.shape}")
else:
    embeddings = None
    print("Embeddings file  : ABSENT")


print("\n========== SYNCHRONIZATION ==========")

if index is not None:
    print(
        f"FAISS vs Chunks      : "
        f"{index.ntotal} == {len(chunks)} -> "
        f"{'OK' if index.ntotal == len(chunks) else 'ERROR'}"
    )

if embeddings is not None:
    print(
        f"Embeddings vs Chunks : "
        f"{embeddings.shape[0]} == {len(chunks)} -> "
        f"{'OK' if embeddings.shape[0] == len(chunks) else 'ERROR'}"
    )

if index is not None and embeddings is not None:
    print(
        f"FAISS vs Embeddings  : "
        f"{index.ntotal} == {embeddings.shape[0]} -> "
        f"{'OK' if index.ntotal == embeddings.shape[0] else 'ERROR'}"
    )

print("\n========== RESULT ==========")

if (
    index is not None
    and embeddings is not None
    and index.ntotal == len(chunks)
    and embeddings.shape[0] == len(chunks)
):
    print("OK - Les trois fichiers sont synchronisés.")
else:
    print("PROBLEME - Les fichiers sont désynchronisés.")