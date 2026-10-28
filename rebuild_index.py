import json
from pathlib import Path

import faiss
import numpy as np

from src.embeddings import E5Embedder

CHUNKS_FILE = Path("data/processed/chunks.json")
EMBEDDINGS_FILE = Path("data/index/embeddings.npy")
FAISS_FILE = Path("data/index/faiss.index")


chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))

print(f"Chunks trouvés : {len(chunks)}")

texts = [chunk["text"].strip() for chunk in chunks]

embedder = E5Embedder()

embeddings = np.asarray(
    embedder.embed_documents(texts, batch_size=32),
    dtype="float32",
)

print(f"Embeddings générés : {embeddings.shape}")

if embeddings.shape[0] != len(chunks):
    raise ValueError(
        f"Mismatch : {embeddings.shape[0]} embeddings pour {len(chunks)} chunks"
    )

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

print(f"FAISS vectors : {index.ntotal}")

np.save(EMBEDDINGS_FILE, embeddings)
faiss.write_index(index, str(FAISS_FILE))

print("OK : index FAISS et embeddings reconstruits.")