import json
from pathlib import Path
import faiss
import numpy as np


def build_faiss_index(embeddings_file, index_file):
    embeddings = np.load(embeddings_file).astype('float32')
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    Path(index_file).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_file))
    return index.ntotal, index.d


def load_store(index_file, chunks_file):
    index = faiss.read_index(str(index_file))
    chunks = json.loads(Path(chunks_file).read_text(encoding='utf-8'))
    return index, chunks
