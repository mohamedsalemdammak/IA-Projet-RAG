import json
from pathlib import Path

from config import CHUNK_OVERLAP, CHUNK_SIZE
from src.ingestion import TOKEN_PATTERN, extract_document_metadata


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if chunk_size <= 0:
        raise ValueError('chunk_size must be greater than zero')
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError('overlap must be between zero and chunk_size - 1')

    tokens = list(TOKEN_PATTERN.finditer(text or ''))
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = text[tokens[start].start():tokens[end - 1].end()]
        chunks.append((chunk, end - start))
        if end == len(tokens):
            break
        start = end - overlap
    return chunks


def build_chunks(report_file, output_file, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    report = json.loads(Path(report_file).read_text(encoding='utf-8'))
    all_chunks = []

    for document in report.get('files', []):
        text = document.get('extracted_text') or ''
        if not text:
            continue

        relative_path = document.get('file', '')
        filename = document.get('document_name') or Path(relative_path).name
        metadata = {
            key: value
            for key, value in document.items()
            if key not in {'extracted_text', 'chunks'}
        }
        derived_metadata = extract_document_metadata(Path(filename), relative_path, text)
        derived_metadata.update(metadata)
        derived_metadata.setdefault('file', relative_path)
        derived_metadata.setdefault('source', relative_path)
        derived_metadata.setdefault('filename', filename)
        derived_metadata.setdefault('extension', Path(filename).suffix.lower())

        for chunk_index, (chunk, token_count) in enumerate(
            chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        ):
            chunk_metadata = {
                **derived_metadata,
                'chunk_index': chunk_index,
                'chunk_id': chunk_index,
                'token_count': token_count,
            }
            all_chunks.append({'text': chunk, 'metadata': chunk_metadata})

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    return all_chunks