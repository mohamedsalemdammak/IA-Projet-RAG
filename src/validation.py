import json
from collections import Counter
from pathlib import Path


def validate_dataset(chunks_file, report_file, expected_chunks=None, expected_files=None):
    chunks = json.loads(Path(chunks_file).read_text(encoding='utf-8'))
    report = json.loads(Path(report_file).read_text(encoding='utf-8'))
    issues = []
    empty_chunks = [i for i, c in enumerate(chunks) if not c.get('text', '').strip()]
    missing_fields = [(i, f) for i, c in enumerate(chunks) for f in ('text', 'metadata') if f not in c]
    metadata_keys = Counter(k for c in chunks for k in c.get('metadata', {}).keys())
    files = {c.get('metadata', {}).get('source') or c.get('metadata', {}).get('filename') for c in chunks}
    files.discard(None)
    ocr_count = sum(bool(c.get('metadata', {}).get('ocr')) for c in chunks)
    if expected_chunks is not None and len(chunks) != expected_chunks:
        issues.append(f'Expected {expected_chunks} chunks, got {len(chunks)}')
    if expected_files is not None and len(files) != expected_files:
        issues.append(f'Expected {expected_files} files, got {len(files)}')
    if empty_chunks: issues.append(f'{len(empty_chunks)} empty chunks')
    if missing_fields: issues.append(f'{len(missing_fields)} missing required fields')
    return {
        'valid': not issues, 'chunks': len(chunks), 'unique_files': len(files),
        'empty_chunks': empty_chunks, 'missing_fields': missing_fields,
        'metadata_keys': dict(metadata_keys), 'ocr_chunks': ocr_count,
        'ingestion_report': report, 'issues': issues,
    }
