"""Corpus ingestion. See SPEC.md Section 10.1.

Idempotent: re-running with an unchanged source makes no changes (content
hash comparison, stored per document). config/sources.yaml is filled in by
the human — CLAUDE.md rule 6: never invent knowledge-base URLs, so this
module refuses to run against an empty/missing sources file rather than
making something up.
"""

import hashlib
import os
from pathlib import Path

import requests
import yaml
from sentence_transformers import SentenceTransformer

from polyglot.retrieval.chunking import chunk_section, get_tokenizer
from polyglot.retrieval.extract import extract_sections
from polyglot.retrieval.store import RetrievalStore

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DSN = "postgresql://polyglot:polyglot@localhost:5433/polyglot"


def load_sources(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("documents", [])


def fetch(url: str) -> tuple[bytes, str]:
    response = requests.get(url, timeout=30, headers={"User-Agent": "polyglot-ingest/0.1"})
    response.raise_for_status()
    return response.content, response.headers.get("content-type", "")


def ingest_document(
    store: RetrievalStore, embedder: SentenceTransformer, doc_id: str, url: str
) -> bool:
    """Returns True if the document was (re)ingested, False if unchanged."""
    content, content_type = fetch(url)
    content_hash = hashlib.sha256(content).hexdigest()

    if store.document_hash(doc_id) == content_hash:
        return False

    sections = extract_sections(content, content_type, url)
    tokenizer = get_tokenizer()
    chunks = [
        chunk
        for section_name, section_text in sections
        for chunk in chunk_section(section_text, section_name, tokenizer=tokenizer)
    ]

    embeddings = (
        embedder.encode([c.text for c in chunks], normalize_embeddings=True) if chunks else []
    )
    rows = [
        (chunk.section, chunk.chunk_index, chunk.text, embedding)
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]
    store.upsert_document(doc_id, url, content_hash, rows)
    return True


def main() -> None:
    sources_path = REPO_ROOT / "config" / "sources.yaml"
    sources = load_sources(sources_path)
    if not sources:
        raise SystemExit(
            f"{sources_path} has no documents listed. This is filled in by the human "
            "(CLAUDE.md rule 6) — Claude Code does not invent knowledge-base URLs."
        )

    dsn = os.environ.get("POLYGLOT_DB_DSN", DEFAULT_DSN)
    store = RetrievalStore(dsn)
    embedder = SentenceTransformer("BAAI/bge-m3")

    for doc in sources:
        changed = ingest_document(store, embedder, doc["doc_id"], doc["url"])
        print(f"{doc['doc_id']}: {'ingested' if changed else 'unchanged'}")

    store.close()


if __name__ == "__main__":
    main()
