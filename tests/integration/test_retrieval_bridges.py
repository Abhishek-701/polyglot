"""Synthetic end-to-end retrieval test: proves the ingest -> store -> bridge
pipeline works mechanically with a tiny in-test corpus. These are NOT real
KB recall numbers — that needs config/sources.yaml (human-provided,
CLAUDE.md rule 6) and a real `make ingest` run; see eval/retrieval_eval.py
for that once sources.yaml exists.

Requires Postgres reachable (docker compose up postgres) and the BGE-M3 /
NLLB / reranker models downloaded; marked @pytest.mark.network. Skips
gracefully if Postgres isn't reachable rather than failing.
"""

import os
import uuid
from collections.abc import AsyncIterator

import pytest
from sentence_transformers import CrossEncoder, SentenceTransformer

from polyglot.retrieval.bridges.hybrid import HybridBridge
from polyglot.retrieval.bridges.mt_bridge import MtBridge, Translator
from polyglot.retrieval.bridges.multilingual import MultilingualBridge
from polyglot.retrieval.chunking import chunk_section
from polyglot.retrieval.store import RetrievalStore

pytestmark = pytest.mark.network

DSN = os.environ.get(
    "POLYGLOT_TEST_DB_DSN", "postgresql://polyglot:polyglot@localhost:5433/polyglot"
)

REFUND_POLICY_TEXT = (
    "Passengers whose flights are cancelled by the airline are entitled to a full refund "
    "of the ticket price, including any optional fees, if they choose not to be rebooked "
    "on an alternative flight."
)
QUERY_ES = "¿Puedo obtener un reembolso por mi vuelo cancelado?"


@pytest.fixture(scope="module")
def populated_store() -> AsyncIterator[tuple[RetrievalStore, SentenceTransformer, str]]:
    try:
        store = RetrievalStore(DSN)
    except Exception as exc:
        pytest.skip(f"Postgres not reachable at {DSN}: {exc}")

    embedder = SentenceTransformer("BAAI/bge-m3")
    doc_id = f"test-doc-{uuid.uuid4().hex[:8]}"
    chunks = chunk_section(
        REFUND_POLICY_TEXT, "refunds", min_tokens=5, max_tokens=50, overlap_tokens=5
    )
    embeddings = embedder.encode([c.text for c in chunks], normalize_embeddings=True)
    rows = [
        (chunk.section, chunk.chunk_index, chunk.text, embedding)
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]
    store.upsert_document(doc_id, "https://example.com/refunds", "testhash", rows)

    yield store, embedder, doc_id

    store.delete_document(doc_id)
    store.close()


async def test_multilingual_bridge_finds_relevant_passage(
    populated_store: tuple[RetrievalStore, SentenceTransformer, str],
) -> None:
    store, embedder, doc_id = populated_store
    bridge = MultilingualBridge(store, embedder)
    results = await bridge.retrieve(QUERY_ES, "es", k=3)
    assert results
    assert any(r.doc_id == doc_id for r in results)


async def test_mt_bridge_finds_relevant_passage(
    populated_store: tuple[RetrievalStore, SentenceTransformer, str],
) -> None:
    store, embedder, doc_id = populated_store
    bridge = MtBridge(store, embedder, Translator())
    results = await bridge.retrieve(QUERY_ES, "es", k=3)
    assert results
    assert any(r.doc_id == doc_id for r in results)


async def test_hybrid_bridge_with_rerank_finds_relevant_passage(
    populated_store: tuple[RetrievalStore, SentenceTransformer, str],
) -> None:
    store, embedder, doc_id = populated_store
    mt = MtBridge(store, embedder, Translator())
    multilingual = MultilingualBridge(store, embedder)
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
    bridge = HybridBridge(mt, multilingual, reranker=reranker)

    results = await bridge.retrieve(QUERY_ES, "es", k=3)
    assert results
    assert any(r.doc_id == doc_id for r in results)
