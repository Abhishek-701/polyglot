# ADR 002: pgvector for the corpus store

## Status

Accepted (M3).

## Context

SPEC.md Section 2 sizes the knowledge base at "a few thousand chunks" (5-10
public policy documents). SPEC.md Section 8.7 asks for pgvector with an HNSW
index and for this ADR to document why pgvector suffices at that scale.

## Decision

Use Postgres + pgvector (`polyglot/retrieval/store.py`) with an HNSW index
on cosine distance, rather than a dedicated vector database (Qdrant,
Weaviate, Milvus, etc.).

## Why this is enough at this scale

- A few thousand vectors at 1024 dimensions (BGE-M3) is well within HNSW's
  comfortable range on a single node — sub-millisecond ANN search, no
  sharding or distributed index needed.
- Postgres already exists in the stack (session state references, and it's
  the natural place for `documents`/`chunks` metadata anyway) — no extra
  service to run, monitor, or back up.
- pgvector's exact search (no index) would already be fast enough at this
  size; HNSW is added for headroom, not because exact search would be a
  bottleneck.
- A dedicated vector DB earns its complexity at millions of vectors with
  update-heavy workloads and horizontal scaling needs — none of which apply
  here (the KB is ~5-10 documents, updated by a manual `make ingest` run,
  not a continuous stream).

## Consequences

- If the KB ever grows to cover many more documents/languages (well beyond
  SPEC.md's stated scope), HNSW recall/latency and index build time should
  be re-checked before assuming this still holds.
- `RetrievalStore` (`polyglot/retrieval/store.py`) is the only place that
  knows this is Postgres; bridges and the `Retriever` protocol only see
  `Passage` objects, so swapping the store later doesn't touch bridge code.
