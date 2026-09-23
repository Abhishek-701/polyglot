"""Multilingual bridge: embed the original-language query directly with
BGE-M3, dense search. See SPEC.md Section 8.7.
"""

import asyncio

import numpy as np
from sentence_transformers import SentenceTransformer

from polyglot.core.types import Passage
from polyglot.retrieval.store import RetrievalStore


class MultilingualBridge:
    def __init__(self, store: RetrievalStore, embedder: SentenceTransformer) -> None:
        self._store = store
        self._embedder = embedder

    async def retrieve(self, query: str, lang: str, k: int) -> list[Passage]:
        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(None, self._embed, query)
        return self._store.search(embedding, k)

    def _embed(self, query: str) -> np.ndarray:
        return self._embedder.encode(query, normalize_embeddings=True)
