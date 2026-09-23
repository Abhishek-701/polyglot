"""Hybrid bridge: reciprocal rank fusion of mt + multilingual, optional
rerank with bge-reranker-v2-m3. See SPEC.md Section 8.7.
"""

import asyncio

from sentence_transformers import CrossEncoder

from polyglot.core.types import Passage
from polyglot.retrieval.bridges.mt_bridge import MtBridge
from polyglot.retrieval.bridges.multilingual import MultilingualBridge

RRF_K = 60


def reciprocal_rank_fusion(rankings: list[list[Passage]], k: int = RRF_K) -> list[Passage]:
    scores: dict[str, float] = {}
    passages_by_id: dict[str, Passage] = {}
    for ranking in rankings:
        for rank, passage in enumerate(ranking):
            scores[passage.id] = scores.get(passage.id, 0.0) + 1.0 / (k + rank + 1)
            passages_by_id[passage.id] = passage
    ranked_ids = sorted(scores, key=lambda pid: scores[pid], reverse=True)
    return [passages_by_id[pid] for pid in ranked_ids]


class HybridBridge:
    def __init__(
        self,
        mt_bridge: MtBridge,
        multilingual_bridge: MultilingualBridge,
        reranker: CrossEncoder | None = None,
    ) -> None:
        self._mt = mt_bridge
        self._multilingual = multilingual_bridge
        self._reranker = reranker

    async def retrieve(self, query: str, lang: str, k: int) -> list[Passage]:
        fetch_k = max(k * 2, 20)
        mt_results, multilingual_results = await asyncio.gather(
            self._mt.retrieve(query, lang, fetch_k),
            self._multilingual.retrieve(query, lang, fetch_k),
        )
        fused = reciprocal_rank_fusion([mt_results, multilingual_results])[:fetch_k]

        if self._reranker is None or not fused:
            return fused[:k]

        loop = asyncio.get_running_loop()
        scores = await loop.run_in_executor(None, self._rerank, query, fused)
        reranked = [
            passage
            for _score, passage in sorted(
                zip(scores, fused, strict=True), key=lambda pair: pair[0], reverse=True
            )
        ]
        return reranked[:k]

    def _rerank(self, query: str, passages: list[Passage]) -> list[float]:
        pairs = [(query, passage.text) for passage in passages]
        return [float(score) for score in self._reranker.predict(pairs)]
