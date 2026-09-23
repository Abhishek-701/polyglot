"""Retrieval metrics. See SPEC.md Section 11.3: recall@1/5/10, MRR, per
language, per bridge.
"""

from dataclasses import dataclass


@dataclass
class RetrievalResult:
    query_id: str
    lang: str
    bridge: str
    expected_passage_ids: set[str]
    retrieved_passage_ids: list[str]  # ranked, best first


def recall_at_k(result: RetrievalResult, k: int) -> float:
    if not result.expected_passage_ids:
        return 0.0
    retrieved_top_k = set(result.retrieved_passage_ids[:k])
    return 1.0 if retrieved_top_k & result.expected_passage_ids else 0.0


def reciprocal_rank(result: RetrievalResult) -> float:
    for rank, passage_id in enumerate(result.retrieved_passage_ids, start=1):
        if passage_id in result.expected_passage_ids:
            return 1.0 / rank
    return 0.0


@dataclass
class RetrievalReport:
    lang: str
    bridge: str
    n_queries: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float


def summarize(results: list[RetrievalResult]) -> list[RetrievalReport]:
    by_group: dict[tuple[str, str], list[RetrievalResult]] = {}
    for result in results:
        by_group.setdefault((result.lang, result.bridge), []).append(result)

    reports = []
    for (lang, bridge), group in sorted(by_group.items()):
        n = len(group)
        reports.append(
            RetrievalReport(
                lang=lang,
                bridge=bridge,
                n_queries=n,
                recall_at_1=sum(recall_at_k(r, 1) for r in group) / n,
                recall_at_5=sum(recall_at_k(r, 5) for r in group) / n,
                recall_at_10=sum(recall_at_k(r, 10) for r in group) / n,
                mrr=sum(reciprocal_rank(r) for r in group) / n,
            )
        )
    return reports
