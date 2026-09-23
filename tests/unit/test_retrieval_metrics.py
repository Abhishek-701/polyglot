from eval.metrics.retrieval import RetrievalResult, recall_at_k, reciprocal_rank, summarize


def _result(
    expected: list[str], retrieved: list[str], lang: str = "en", bridge: str = "mt"
) -> RetrievalResult:
    return RetrievalResult(
        query_id="q1",
        lang=lang,
        bridge=bridge,
        expected_passage_ids=set(expected),
        retrieved_passage_ids=retrieved,
    )


def test_recall_at_k_hit_and_miss_by_k() -> None:
    result = _result(["p2"], ["p1", "p2", "p3"])
    assert recall_at_k(result, 1) == 0.0
    assert recall_at_k(result, 5) == 1.0


def test_recall_at_k_miss() -> None:
    result = _result(["p9"], ["p1", "p2", "p3"])
    assert recall_at_k(result, 10) == 0.0


def test_recall_at_k_no_expected_passages() -> None:
    result = _result([], ["p1"])
    assert recall_at_k(result, 5) == 0.0


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(_result(["p3"], ["p1", "p2", "p3"])) == 1 / 3
    assert reciprocal_rank(_result(["p1"], ["p1", "p2"])) == 1.0
    assert reciprocal_rank(_result(["p9"], ["p1"])) == 0.0


def test_summarize_groups_by_lang_and_bridge() -> None:
    results = [
        _result(["p1"], ["p1"], lang="en", bridge="mt"),
        _result(["p9"], ["p1"], lang="en", bridge="mt"),
        _result(["p1"], ["p1"], lang="es", bridge="mt"),
    ]
    reports = summarize(results)
    by_key = {(r.lang, r.bridge): r for r in reports}
    assert by_key[("en", "mt")].n_queries == 2
    assert by_key[("en", "mt")].recall_at_1 == 0.5
    assert by_key[("es", "mt")].recall_at_1 == 1.0
