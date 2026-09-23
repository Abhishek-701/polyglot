from polyglot.core.types import Passage
from polyglot.policy.guard import SAFE_HEDGE, check_guard


def _passage(text: str) -> Passage:
    return Passage(id="p1", doc_id="doc1", text=text, source_url="https://x", score=0.9)


def test_no_figures_passes() -> None:
    assert check_guard("You may be entitled to a refund.", []) is None


def test_grounded_figure_passes() -> None:
    passages = [_passage("Compensation is $400 for delays over 3 hours.")]
    assert check_guard("You are entitled to $400 in compensation.", passages) is None


def test_ungrounded_figure_triggers_hedge() -> None:
    passages = [_passage("Compensation is $400 for delays over 3 hours.")]
    result = check_guard("You are entitled to $999 in compensation.", passages)
    assert result == SAFE_HEDGE


def test_ungrounded_figure_with_no_passages_triggers_hedge() -> None:
    assert check_guard("You get 50% off your next flight.", []) == SAFE_HEDGE


def test_percentage_and_day_count_detected() -> None:
    passages = [_passage("You must file within 30 days for a full refund.")]
    assert check_guard("You have 30 days to file.", passages) is None
    assert check_guard("You have 45 days to file.", passages) == SAFE_HEDGE
