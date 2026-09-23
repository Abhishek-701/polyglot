from pathlib import Path

from eval.golden.schema import GoldenItem, Speaker
from eval.golden.validate import check_coverage, load_and_validate


def _item(lang: str, code_switched: bool = False, answerable: bool = True) -> GoldenItem:
    return GoldenItem(
        id=f"{lang}-{code_switched}-{answerable}-x",
        lang=lang,
        audio_path=f"eval/golden/audio/{lang}.wav",
        reference_transcript="sample",
        code_switched=code_switched,
        speaker=Speaker(native_lang=lang, recorded_by="human"),
        question_type="refund",
        expected_doc_ids=["doc1"],
        expected_passage_ids=["doc1#0"],
        reference_answer="answer",
        answerable=answerable,
    )


def test_load_and_validate_reports_schema_errors(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text('{"not": "a golden item"}\n', encoding="utf-8")
    outcome = load_and_validate(path)
    assert outcome.valid_items == []
    assert len(outcome.schema_errors) == 1


def test_load_and_validate_accepts_valid_items(tmp_path: Path) -> None:
    item = _item("en")
    path = tmp_path / "golden.jsonl"
    path.write_text(item.model_dump_json() + "\n", encoding="utf-8")
    outcome = load_and_validate(path)
    assert outcome.schema_errors == []
    assert len(outcome.valid_items) == 1


def test_check_coverage_fails_below_targets() -> None:
    items = [_item("en")]
    lines = check_coverage(items)
    assert any("FAIL" in line and "total items 1" in line for line in lines)


def test_check_coverage_passes_when_targets_met() -> None:
    # 4 languages x 50 items = 200 total; es/hi get >=20% code-switched;
    # overall >=15% unanswerable.
    items = []
    for lang in ("en", "es", "hi", "zh"):
        for i in range(50):
            code_switched = lang in ("es", "hi") and i < 15  # 30% of 50
            answerable = i >= 8  # 84% answerable -> 16% unanswerable
            items.append(_item(lang, code_switched=code_switched, answerable=answerable))

    lines = check_coverage(items)
    assert all("FAIL" not in line for line in lines), lines
