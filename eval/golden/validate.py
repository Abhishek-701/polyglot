"""Validates eval/golden/golden.jsonl against its schema and SPEC.md Section
10.3's coverage targets (200 items total, >=40 per language, >=20%
code-switched for es/hi, >=15% unanswerable). Coverage checks are printed as
pass/fail, not enforced as a hard gate — the human still finalizes the set.

Usage: uv run python -m eval.golden.validate
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from eval.golden.schema import GoldenItem

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GOLDEN_PATH = REPO_ROOT / "eval" / "golden" / "golden.jsonl"

CODE_SWITCH_LANGS = ("es", "hi")
MIN_TOTAL = 200
MIN_PER_LANGUAGE = 40
MIN_CODE_SWITCHED_SHARE = 0.20
MIN_UNANSWERABLE_SHARE = 0.15


@dataclass
class ValidationOutcome:
    valid_items: list[GoldenItem]
    schema_errors: list[str]


def load_and_validate(path: Path) -> ValidationOutcome:
    valid_items: list[GoldenItem] = []
    schema_errors: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                valid_items.append(GoldenItem.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValidationError) as exc:
                schema_errors.append(f"line {line_number}: {exc}")
    return ValidationOutcome(valid_items=valid_items, schema_errors=schema_errors)


def check_coverage(items: list[GoldenItem]) -> list[str]:
    """Returns a list of human-readable pass/fail lines, SPEC.md Section 10.3 targets."""
    lines = []
    total = len(items)
    lines.append(
        f"{'PASS' if total >= MIN_TOTAL else 'FAIL'}: total items {total} (target >= {MIN_TOTAL})"
    )

    by_lang: dict[str, list[GoldenItem]] = {}
    for item in items:
        by_lang.setdefault(item.lang, []).append(item)
    for lang, lang_items in sorted(by_lang.items()):
        n = len(lang_items)
        lines.append(
            f"{'PASS' if n >= MIN_PER_LANGUAGE else 'FAIL'}: {lang} has {n} items "
            f"(target >= {MIN_PER_LANGUAGE})"
        )

    for lang in CODE_SWITCH_LANGS:
        lang_items = by_lang.get(lang, [])
        if not lang_items:
            lines.append(f"FAIL: {lang} has no items, cannot check code-switch share")
            continue
        share = sum(1 for i in lang_items if i.code_switched) / len(lang_items)
        status = "PASS" if share >= MIN_CODE_SWITCHED_SHARE else "FAIL"
        lines.append(
            f"{status}: {lang} code-switched share {share:.1%} (target >= "
            f"{MIN_CODE_SWITCHED_SHARE:.0%})"
        )

    if total:
        unanswerable_share = sum(1 for i in items if not i.answerable) / total
        status = "PASS" if unanswerable_share >= MIN_UNANSWERABLE_SHARE else "FAIL"
        lines.append(
            f"{status}: unanswerable share {unanswerable_share:.1%} (target >= "
            f"{MIN_UNANSWERABLE_SHARE:.0%})"
        )

    return lines


def main() -> None:
    if not GOLDEN_PATH.exists():
        print(f"{GOLDEN_PATH} does not exist yet — nothing to validate.")
        sys.exit(1)

    outcome = load_and_validate(GOLDEN_PATH)

    if outcome.schema_errors:
        print(f"{len(outcome.schema_errors)} schema errors:")
        for error in outcome.schema_errors:
            print(f"  {error}")

    print(f"\n{len(outcome.valid_items)} valid items. Coverage against SPEC.md Section 10.3:")
    for line in check_coverage(outcome.valid_items):
        print(f"  {line}")

    if outcome.schema_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
