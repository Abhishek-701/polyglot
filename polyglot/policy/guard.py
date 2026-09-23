"""Inline grounding guard. See SPEC.md Section 8.9: "inline check that the
response does not state refund amounts or deadlines absent from retrieved
passages; if it fails, fall back to a safe hedge."

Deliberately a cheap regex/string-membership check, not another model call:
extracts money/percentage/day-count figures the assistant claimed, and fails
if any of them don't appear verbatim in the retrieved passage text. This
catches fabricated numbers, not fabricated reasoning in general — a known,
documented limitation, not a bug (a full NLI-based grounding check would fit
the same call site but adds another model in the latency path SPEC.md is
trying to keep out of the generate step).
"""

import re

from polyglot.core.types import Passage

SAFE_HEDGE = (
    "I don't have that exact figure confirmed on file. Let me get you accurate "
    "details rather than guess — please contact support directly for the specific amount."
)

_FIGURE_RE = re.compile(
    r"\$\s?\d+(?:[.,]\d+)?|€\s?\d+(?:[.,]\d+)?|\d+\s?%|\d+\s?(?:days?|hours?|hrs?)\b",
    re.IGNORECASE,
)


def check_guard(assistant_text: str, passages: list[Passage]) -> str | None:
    """Returns a safe-hedge replacement string if assistant_text states a
    figure not grounded in the retrieved passages, else None (response is
    fine as-is).
    """
    claimed_figures = {match.group().strip() for match in _FIGURE_RE.finditer(assistant_text)}
    if not claimed_figures:
        return None

    passages_text = " ".join(p.text for p in passages)
    ungrounded = [figure for figure in claimed_figures if figure not in passages_text]
    return SAFE_HEDGE if ungrounded else None
