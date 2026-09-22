"""Hallucination guard metrics. See SPEC.md Section 11.3.

Non-empty output rate on silence/noise clips (target: at most 1%, SPEC.md
Section 3), and a breakdown of suppressions by reason.
"""

from dataclasses import dataclass


@dataclass
class ClipResult:
    path: str
    category: str  # "silence" | "noise"
    kept_text: str
    suppressed_reason: str | None


@dataclass
class HallucinationReport:
    category: str
    n_clips: int
    n_non_empty: int
    non_empty_rate: float
    suppressions_by_reason: dict[str, int]


def summarize(results: list[ClipResult]) -> list[HallucinationReport]:
    by_category: dict[str, list[ClipResult]] = {}
    for result in results:
        by_category.setdefault(result.category, []).append(result)

    reports = []
    for category, clips in sorted(by_category.items()):
        non_empty = [c for c in clips if c.kept_text.strip()]
        reasons: dict[str, int] = {}
        for clip in clips:
            if clip.suppressed_reason is not None:
                reasons[clip.suppressed_reason] = reasons.get(clip.suppressed_reason, 0) + 1
        reports.append(
            HallucinationReport(
                category=category,
                n_clips=len(clips),
                n_non_empty=len(non_empty),
                non_empty_rate=len(non_empty) / len(clips) if clips else 0.0,
                suppressions_by_reason=reasons,
            )
        )
    return reports
