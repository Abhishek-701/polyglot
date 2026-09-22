"""WER/CER metrics. See SPEC.md Section 11.3.

Per-language, split by code-switched vs not / speaker nativeness / recorded_by
in the full version (M7, once the golden set exists); M2 only needs a flat
per-language WER/CER over a FLEURS subset.
"""

import re
from dataclasses import dataclass

import jiwer

_PUNCTUATION_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    FLEURS' own `transcription` field is already normalized this way;
    Whisper's raw output isn't, so without this WER would mostly measure
    casing/punctuation differences instead of transcription accuracy.
    """
    text = text.lower()
    text = _PUNCTUATION_RE.sub("", text)
    return " ".join(text.split())


@dataclass
class WerResult:
    lang: str
    n_clips: int
    wer: float
    cer: float


def compute_wer_cer(references: list[str], hypotheses: list[str]) -> tuple[float, float]:
    norm_references = [normalize(r) for r in references]
    norm_hypotheses = [normalize(h) for h in hypotheses]
    wer = jiwer.wer(norm_references, norm_hypotheses)
    cer = jiwer.cer(norm_references, norm_hypotheses)
    return wer, cer


def compute_wer_by_language(
    by_language: dict[str, list[tuple[str, str]]],
) -> list[WerResult]:
    """by_language: lang -> list of (reference, hypothesis) pairs."""
    results = []
    for lang, pairs in by_language.items():
        references = [ref for ref, _hyp in pairs]
        hypotheses = [hyp for _ref, hyp in pairs]
        wer, cer = compute_wer_cer(references, hypotheses)
        results.append(WerResult(lang=lang, n_clips=len(pairs), wer=wer, cer=cer))
    return results
