"""WER/CER metrics. See SPEC.md Section 11.3.

Per-language, split by code-switched vs not / speaker nativeness / recorded_by
in the full version (M7, once the golden set exists); M2 only needs a flat
per-language WER/CER over a FLEURS subset.
"""

import re
from dataclasses import dataclass

import jiwer

_PUNCTUATION_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)

# Languages with no whitespace between words. WER is not a meaningful metric
# here at all (there's no word boundary to score), and — the real bug this
# was written to fix — FLEURS' own Mandarin `transcription` field puts a
# space between every single character (e.g. "这 并 不 是"), while real ASR
# output has none. Collapsing multiple spaces to one (the space-delimited
# languages' normalization) leaves those spurious inter-character spaces in
# place, which misaligns the character-level edit distance and inflates CER
# for no real reason. Confirmed by hand: WER measured ~1.0 on a clip whose
# actual transcription was correct aside from one real error. Fixed by
# stripping ALL whitespace for these languages before computing CER, and not
# computing WER for them at all.
NO_WORD_BOUNDARY_LANGUAGES = frozenset({"zh"})


def normalize(text: str, strip_all_whitespace: bool = False) -> str:
    """Lowercase, strip punctuation, and either collapse whitespace to single
    spaces (space-delimited languages) or remove it entirely (languages with
    no word boundaries — see NO_WORD_BOUNDARY_LANGUAGES).
    """
    text = text.lower()
    text = _PUNCTUATION_RE.sub("", text)
    if strip_all_whitespace:
        return "".join(text.split())
    return " ".join(text.split())


@dataclass
class WerResult:
    lang: str
    n_clips: int
    wer: float | None  # None where WER isn't meaningful (see NO_WORD_BOUNDARY_LANGUAGES)
    cer: float


def compute_wer_cer(
    references: list[str], hypotheses: list[str], lang: str = "en"
) -> tuple[float | None, float]:
    strip_all = lang in NO_WORD_BOUNDARY_LANGUAGES
    norm_references = [normalize(r, strip_all_whitespace=strip_all) for r in references]
    norm_hypotheses = [normalize(h, strip_all_whitespace=strip_all) for h in hypotheses]

    cer = jiwer.cer(norm_references, norm_hypotheses)
    if strip_all:
        return None, cer
    wer = jiwer.wer(norm_references, norm_hypotheses)
    return wer, cer


def compute_wer_by_language(
    by_language: dict[str, list[tuple[str, str]]],
) -> list[WerResult]:
    """by_language: lang -> list of (reference, hypothesis) pairs."""
    results = []
    for lang, pairs in by_language.items():
        references = [ref for ref, _hyp in pairs]
        hypotheses = [hyp for _ref, hyp in pairs]
        wer, cer = compute_wer_cer(references, hypotheses, lang=lang)
        results.append(WerResult(lang=lang, n_clips=len(pairs), wer=wer, cer=cer))
    return results
