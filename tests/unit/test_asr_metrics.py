"""eval/metrics/asr.py tests. The Mandarin case exists specifically because
of a real bug found in production use: FLEURS' Mandarin reference text puts
a space between every character, while real ASR output has none — naive
whitespace-collapsing normalization leaves those spurious spaces in place
and badly inflates CER (word-level WER was measuring ~1.0 on a clip that
was actually almost entirely correct).
"""

from eval.metrics.asr import compute_wer_by_language, compute_wer_cer, normalize


def test_normalize_lowercases_and_strips_punctuation() -> None:
    assert normalize("Hello, World!") == "hello world"


def test_normalize_collapses_whitespace_by_default() -> None:
    assert normalize("hello   world") == "hello world"


def test_normalize_strips_all_whitespace_when_requested() -> None:
    assert normalize("这 并 不 是", strip_all_whitespace=True) == "这并不是"


def test_wer_cer_english_unaffected() -> None:
    wer, cer = compute_wer_cer(["hello world"], ["hello world"], lang="en")
    assert wer == 0.0
    assert cer == 0.0


def test_mandarin_returns_no_wer() -> None:
    wer, _cer = compute_wer_cer(["这并不是"], ["这并不是"], lang="zh")
    assert wer is None


def test_mandarin_cer_ignores_fleurs_style_spacing() -> None:
    # Same content, reference has FLEURS-style inter-character spaces,
    # hypothesis (real ASR output) has none — should be a perfect match,
    # not inflated by the spacing difference.
    reference = "这 并 不 是 告 别"
    hypothesis = "这并不是告别"
    _wer, cer = compute_wer_cer([reference], [hypothesis], lang="zh")
    assert cer == 0.0


def test_mandarin_cer_still_detects_real_errors() -> None:
    reference = "这 并 不 是 告 别"
    hypothesis = "这并不是错误"  # last two characters wrong
    _wer, cer = compute_wer_cer([reference], [hypothesis], lang="zh")
    assert cer > 0.0


def test_compute_wer_by_language_mixes_languages_correctly() -> None:
    results = compute_wer_by_language(
        {
            "en": [("hello world", "hello world")],
            "zh": [("这 并 不 是", "这并不是")],
        }
    )
    by_lang = {r.lang: r for r in results}
    assert by_lang["en"].wer == 0.0
    assert by_lang["zh"].wer is None
    assert by_lang["zh"].cer == 0.0
