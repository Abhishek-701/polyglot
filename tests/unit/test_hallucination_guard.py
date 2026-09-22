from polyglot.asr.hallucination_guard import HallucinationGuardConfig, check_hallucination


def test_keeps_normal_segment() -> None:
    config = HallucinationGuardConfig()
    assert check_hallucination("my flight was cancelled", 0.1, -0.3, 1.5, config) is None


def test_suppresses_high_no_speech_prob() -> None:
    config = HallucinationGuardConfig()
    assert check_hallucination("thank you", 0.9, -0.3, 1.5, config) == "no_speech_prob"


def test_suppresses_low_avg_logprob() -> None:
    config = HallucinationGuardConfig()
    assert check_hallucination("gibberish", 0.1, -2.0, 1.5, config) == "avg_logprob"


def test_suppresses_high_compression_ratio() -> None:
    config = HallucinationGuardConfig()
    assert check_hallucination("la la la la la la", 0.1, -0.3, 3.0, config) == "compression_ratio"


def test_suppresses_blocklist_phrase_case_insensitive() -> None:
    config = HallucinationGuardConfig(blocklist=["thank you for watching"])
    assert check_hallucination("Thank You For Watching!", 0.1, -0.3, 1.5, config) == "blocklist"


def test_no_speech_prob_checked_before_avg_logprob() -> None:
    config = HallucinationGuardConfig()
    # Both conditions independently trigger; no_speech_prob is checked first (SPEC.md 8.3 order).
    assert check_hallucination("x", 0.9, -2.0, 1.5, config) == "no_speech_prob"


def test_custom_thresholds() -> None:
    config = HallucinationGuardConfig(no_speech_prob_threshold=0.99)
    assert check_hallucination("ok", 0.9, -0.3, 1.5, config) is None
