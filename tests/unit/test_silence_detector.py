from polyglot.turn.silence_detector import SilenceTurnDetector


def test_below_threshold_does_not_commit() -> None:
    detector = SilenceTurnDetector(min_silence_ms=500)
    assert detector.end_of_turn_prob("hi", "en", 100, []) == 0.0


def test_at_threshold_commits() -> None:
    detector = SilenceTurnDetector(min_silence_ms=500)
    assert detector.end_of_turn_prob("hi", "en", 500, []) == 1.0


def test_above_threshold_commits() -> None:
    detector = SilenceTurnDetector(min_silence_ms=500)
    assert detector.end_of_turn_prob("hi", "en", 900, []) == 1.0


def test_custom_threshold() -> None:
    detector = SilenceTurnDetector(min_silence_ms=200)
    assert detector.end_of_turn_prob("hi", "en", 150, []) == 0.0
    assert detector.end_of_turn_prob("hi", "en", 250, []) == 1.0
