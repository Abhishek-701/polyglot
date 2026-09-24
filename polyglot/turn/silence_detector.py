"""Naive turn detector. See SPEC.md Section 8.5: "commit the turn after
min_silence_ms (default 500ms) of VAD silence."

The naive baseline for the naive/tuned ablation — semantic_detector.py
(ONNX turn model) is M5 work and must not change this file's behavior.
"""

from polyglot.core.types import Message


class SilenceTurnDetector:
    def __init__(self, min_silence_ms: int = 500) -> None:
        self.min_silence_ms = min_silence_ms

    def end_of_turn_prob(
        self, text: str, lang: str, trailing_silence_ms: int, history: list[Message]
    ) -> float:
        return 1.0 if trailing_silence_ms >= self.min_silence_ms else 0.0
