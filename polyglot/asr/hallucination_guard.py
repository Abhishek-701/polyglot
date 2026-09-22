"""Hallucination guard. See SPEC.md Section 8.3.

Pure function over one decoded segment's stats: no EventLog access here (that
wiring — "log every suppression as an event with the reason" — happens where
this is called, once the real pipeline uses it in M4). The "segment lies
outside VAD speech" check from Section 8.3 also isn't here: it needs VAD
context this function doesn't have, and belongs to whatever wires ASR output
against VAD regions.
"""

from dataclasses import dataclass, field


@dataclass
class HallucinationGuardConfig:
    no_speech_prob_threshold: float = 0.6
    avg_logprob_threshold: float = -1.0
    compression_ratio_threshold: float = 2.4
    blocklist: list[str] = field(default_factory=list)


def check_hallucination(
    text: str,
    no_speech_prob: float,
    avg_logprob: float,
    compression_ratio: float,
    config: HallucinationGuardConfig,
) -> str | None:
    """Returns a suppression reason, or None if the segment should be kept."""
    if no_speech_prob > config.no_speech_prob_threshold:
        return "no_speech_prob"
    if avg_logprob < config.avg_logprob_threshold:
        return "avg_logprob"
    if compression_ratio > config.compression_ratio_threshold:
        return "compression_ratio"
    normalized = text.strip().lower()
    for phrase in config.blocklist:
        if phrase.lower() in normalized:
            return "blocklist"
    return None
