"""Silero VAD tests. No network needed: silero-vad ships its model weights
in the pip package. Only the "stays silent on non-speech" property is tested
here with synthetic noise — a sine tone or noise floor isn't real speech, so
it can't reliably exercise the speech_start path. See
tests/integration/test_real_audio_components.py for that, against real
downloaded speech.
"""

import numpy as np
import pytest

from polyglot.audio.frames import float32_to_pcm16
from polyglot.audio.vad import SileroVAD
from polyglot.core.types import AudioFrame


def _frame(samples: np.ndarray, t_ms: int) -> AudioFrame:
    return AudioFrame(pcm=float32_to_pcm16(samples), t_ms=t_ms)


def test_near_silence_produces_no_events() -> None:
    vad = SileroVAD()
    rng = np.random.default_rng(0)
    silence = rng.normal(0, 0.001, 16000).astype(np.float32)  # 1s of a quiet noise floor

    events = []
    step = 512
    for i in range(0, len(silence) - step + 1, step):
        events.extend(vad.process(_frame(silence[i : i + step], t_ms=round(i / 16000 * 1000))))

    assert events == []


def test_rejects_frame_at_wrong_sample_rate() -> None:
    vad = SileroVAD(sample_rate=16000)
    bad_frame = AudioFrame(pcm=b"\x00\x00" * 512, sample_rate=8000, t_ms=0)
    with pytest.raises(ValueError, match="16000 Hz"):
        vad.process(bad_frame)


def test_reset_clears_internal_state() -> None:
    vad = SileroVAD()
    vad.process(_frame(np.zeros(512, dtype=np.float32), t_ms=0))
    vad.reset()
    assert len(vad._buffer) == 0
    assert vad._next_window_t_ms is None
