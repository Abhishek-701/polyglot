"""Silero VAD. See SPEC.md Section 8.1.

Verified against installed silero-vad==6.2.2: the loaded JIT model raises
ValueError below its minimum chunk size (sample_rate / window_samples must be
<= 31.25, i.e. exactly 512 samples at 16kHz / 256 at 8kHz — a 32ms/32ms
window). We reimplement the speech_start/speech_end state machine ourselves
(rather than using silero_vad.VADIterator) so min_speech_ms is enforced in
addition to min_silence_ms, per SPEC.md's own config (VADIterator only
exposes min_silence_duration_ms).
"""

import torch
from silero_vad import load_silero_vad

from polyglot.audio.frames import SampleBuffer, pcm16_to_float32
from polyglot.core.types import AudioFrame, VADEvent

_SAMPLE_RATE_TO_WINDOW = {16000: 512, 8000: 256}


class SileroVAD:
    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 100,
        sample_rate: int = 16000,
    ) -> None:
        if sample_rate not in _SAMPLE_RATE_TO_WINDOW:
            raise ValueError(f"SileroVAD supports 8000/16000 Hz, got {sample_rate}")
        self._model = load_silero_vad(onnx=False)
        self._sample_rate = sample_rate
        self._window_samples = _SAMPLE_RATE_TO_WINDOW[sample_rate]
        self._window_ms = round(self._window_samples / sample_rate * 1000)
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self._buffer = SampleBuffer()
        self._next_window_t_ms: int | None = None
        self._in_speech = False
        self._speech_run_ms = 0
        self._silence_run_ms = 0
        self._model.reset_states()

    def reset(self) -> None:
        self._model.reset_states()
        self._buffer = SampleBuffer()
        self._next_window_t_ms = None
        self._in_speech = False
        self._speech_run_ms = 0
        self._silence_run_ms = 0

    def process(self, frame: AudioFrame) -> list[VADEvent]:
        if frame.sample_rate != self._sample_rate:
            raise ValueError(
                f"SileroVAD configured for {self._sample_rate} Hz, "
                f"got a frame at {frame.sample_rate} Hz; resample first"
            )
        if self._next_window_t_ms is None:
            self._next_window_t_ms = frame.t_ms

        self._buffer.push(pcm16_to_float32(frame.pcm))
        events: list[VADEvent] = []

        while True:
            window = self._buffer.pop_window(self._window_samples)
            if window is None:
                break
            window_t_ms = self._next_window_t_ms
            self._next_window_t_ms += self._window_ms

            with torch.no_grad():
                tensor = torch.from_numpy(window).unsqueeze(0)
                prob = float(self._model(tensor, self._sample_rate).item())

            if prob >= self.threshold:
                self._speech_run_ms += self._window_ms
                self._silence_run_ms = 0
                if not self._in_speech and self._speech_run_ms >= self.min_speech_ms:
                    self._in_speech = True
                    events.append(VADEvent(kind="speech_start", t_ms=window_t_ms, prob=prob))
            else:
                self._silence_run_ms += self._window_ms
                self._speech_run_ms = 0
                if self._in_speech and self._silence_run_ms >= self.min_silence_ms:
                    self._in_speech = False
                    events.append(VADEvent(kind="speech_end", t_ms=window_t_ms, prob=prob))

        return events
