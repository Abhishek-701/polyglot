"""faster-whisper streaming ASR engine. See SPEC.md Section 8.2.

Verified against installed faster-whisper==1.2.1: WhisperModel.transcribe()
accepts a float32 np.ndarray directly (no temp file needed), and its default
compression_ratio_threshold/log_prob_threshold/no_speech_threshold (2.4,
-1.0, 0.6) are exactly SPEC.md Section 8.3's hallucination guard defaults —
those come from faster-whisper's own built-in filtering, separate from our
own check_hallucination() applied per returned segment below.

Re-decodes the buffered audio every asr_step_ms (default 200ms), matching
Section 8.2. WhisperModel.transcribe() is a blocking CPU call, so it runs in
the default executor per CLAUDE.md's "nothing blocking on the audio path"
rule.
"""

import asyncio
from collections.abc import AsyncIterator

import numpy as np
from faster_whisper import WhisperModel

from polyglot.asr.hallucination_guard import HallucinationGuardConfig, check_hallucination
from polyglot.asr.local_agreement import LocalAgreement
from polyglot.audio.frames import pcm16_to_float32
from polyglot.core.types import AudioFrame, TranscriptPartial

DecodeResult = tuple[str, float, float, str, float]


class FasterWhisperEngine:
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        asr_step_ms: int = 200,
        sample_rate: int = 16000,
        guard_config: HallucinationGuardConfig | None = None,
    ) -> None:
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._step_ms = asr_step_ms
        self._sample_rate = sample_rate
        self._guard_config = guard_config or HallucinationGuardConfig()

    async def stream(
        self, frames: AsyncIterator[AudioFrame], lang_hint: str | None
    ) -> AsyncIterator[TranscriptPartial]:
        loop = asyncio.get_running_loop()
        buffer = np.zeros(0, dtype=np.float32)
        buffer_start_ms = 0
        last_decode_ms: int | None = None
        agreement = LocalAgreement()

        async for frame in frames:
            if frame.sample_rate != self._sample_rate:
                raise ValueError(
                    f"FasterWhisperEngine configured for {self._sample_rate} Hz, "
                    f"got a frame at {frame.sample_rate} Hz; resample first"
                )
            if last_decode_ms is None:
                buffer_start_ms = frame.t_ms
                last_decode_ms = frame.t_ms

            buffer = np.concatenate([buffer, pcm16_to_float32(frame.pcm)])

            if frame.t_ms - last_decode_ms >= self._step_ms:
                last_decode_ms = frame.t_ms
                text, no_speech_prob, avg_logprob, lang, lang_prob = await loop.run_in_executor(
                    None, self._decode, buffer.copy(), lang_hint
                )
                stable = agreement.update(text)
                yield TranscriptPartial(
                    text=text,
                    stable_prefix=stable,
                    lang=lang,
                    lang_prob=lang_prob,
                    is_final=False,
                    t_start_ms=buffer_start_ms,
                    t_end_ms=frame.t_ms,
                    no_speech_prob=no_speech_prob,
                    avg_logprob=avg_logprob,
                )

        if len(buffer) > 0:
            text, no_speech_prob, avg_logprob, lang, lang_prob = await loop.run_in_executor(
                None, self._decode, buffer.copy(), lang_hint
            )
            yield TranscriptPartial(
                text=text,
                stable_prefix=text,
                lang=lang,
                lang_prob=lang_prob,
                is_final=True,
                t_start_ms=buffer_start_ms,
                t_end_ms=last_decode_ms or buffer_start_ms,
                no_speech_prob=no_speech_prob,
                avg_logprob=avg_logprob,
            )

    def _decode(self, buffer: np.ndarray, lang_hint: str | None) -> DecodeResult:
        segments, info = self._model.transcribe(buffer, language=lang_hint)
        kept_texts: list[str] = []
        no_speech_probs: list[float] = []
        avg_logprobs: list[float] = []
        for segment in segments:
            reason = check_hallucination(
                segment.text,
                segment.no_speech_prob,
                segment.avg_logprob,
                segment.compression_ratio,
                self._guard_config,
            )
            if reason is None:
                kept_texts.append(segment.text.strip())
                no_speech_probs.append(segment.no_speech_prob)
                avg_logprobs.append(segment.avg_logprob)

        text = " ".join(kept_texts).strip()
        no_speech_prob = max(no_speech_probs) if no_speech_probs else 1.0
        avg_logprob = min(avg_logprobs) if avg_logprobs else -10.0
        return text, no_speech_prob, avg_logprob, info.language, info.language_probability
