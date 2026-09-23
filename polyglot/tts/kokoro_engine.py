"""Kokoro-82M TTS engine. See SPEC.md Section 8.12 / Section 5.

Verified against installed kokoro==0.9.4: `KPipeline(lang_code=...)` with
`'a'` (English), `'e'` (Spanish), `'h'` (Hindi) all produced real audio on
CPU in testing, with no espeak-ng install needed (VOICES.md lists
espeak-ng as a fallback phonemizer; misaki's bundled G2P handled all three
without it here — not something to assume holds on every machine, since
that fallback path wasn't exercised). No Tagalog support at all — see
docs/tts_language_matrix.md.

CPU inference blocks, so synthesis runs in the default executor per
CLAUDE.md's "nothing blocking on the audio path" rule. Output stays at
Kokoro's native 24kHz; resampling to the transport's rate is a downstream
concern (SPEC.md 8.12), not this engine's job.
"""

import asyncio
from collections.abc import AsyncIterator

import numpy as np
from kokoro import KPipeline

from polyglot.audio.frames import float32_to_pcm16
from polyglot.core.types import AudioFrame

SAMPLE_RATE = 24000

_LANG_CODES = {"en": "a", "es": "e", "hi": "h"}
_DEFAULT_VOICES = {"en": "af_heart", "es": "ef_dora", "hi": "hf_alpha"}


class KokoroEngine:
    languages: set[str] = set(_LANG_CODES)

    def __init__(self) -> None:
        self._pipelines: dict[str, KPipeline] = {}

    def _get_pipeline(self, lang: str) -> KPipeline:
        if lang not in self._pipelines:
            code = _LANG_CODES.get(lang)
            if code is None:
                raise ValueError(
                    f"KokoroEngine does not support {lang!r} — see docs/tts_language_matrix.md"
                )
            self._pipelines[lang] = KPipeline(lang_code=code)
        return self._pipelines[lang]

    async def synthesize(
        self, chunks: AsyncIterator[str], lang: str, voice: str
    ) -> AsyncIterator[AudioFrame]:
        loop = asyncio.get_running_loop()
        pipeline = self._get_pipeline(lang)
        voice = voice or _DEFAULT_VOICES.get(lang, "af_heart")
        t_ms = 0

        async for text in chunks:
            if not text.strip():
                continue
            results = await loop.run_in_executor(None, self._run, pipeline, text, voice)
            for result in results:
                if result.audio is None:
                    continue
                samples = result.audio.numpy().astype(np.float32)
                yield AudioFrame(pcm=float32_to_pcm16(samples), sample_rate=SAMPLE_RATE, t_ms=t_ms)
                t_ms += round(len(samples) / SAMPLE_RATE * 1000)

    @staticmethod
    def _run(pipeline: KPipeline, text: str, voice: str) -> list:  # type: ignore[type-arg]
        return list(pipeline(text, voice=voice))
