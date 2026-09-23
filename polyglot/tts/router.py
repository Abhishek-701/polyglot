"""TTS engine router. See SPEC.md Section 8.12: "pick the engine by
language from languages.yaml." Composes multiple engines behind one
`TTSEngine`-shaped interface, picking by language; the first engine in the
list wins for any language more than one of them supports.

The fourth project language was swapped from Tagalog to Mandarin in M4
because no engine evaluated supported Tagalog at all — see
docs/tts_language_matrix.md. A call for an unsupported language raises
rather than silently failing.
"""

from collections.abc import AsyncIterator

from polyglot.core.interfaces import TTSEngine
from polyglot.core.types import AudioFrame


class TTSRouter:
    def __init__(self, engines: list[TTSEngine]) -> None:
        self._by_lang: dict[str, TTSEngine] = {}
        for engine in engines:
            for lang in engine.languages:
                self._by_lang.setdefault(lang, engine)
        self.languages: set[str] = set(self._by_lang)

    async def synthesize(
        self, chunks: AsyncIterator[str], lang: str, voice: str
    ) -> AsyncIterator[AudioFrame]:
        engine = self._by_lang.get(lang)
        if engine is None:
            raise ValueError(
                f"No TTS engine configured for {lang!r} — see docs/tts_language_matrix.md"
            )
        async for frame in engine.synthesize(chunks, lang, voice):
            yield frame
