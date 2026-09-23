from collections.abc import AsyncIterator

import pytest

from polyglot.core.types import AudioFrame
from polyglot.tts.router import TTSRouter


class _StubEngine:
    def __init__(self, languages: set[str], tag: str) -> None:
        self.languages = languages
        self.tag = tag

    async def synthesize(
        self, chunks: AsyncIterator[str], lang: str, voice: str
    ) -> AsyncIterator[AudioFrame]:
        yield AudioFrame(pcm=self.tag.encode(), t_ms=0)


async def _collect(chunks: AsyncIterator[AudioFrame]) -> list[AudioFrame]:
    return [c async for c in chunks]


async def _one_chunk() -> AsyncIterator[str]:
    yield "hello"


async def test_routes_to_engine_supporting_the_language() -> None:
    en_engine = _StubEngine({"en"}, "en-engine")
    es_engine = _StubEngine({"es"}, "es-engine")
    router = TTSRouter([en_engine, es_engine])

    frames = await _collect(router.synthesize(_one_chunk(), "es", "voice"))
    assert frames[0].pcm == b"es-engine"


async def test_first_engine_wins_on_overlap() -> None:
    first = _StubEngine({"en"}, "first")
    second = _StubEngine({"en"}, "second")
    router = TTSRouter([first, second])

    frames = await _collect(router.synthesize(_one_chunk(), "en", "voice"))
    assert frames[0].pcm == b"first"


def test_languages_is_the_union() -> None:
    router = TTSRouter([_StubEngine({"en"}, "a"), _StubEngine({"es", "hi"}, "b")])
    assert router.languages == {"en", "es", "hi"}


async def test_unsupported_language_raises() -> None:
    router = TTSRouter([_StubEngine({"en"}, "a")])
    with pytest.raises(ValueError, match="tl"):
        await _collect(router.synthesize(_one_chunk(), "tl", "voice"))
