"""Real Kokoro-82M synthesis. Marked @pytest.mark.network (downloads the
model on first use).
"""

from collections.abc import AsyncIterator

import pytest

from polyglot.tts.kokoro_engine import SAMPLE_RATE, KokoroEngine

pytestmark = pytest.mark.network


async def _one_chunk(text: str) -> AsyncIterator[str]:
    yield text


@pytest.mark.parametrize(
    "lang,text",
    [("en", "Hello there."), ("es", "Hola."), ("hi", "नमस्ते।"), ("zh", "你好。")],
)
async def test_synthesizes_real_audio(lang: str, text: str) -> None:
    engine = KokoroEngine()
    frames = [frame async for frame in engine.synthesize(_one_chunk(text), lang, "")]
    assert frames
    assert all(frame.sample_rate == SAMPLE_RATE for frame in frames)
    assert all(len(frame.pcm) > 0 for frame in frames)


async def test_unsupported_language_raises() -> None:
    engine = KokoroEngine()
    with pytest.raises(ValueError, match="tl"):
        async for _ in engine.synthesize(_one_chunk("test"), "tl", ""):
            pass
