"""Fakes for every Protocol in polyglot/core/interfaces.py. See SPEC.md Section 14."""

from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from typing import Any

from polyglot.core.clock import SimulatedClock
from polyglot.core.types import AudioFrame, LLMDelta, Message, Passage, TranscriptPartial, VADEvent


class FakeClock(SimulatedClock):
    """SimulatedClock already satisfies the Clock protocol; alias for readability in tests."""


class FakeVAD:
    def __init__(self, events: list[VADEvent] | None = None) -> None:
        self._events = events or []

    def process(self, frame: AudioFrame) -> list[VADEvent]:
        return self._events


class FakeASREngine:
    def __init__(self, partials: list[TranscriptPartial] | None = None) -> None:
        self._partials = partials or []

    async def stream(
        self, frames: AsyncIterator[AudioFrame], lang_hint: str | None
    ) -> AsyncIterator[TranscriptPartial]:
        for partial in self._partials:
            yield partial


class FakeTurnDetector:
    def __init__(self, prob: float = 1.0) -> None:
        self._prob = prob

    def end_of_turn_prob(
        self, text: str, lang: str, trailing_silence_ms: int, history: list[Message]
    ) -> float:
        return self._prob


class FakeRetriever:
    def __init__(self, passages: list[Passage] | None = None) -> None:
        self._passages = passages or []

    async def retrieve(self, query: str, lang: str, k: int) -> list[Passage]:
        return self._passages[:k]


class FakeLLMClient:
    def __init__(self, deltas: list[LLMDelta] | None = None) -> None:
        self._deltas = deltas or []

    async def stream(
        self, messages: list[Message], tools: list[dict[str, Any]] | None
    ) -> AsyncIterator[LLMDelta]:
        for delta in self._deltas:
            yield delta


class FakeTTSEngine:
    languages: set[str] = {"en", "es", "hi", "tl"}

    async def synthesize(
        self, chunks: AsyncIterator[str], lang: str, voice: str
    ) -> AsyncIterator[AudioFrame]:
        t_ms = 0
        async for chunk in chunks:
            yield AudioFrame(pcm=chunk.encode("utf-8"), t_ms=t_ms)
            t_ms += 20


class FakeTracer:
    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[None]:
        yield None
