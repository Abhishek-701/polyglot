"""Component protocols. See SPEC.md Section 7.2.

Every component depends on these Protocols, not on concrete classes. Marked
@runtime_checkable so tests can assert a Fake structurally satisfies its
Protocol with isinstance().
"""

from collections.abc import AsyncIterator
from contextlib import AbstractContextManager
from typing import Any, Protocol, runtime_checkable

from polyglot.core.types import AudioFrame, LLMDelta, Message, Passage, TranscriptPartial, VADEvent


@runtime_checkable
class Clock(Protocol):
    def now_ms(self) -> int: ...
    async def sleep_ms(self, ms: int) -> None: ...


@runtime_checkable
class VAD(Protocol):
    def process(self, frame: AudioFrame) -> list[VADEvent]: ...


@runtime_checkable
class ASREngine(Protocol):
    def stream(
        self, frames: AsyncIterator[AudioFrame], lang_hint: str | None
    ) -> AsyncIterator[TranscriptPartial]: ...


@runtime_checkable
class TurnDetector(Protocol):
    def end_of_turn_prob(
        self, text: str, lang: str, trailing_silence_ms: int, history: list[Message]
    ) -> float: ...


@runtime_checkable
class Retriever(Protocol):
    async def retrieve(self, query: str, lang: str, k: int) -> list[Passage]: ...


@runtime_checkable
class LLMClient(Protocol):
    def stream(
        self, messages: list[Message], tools: list[dict[str, Any]] | None
    ) -> AsyncIterator[LLMDelta]: ...


@runtime_checkable
class TTSEngine(Protocol):
    languages: set[str]

    def synthesize(
        self, chunks: AsyncIterator[str], lang: str, voice: str
    ) -> AsyncIterator[AudioFrame]: ...


@runtime_checkable
class Tracer(Protocol):
    def span(self, name: str, **attrs: Any) -> AbstractContextManager[Any]: ...
