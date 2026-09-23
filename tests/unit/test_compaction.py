from polyglot.core.types import LLMDelta, Message
from polyglot.fakes import FakeLLMClient
from polyglot.policy.compaction import split_recent_and_older, summarize_older_turns


def _messages(n: int) -> list[Message]:
    return [Message(role="user", content=f"turn {i}") for i in range(n)]


def test_split_keeps_recent_when_short() -> None:
    history = _messages(4)
    recent, older = split_recent_and_older(history, keep_recent=6)
    assert recent == history
    assert older == []


def test_split_separates_older_from_recent() -> None:
    history = _messages(10)
    recent, older = split_recent_and_older(history, keep_recent=6)
    assert recent == history[-6:]
    assert older == history[:-6]


async def test_summarize_older_turns_calls_llm() -> None:
    llm = FakeLLMClient(deltas=[LLMDelta(text="Caller asked about a refund.")])
    summary = await summarize_older_turns(_messages(3), llm, previous_summary=None)
    assert summary == "Caller asked about a refund."


async def test_summarize_older_turns_empty_returns_previous() -> None:
    llm = FakeLLMClient(deltas=[])
    summary = await summarize_older_turns([], llm, previous_summary="old summary")
    assert summary == "old summary"
