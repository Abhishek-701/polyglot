"""Policy graph routing tests — fully fake (LLM, retriever, and intent
classifier all injected), no network/model needed. Real end-to-end
behavior with the actual Claude client, retriever, and zero-shot classifier
was hand-verified separately (PROGRESS.md M4) since that needs a live API
key and downloaded models, not something to require for every test run.
"""

from collections.abc import AsyncIterator
from typing import Any

from polyglot.core.types import LLMDelta, Message, Passage
from polyglot.fakes import FakeRetriever
from polyglot.policy.graph import build_policy_graph
from polyglot.policy.intents import IntentResult


def _classifier(label: str, confidence: float):
    def classify(text: str) -> IntentResult:
        return IntentResult(label=label, confidence=confidence)  # type: ignore[arg-type]

    return classify


class _ScriptedLLM:
    """A tools-call yields tool_call_input (if any); a no-tools call yields generate_text."""

    def __init__(self, tool_call_input: dict[str, Any] | None = None, generate_text: str = "ok"):
        self._tool_call_input = tool_call_input
        self._generate_text = generate_text

    async def stream(
        self, messages: list[Message], tools: list[dict[str, Any]] | None
    ) -> AsyncIterator[LLMDelta]:
        if tools:
            if self._tool_call_input is not None:
                yield LLMDelta(tool_call={"name": tools[0]["name"], "input": self._tool_call_input})
        else:
            yield LLMDelta(text=self._generate_text)


async def test_low_confidence_routes_to_clarify() -> None:
    graph = build_policy_graph(
        _ScriptedLLM(generate_text="unused"),
        FakeRetriever(),
        confidence_floor=0.5,
        intent_classifier=_classifier("refund", 0.1),
    )
    result = await graph.ainvoke({"user_text": "hmm", "lang": "en", "history": []})
    assert "help" in result["assistant_text"].lower() or "más" in result["assistant_text"].lower()
    assert "passages" not in result


async def test_handoff_keyword_overrides_intent() -> None:
    graph = build_policy_graph(
        _ScriptedLLM(generate_text="unused"),
        FakeRetriever(),
        intent_classifier=_classifier("refund", 0.9),
    )
    result = await graph.ainvoke(
        {"user_text": "I want to speak to a human agent", "lang": "en", "history": []}
    )
    assert result["handoff"] is True


async def test_refund_intent_retrieves_and_generates() -> None:
    passage = Passage(
        id="p1", doc_id="dot", text="Refunds within 7 days.", source_url="x", score=0.9
    )
    graph = build_policy_graph(
        _ScriptedLLM(generate_text="You get a refund within 7 days."),
        FakeRetriever(passages=[passage]),
        intent_classifier=_classifier("refund", 0.9),
    )
    result = await graph.ainvoke({"user_text": "refund please", "lang": "en", "history": []})
    assert result["passages"] == [passage]
    assert result["assistant_text"] == "You get a refund within 7 days."


async def test_smalltalk_skips_retrieval() -> None:
    graph = build_policy_graph(
        _ScriptedLLM(generate_text="Hello! How can I help?"),
        FakeRetriever(passages=[Passage(id="p1", doc_id="d", text="x", source_url="x", score=1)]),
        intent_classifier=_classifier("smalltalk", 0.9),
    )
    result = await graph.ainvoke({"user_text": "hi", "lang": "en", "history": []})
    assert "passages" not in result
    assert result["assistant_text"] == "Hello! How can I help?"


async def test_flight_status_intent_calls_tool() -> None:
    graph = build_policy_graph(
        _ScriptedLLM(
            tool_call_input={"flight_number": "AA100", "date": "2026-09-25"},
            generate_text="Your flight is on time.",
        ),
        FakeRetriever(),
        intent_classifier=_classifier("flight_status", 0.9),
    )
    result = await graph.ainvoke(
        {"user_text": "is AA100 on time on 2026-09-25", "lang": "en", "history": []}
    )
    assert result["tool_result"] is not None
    assert result["tool_result"]["status"] == "on_time"
    assert result["assistant_text"] == "Your flight is on time."


async def test_flight_status_no_match_notes_it_in_generate() -> None:
    graph = build_policy_graph(
        _ScriptedLLM(
            tool_call_input={"flight_number": "ZZ999", "date": "2099-01-01"},
            generate_text="I couldn't find that flight.",
        ),
        FakeRetriever(),
        intent_classifier=_classifier("flight_status", 0.9),
    )
    result = await graph.ainvoke({"user_text": "status of ZZ999", "lang": "en", "history": []})
    assert result["tool_result"] is None
    assert result["assistant_text"] == "I couldn't find that flight."


async def test_guard_replaces_ungrounded_figure() -> None:
    passage = Passage(
        id="p1", doc_id="dot", text="Compensation is $400.", source_url="x", score=0.9
    )
    graph = build_policy_graph(
        _ScriptedLLM(generate_text="You're entitled to $999 in compensation."),
        FakeRetriever(passages=[passage]),
        intent_classifier=_classifier("compensation", 0.9),
    )
    result = await graph.ainvoke(
        {"user_text": "how much compensation", "lang": "en", "history": []}
    )
    assert result["guard_triggered"] is True
    assert "$999" not in result["assistant_text"]
