"""LangGraph dialogue policy. See SPEC.md Section 8.9.

Nodes: classify_intent -> {clarify | handoff | tool_call | retrieve |
generate (smalltalk)} -> generate -> guard -> END. Verified against
installed langgraph==1.2.12: async node functions, StateGraph with a
TypedDict schema, and add_conditional_edges with a plain (non-async)
router function all work as expected (smoke-tested standalone before
writing this).

Only the policy graph lives here — the audio pipeline (core/pipeline.py)
stays plain asyncio per CLAUDE.md's architecture rule ("LangGraph is used
only for the dialogue policy").
"""

import json
from collections.abc import Callable
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from polyglot.core.interfaces import LLMClient, Retriever
from polyglot.core.types import Message, Passage
from polyglot.policy.guard import check_guard
from polyglot.policy.intents import IntentResult
from polyglot.policy.intents import classify_intent as default_classify_intent
from polyglot.policy.prompts import build_messages
from polyglot.tools.booking import lookup_booking
from polyglot.tools.flight_status import get_flight_status

DEFAULT_CONFIDENCE_FLOOR = 0.25  # rough heuristic — see policy/intents.py docstring

_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "flight_status": {
        "name": "get_flight_status",
        "description": "Get the status of a flight by flight number and date.",
        "input_schema": {
            "type": "object",
            "properties": {
                "flight_number": {"type": "string"},
                "date": {"type": "string", "description": "YYYY-MM-DD"},
            },
            "required": ["flight_number", "date"],
        },
    },
    "booking_lookup": {
        "name": "lookup_booking",
        "description": "Look up a booking by confirmation code and last name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "confirmation_code": {"type": "string"},
                "last_name": {"type": "string"},
            },
            "required": ["confirmation_code", "last_name"],
        },
    },
}

_CLARIFY_TEXT = {
    "en": "Could you tell me a bit more about what you need help with?",
    "es": "¿Podrías contarme un poco más sobre en qué necesitas ayuda?",
    "hi": "क्या आप बता सकते हैं कि आपको किस बारे में मदद चाहिए?",
    "zh": "您能再多告诉我一些您需要什么帮助吗？",
}

_HANDOFF_TEXT = {
    "en": "Let me connect you with a member of our team who can help further.",
    "es": "Permíteme conectarte con un miembro de nuestro equipo que pueda ayudarte más.",
    "hi": "मुझे आपको हमारी टीम के किसी सदस्य से जोड़ने दीजिए जो आगे मदद कर सके।",
    "zh": "让我为您转接一位团队成员，为您提供进一步帮助。",
}

_HANDOFF_TRIGGERS = (
    "human agent",
    "speak to a human",
    "talk to a person",
    "representative",
    "hablar con una persona",
    "agente humano",
    "人工客服",
    "转人工",
)


def wants_handoff(text: str) -> bool:
    lowered = text.lower()
    return any(trigger in lowered for trigger in _HANDOFF_TRIGGERS)


class PolicyState(TypedDict, total=False):
    user_text: str
    lang: str
    history: list[Message]
    passages: list[Passage]
    intent: str
    intent_confidence: float
    tool_result: dict[str, Any] | None
    assistant_text: str
    guard_triggered: bool
    handoff: bool


def build_policy_graph(
    llm: LLMClient,
    retriever: Retriever,
    *,
    k: int = 5,
    confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR,
    prefix_cache_prompt_order: bool = False,
    intent_classifier: Callable[[str], IntentResult] = default_classify_intent,
) -> Any:
    async def classify_intent_node(state: PolicyState) -> dict[str, Any]:
        result = intent_classifier(state["user_text"])
        return {"intent": result.label, "intent_confidence": result.confidence}

    async def retrieve_node(state: PolicyState) -> dict[str, Any]:
        passages = await retriever.retrieve(state["user_text"], state["lang"], k)
        return {"passages": passages}

    async def tool_call_node(state: PolicyState) -> dict[str, Any]:
        schema = _TOOL_SCHEMAS[state["intent"]]
        extraction_messages = [
            Message(
                role="system",
                content=(
                    "Extract the parameters for the tool call from the user's message "
                    "and conversation history. If a parameter is missing, use an empty string."
                ),
            ),
            *state.get("history", []),
            Message(role="user", content=state["user_text"]),
        ]
        tool_input: dict[str, Any] | None = None
        async for delta in llm.stream(extraction_messages, [schema]):
            if delta.tool_call:
                tool_input = delta.tool_call["input"]

        if tool_input is None:
            return {"tool_result": None}
        if state["intent"] == "flight_status":
            result = get_flight_status(
                tool_input.get("flight_number", ""), tool_input.get("date", "")
            )
        else:
            result = lookup_booking(
                tool_input.get("confirmation_code", ""), tool_input.get("last_name", "")
            )
        return {"tool_result": result}

    async def clarify_node(state: PolicyState) -> dict[str, Any]:
        return {"assistant_text": _CLARIFY_TEXT.get(state["lang"], _CLARIFY_TEXT["en"])}

    async def handoff_node(state: PolicyState) -> dict[str, Any]:
        return {
            "assistant_text": _HANDOFF_TEXT.get(state["lang"], _HANDOFF_TEXT["en"]),
            "handoff": True,
        }

    async def generate_node(state: PolicyState) -> dict[str, Any]:
        passages = state.get("passages", [])
        messages = build_messages(
            history=state.get("history", []),
            passages=passages,
            user_text=state["user_text"],
            lang=state["lang"],
            prefix_cache_prompt_order=prefix_cache_prompt_order,
        )
        if "tool_result" in state:
            note = (
                f"Tool result: {json.dumps(state['tool_result'])}"
                if state["tool_result"] is not None
                else "No matching record was found for the details provided."
            )
            messages.insert(-1, Message(role="system", content=note))

        text = ""
        async for delta in llm.stream(messages, None):
            if delta.text:
                text += delta.text
        return {"assistant_text": text}

    async def guard_node(state: PolicyState) -> dict[str, Any]:
        hedge = check_guard(state["assistant_text"], state.get("passages", []))
        if hedge is not None:
            return {"assistant_text": hedge, "guard_triggered": True}
        return {}

    def route_after_classify(state: PolicyState) -> str:
        if wants_handoff(state["user_text"]):
            return "handoff"
        if state["intent_confidence"] < confidence_floor:
            return "clarify"
        if state["intent"] in ("flight_status", "booking_lookup"):
            return "tool_call"
        if state["intent"] == "smalltalk":
            return "generate"
        return "retrieve"

    graph = StateGraph(PolicyState)
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("tool_call", tool_call_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("handoff", handoff_node)
    graph.add_node("generate", generate_node)
    graph.add_node("guard", guard_node)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {
            "handoff": "handoff",
            "clarify": "clarify",
            "tool_call": "tool_call",
            "retrieve": "retrieve",
            "generate": "generate",
        },
    )
    graph.add_edge("retrieve", "generate")
    graph.add_edge("tool_call", "generate")
    graph.add_edge("generate", "guard")
    graph.add_edge("guard", END)
    graph.add_edge("clarify", END)
    graph.add_edge("handoff", END)

    return graph.compile()
