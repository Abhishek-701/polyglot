"""Real Anthropic API calls. Requires ANTHROPIC_API_KEY in .env.
Marked @pytest.mark.network; skips if the key isn't set.
"""

import os

import pytest
from dotenv import load_dotenv

from polyglot.core.interfaces import LLMClient
from polyglot.core.types import Message
from polyglot.llm.client import ClaudeLLMClient

load_dotenv()

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"),
]


def test_satisfies_llm_client_protocol() -> None:
    assert isinstance(ClaudeLLMClient(), LLMClient)


async def test_streams_real_text() -> None:
    client = ClaudeLLMClient()
    messages = [Message(role="user", content="Reply with exactly the word: pong")]
    text = ""
    finish_reason = None
    async for delta in client.stream(messages, None):
        if delta.text:
            text += delta.text
        if delta.finish_reason:
            finish_reason = delta.finish_reason
    assert "pong" in text.lower()
    assert finish_reason == "end_turn"


async def test_streams_real_tool_call() -> None:
    client = ClaudeLLMClient()
    tools = [
        {
            "name": "get_flight_status",
            "description": "Get the status of a flight",
            "input_schema": {
                "type": "object",
                "properties": {"flight_number": {"type": "string"}},
                "required": ["flight_number"],
            },
        }
    ]
    messages = [Message(role="user", content="Is flight AA100 delayed?")]
    tool_call = None
    async for delta in client.stream(messages, tools):
        if delta.tool_call:
            tool_call = delta.tool_call
    assert tool_call is not None
    assert tool_call["name"] == "get_flight_status"
    assert tool_call["input"]["flight_number"] == "AA100"


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        ClaudeLLMClient(api_key=None)
