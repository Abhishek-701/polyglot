"""Claude-backed LLM client. See SPEC.md Section 5 / 8.11.

SPEC.md names vLLM/Qwen2.5-7B-Instruct or a hosted OpenAI-compatible
endpoint. This project uses Anthropic's API instead (PROGRESS.md M4
decision: the user has an Anthropic key available now, not a GPU or an
OpenAI-compatible key). `LLMClient` (core/interfaces.py) is backend-agnostic
by design, so this substitution doesn't touch anything else in `core/`.

Verified against installed anthropic==1.8.0: streaming text arrives as
`anthropic.lib.streaming.TextEvent` (`.text`); `stream.get_final_message()`
returns fully-parsed `ToolUseBlock.input` (no manual partial-JSON
accumulation needed) and `stop_reason`.

Tool calling here is single-turn parameter extraction only (the dialogue
policy's `tool_call` node uses it to pull structured args like
`flight_number` out of the user's utterance, then calls the actual mock
tool function itself) — never a multi-turn tool_use/tool_result thread.
`core/types.py`'s `Message` has no `tool_use_id` field to thread that
through anyway, and SPEC.md Section 8.9 already splits `tool_call` and
`generate` into separate graph nodes, implying the tool's result is folded
into the prompt for `generate`, not preserved as a formal tool-result
message.
"""

import os
from collections.abc import AsyncIterator
from typing import Any

import anthropic
from anthropic.lib.streaming import TextEvent
from dotenv import load_dotenv

from polyglot.core.types import LLMDelta, Message

load_dotenv()  # picks up .env's ANTHROPIC_API_KEY without every caller doing it

DEFAULT_MODEL = "claude-haiku-4-5-20251001"  # chosen for TTFT; see PROGRESS.md M4


def _split_system_and_messages(messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
    """Anthropic's Messages API takes the system prompt as a separate
    top-level string, not interleaved in the messages array. All
    "system"-role Messages (system prompt + retrieved-passages context, per
    SPEC.md 8.10) are joined into one system string; everything else maps
    straight across. A stray "tool"-role Message (shouldn't occur — see
    module docstring) falls back to a plain user message rather than
    erroring, since Anthropic has no bare "tool" role.
    """
    system_parts = [m.content for m in messages if m.role == "system"]
    system = "\n\n".join(system_parts) if system_parts else None

    anthropic_messages: list[dict[str, Any]] = [
        {"role": "user" if m.role == "tool" else m.role, "content": m.content}
        for m in messages
        if m.role != "system"
    ]
    return system, anthropic_messages


class ClaudeLLMClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 1024,
        api_key: str | None = None,
    ) -> None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        self._client = anthropic.AsyncAnthropic(api_key=key)
        self._model = model
        self._max_tokens = max_tokens

    async def stream(
        self, messages: list[Message], tools: list[dict[str, Any]] | None
    ) -> AsyncIterator[LLMDelta]:
        system, anthropic_messages = _split_system_and_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": anthropic_messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        async with self._client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if isinstance(event, TextEvent):
                    yield LLMDelta(text=event.text)

            final = await stream.get_final_message()
            for block in final.content:
                if block.type == "tool_use":
                    yield LLMDelta(tool_call={"name": block.name, "input": block.input})
            yield LLMDelta(finish_reason=final.stop_reason)
