"""History compaction. See SPEC.md Section 8.10.

Keeps the last N turns raw; older turns get folded into a running summary.
`summarize_older_turns` is meant to be scheduled as a fire-and-forget
background task after a turn commits (e.g. `asyncio.create_task(...)`) —
never awaited inline in a turn's own critical path. The summary it produces
is only consumed by the *next* turn's prompt assembly.
"""

from polyglot.core.interfaces import LLMClient
from polyglot.core.types import Message

DEFAULT_KEEP_RECENT = 6


def split_recent_and_older(
    history: list[Message], keep_recent: int = DEFAULT_KEEP_RECENT
) -> tuple[list[Message], list[Message]]:
    if len(history) <= keep_recent:
        return history, []
    return history[-keep_recent:], history[:-keep_recent]


async def summarize_older_turns(
    older: list[Message], llm: LLMClient, previous_summary: str | None
) -> str:
    if not older:
        return previous_summary or ""

    transcript = "\n".join(f"{m.role}: {m.content}" for m in older)
    prompt = (
        "Summarize this airline support conversation so far in 2-3 sentences, "
        "keeping any facts that might matter later (names, confirmation codes, "
        "flight numbers, what was already resolved).\n\n"
    )
    if previous_summary:
        prompt += f"Previous summary: {previous_summary}\n\n"
    prompt += f"New turns to fold in:\n{transcript}"

    text = ""
    async for delta in llm.stream([Message(role="user", content=prompt)], None):
        if delta.text:
            text += delta.text
    return text.strip()
