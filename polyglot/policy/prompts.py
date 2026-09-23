"""Prompt assembly. See SPEC.md Section 8.10.

Naive order: retrieved passages first, then system prompt, then full
history, then the current turn. Tuned order (`prefix_cache_prompt_order`
flag): static system prompt first, then the compacted summary, then recent
raw turns, then retrieved passages, then the current turn — the parts that
don't change turn to turn come first, so a prefix-caching LLM backend can
reuse them.
"""

from polyglot.core.types import Message, Passage

SYSTEM_PROMPT = (
    "You are a support agent for airline passengers affected by flight "
    "disruptions. Answer only from the policy passages provided below. Cite "
    "the source document name when stating a rule. If the policy text does "
    "not cover the question, say so plainly instead of guessing. Keep "
    "spoken responses under 3 sentences unless the caller asks for detail. "
    "Respond in the caller's language. Do not use markdown or lists — this "
    "is spoken aloud."
)


def _passages_block(passages: list[Passage]) -> str:
    if not passages:
        return "No relevant policy passages were found for this question."
    lines = [f"[{p.doc_id}] {p.text}" for p in passages]
    return "Relevant policy passages:\n" + "\n\n".join(lines)


def build_messages(
    *,
    history: list[Message],
    passages: list[Passage],
    user_text: str,
    lang: str,
    summary: str | None = None,
    prefix_cache_prompt_order: bool = False,
) -> list[Message]:
    system = Message(role="system", content=SYSTEM_PROMPT)
    passages_message = Message(role="system", content=_passages_block(passages))
    user_message = Message(role="user", content=user_text, lang=lang)

    if prefix_cache_prompt_order:
        messages = [system]
        if summary:
            messages.append(
                Message(role="system", content=f"Conversation summary so far: {summary}")
            )
        messages.extend(history)
        messages.append(passages_message)
        messages.append(user_message)
        return messages

    return [passages_message, system, *history, user_message]
