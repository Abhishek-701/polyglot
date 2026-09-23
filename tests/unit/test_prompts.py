from polyglot.core.types import Message, Passage
from polyglot.policy.prompts import SYSTEM_PROMPT, build_messages

PASSAGE = Passage(
    id="p1", doc_id="dot-refunds", text="Refunds are due within 7 days.", source_url="x", score=0.9
)
HISTORY = [Message(role="user", content="hi"), Message(role="assistant", content="hello")]


def test_naive_order_puts_passages_first() -> None:
    messages = build_messages(
        history=HISTORY,
        passages=[PASSAGE],
        user_text="What about refunds?",
        lang="en",
        prefix_cache_prompt_order=False,
    )
    assert "Refunds are due" in messages[0].content
    assert messages[1].content == SYSTEM_PROMPT
    assert messages[2:4] == HISTORY
    assert messages[-1].content == "What about refunds?"


def test_tuned_order_puts_system_prompt_first() -> None:
    messages = build_messages(
        history=HISTORY,
        passages=[PASSAGE],
        user_text="What about refunds?",
        lang="en",
        prefix_cache_prompt_order=True,
    )
    assert messages[0].content == SYSTEM_PROMPT
    assert messages[-1].content == "What about refunds?"
    assert any("Refunds are due" in m.content for m in messages)


def test_tuned_order_includes_summary_when_given() -> None:
    messages = build_messages(
        history=[],
        passages=[],
        user_text="hi",
        lang="en",
        summary="Caller asked about a cancelled flight earlier.",
        prefix_cache_prompt_order=True,
    )
    assert any("cancelled flight earlier" in m.content for m in messages)


def test_no_passages_says_so_explicitly() -> None:
    messages = build_messages(history=[], passages=[], user_text="hi", lang="en")
    assert any("No relevant policy passages" in m.content for m in messages)
