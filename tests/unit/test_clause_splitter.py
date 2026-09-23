"""clause_splitter tests, including a Hypothesis property test (SPEC.md
Section 14 explicitly wants property-based tests for this module): no
matter how the same text is chopped into streaming deltas, the words
extracted from the emitted chunks reconstruct the original word sequence.
"""

import asyncio
from collections.abc import AsyncIterator

from hypothesis import given
from hypothesis import strategies as st

from polyglot.tts.clause_splitter import split_stream


async def _feed(deltas: list[str]) -> AsyncIterator[str]:
    for delta in deltas:
        yield delta


def _run_split(deltas: list[str], **kwargs: object) -> list[str]:
    async def collect() -> list[str]:
        return [chunk async for chunk in split_stream(_feed(deltas), **kwargs)]  # type: ignore[arg-type]

    return asyncio.run(collect())


def test_full_mode_yields_a_single_chunk() -> None:
    text = "Hello world. This is a test."
    assert _run_split(list(text), mode="full") == [text]


def test_sentence_mode_splits_on_period() -> None:
    text = "This is one sentence. This is another sentence."
    chunks = _run_split([text], mode="sentence")
    assert chunks == ["This is one sentence.", "This is another sentence."]


def test_clause_mode_respects_min_chunk_words() -> None:
    text = "Ok, well, here is a longer clause, and then more."
    chunks = _run_split([text], mode="clause", min_chunk_words=4)
    for chunk in chunks[:-1]:
        assert len(chunk.split()) >= 4


def test_clause_mode_force_emits_at_max_chunk_words() -> None:
    words = [f"word{i}" for i in range(40)]  # no punctuation at all
    text = " ".join(words)
    chunks = _run_split([text], mode="clause", max_chunk_words=18)
    assert all(len(chunk.split()) <= 18 for chunk in chunks)
    assert len(chunks) > 1


def test_hindi_danda_is_a_sentence_boundary() -> None:
    text = "यह एक वाक्य है। यह दूसरा वाक्य है।"
    chunks = _run_split([text], mode="sentence")
    assert len(chunks) == 2


def test_delta_splitting_mid_word_does_not_break_words() -> None:
    # "world." split as "wor" + "ld." across two deltas — must not appear as
    # two separate words in the output.
    chunks = _run_split(["Hello wor", "ld. Another sentence."], mode="sentence")
    words = " ".join(chunks).split()
    assert "wor" not in words
    assert "ld." not in words
    assert "world." in words


_WORD = st.sampled_from(["hello", "world,", "this", "is", "a", "test.", "of", "splitting;", "ok"])


@given(
    st.lists(_WORD, min_size=5, max_size=25),
    st.integers(min_value=1, max_value=7),
    st.sampled_from(["full", "sentence", "clause"]),
)
def test_word_reconstruction_property(words: list[str], delta_size: int, mode: str) -> None:
    text = " ".join(words)
    deltas = [text[i : i + delta_size] for i in range(0, len(text), delta_size)] or [""]
    chunks = _run_split(deltas, mode=mode)
    assert " ".join(chunks).split() == words
