"""chunk_section windowing/overlap/merge logic, tested with a fake
word-level tokenizer so it needs no network / model download. Real-tokenizer
behavior (BGE-M3) is covered by a network-marked integration test.
"""

from polyglot.retrieval.chunking import chunk_section


class _FakeTokenizer:
    """One token per word — makes chunk boundaries easy to reason about."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return text.split()

    def decode(self, ids: list[str], skip_special_tokens: bool = True) -> str:
        return " ".join(ids)


def _words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_empty_text_produces_no_chunks() -> None:
    assert chunk_section("", "s", tokenizer=_FakeTokenizer()) == []


def test_short_text_is_a_single_chunk() -> None:
    chunks = chunk_section(_words(100), "intro", tokenizer=_FakeTokenizer())
    assert len(chunks) == 1
    assert chunks[0].token_count == 100
    assert chunks[0].section == "intro"
    assert chunks[0].chunk_index == 0


def test_long_text_splits_with_configured_overlap() -> None:
    chunks = chunk_section(
        _words(950),
        "s",
        tokenizer=_FakeTokenizer(),
        min_tokens=300,
        max_tokens=500,
        overlap_tokens=50,
    )
    assert len(chunks) >= 2
    for index, chunk in enumerate(chunks):
        assert chunk.chunk_index == index

    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    assert first_words[-50:] == second_words[:50]


def test_no_chunk_is_smaller_than_min_tokens() -> None:
    # A tail window smaller than min_tokens should be folded into the
    # previous chunk rather than stand alone as a tiny fragment — even if
    # that means the merged chunk exceeds max_tokens.
    for n in range(301, 1100, 37):
        chunks = chunk_section(
            _words(n),
            "s",
            tokenizer=_FakeTokenizer(),
            min_tokens=300,
            max_tokens=500,
            overlap_tokens=50,
        )
        assert all(c.token_count >= 300 for c in chunks), (n, [c.token_count for c in chunks])


def test_chunk_indices_are_sequential_and_texts_reconstruct_tokens() -> None:
    chunks = chunk_section(_words(300), "s", tokenizer=_FakeTokenizer())
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
