"""Clause splitter. See SPEC.md Section 8.12.

Takes a stream of LLM text deltas (which can split mid-word — token
boundaries don't respect word boundaries) and emits speakable chunks per
the `tts_chunking` mode: `full` (one chunk at the end), `sentence` (split
on sentence punctuation), `clause` (split on clause punctuation once at
least `min_chunk_words` words have accumulated; force an emit at
`max_chunk_words` regardless). The Devanagari danda (।) counts as a
sentence/clause boundary for Hindi.

Only complete, whitespace-delimited words are ever moved out of the raw
text buffer — a delta ending mid-word leaves that partial word in the
buffer until more text confirms where it ends.
"""

from collections.abc import AsyncIterator
from typing import Literal

TtsChunkingMode = Literal["full", "sentence", "clause"]

_SENTENCE_BOUNDARY_CHARS = frozenset(".!?।")
_CLAUSE_BOUNDARY_CHARS = frozenset(",;:.!?।")


def _extract_complete_words(buffer: str) -> tuple[list[str], str]:
    """Splits off whitespace-delimited words that are definitely complete,
    leaving a possibly-incomplete trailing word (or "") in the remainder.
    """
    if buffer == "":
        return [], ""
    if buffer[-1].isspace():
        return buffer.split(), ""
    parts = buffer.split()
    if len(parts) <= 1:
        return [], buffer
    return parts[:-1], parts[-1]


def _find_boundary_index(words: list[str], boundary_chars: frozenset[str]) -> int | None:
    for index, word in enumerate(words):
        if word and word[-1] in boundary_chars:
            return index
    return None


async def split_stream(
    text_stream: AsyncIterator[str],
    mode: TtsChunkingMode = "full",
    min_chunk_words: int = 4,
    max_chunk_words: int = 18,
) -> AsyncIterator[str]:
    if mode == "full":
        buffer = ""
        async for delta in text_stream:
            buffer += delta
        if buffer.strip():
            yield buffer.strip()
        return

    boundary_chars = _SENTENCE_BOUNDARY_CHARS if mode == "sentence" else _CLAUSE_BOUNDARY_CHARS
    raw_buffer = ""
    pending_words: list[str] = []

    async for delta in text_stream:
        raw_buffer += delta
        new_words, raw_buffer = _extract_complete_words(raw_buffer)
        pending_words.extend(new_words)

        while True:
            boundary_index = _find_boundary_index(pending_words, boundary_chars)
            if boundary_index is not None and boundary_index + 1 >= min_chunk_words:
                yield " ".join(pending_words[: boundary_index + 1])
                pending_words = pending_words[boundary_index + 1 :]
                continue
            if len(pending_words) >= max_chunk_words:
                yield " ".join(pending_words[:max_chunk_words])
                pending_words = pending_words[max_chunk_words:]
                continue
            break

    remainder = pending_words + ([raw_buffer] if raw_buffer else [])
    if remainder:
        yield " ".join(remainder)
