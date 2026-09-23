"""Clause splitter. See SPEC.md Section 8.12.

Takes a stream of LLM text deltas and emits speakable chunks per the
`tts_chunking` mode: `full` (one chunk at the end), `sentence` (split on
sentence punctuation), `clause` (split on clause punctuation once at least
`min_chunk_words` units have accumulated; force an emit at
`max_chunk_words` regardless). The Devanagari danda (।) counts as a
sentence/clause boundary for Hindi; full-width Chinese punctuation
(。！？，；：) counts for Mandarin.

"Word" is whitespace-delimited for space-delimited languages. Mandarin has
no spaces between words at all, so for `lang="zh"` the unit is one
character instead — `min_chunk_words`/`max_chunk_words` then mean character
counts, and chunks are joined with "" instead of " " (CJK_LANGUAGES below).
Getting this distinction wrong either produces one giant unsplittable
"word" for Chinese, or literally injects a space between every character
when rejoining — the latter breaks a TTS engine's own text normalization
(and just reads/sounds strange to a fluent reader/listener).

For space-delimited languages: LLM stream deltas split mid-word — token
boundaries don't respect word boundaries — so only complete words are ever
moved out of the raw text buffer; a delta ending mid-word leaves that
partial word in the buffer until more text confirms where it ends. This
concern doesn't apply to the character-unit path: every character in a
Python str is already complete regardless of how deltas are chopped.
"""

from collections.abc import AsyncIterator
from typing import Literal

TtsChunkingMode = Literal["full", "sentence", "clause"]

CJK_LANGUAGES = frozenset({"zh"})

_SENTENCE_BOUNDARY_CHARS = frozenset(".!?।。！？")
_CLAUSE_BOUNDARY_CHARS = frozenset(",;:.!?।。！？，；：")


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


def _extract_complete_units(buffer: str, use_characters: bool) -> tuple[list[str], str]:
    if use_characters:
        return list(buffer), ""
    return _extract_complete_words(buffer)


def _find_boundary_index(units: list[str], boundary_chars: frozenset[str]) -> int | None:
    for index, unit in enumerate(units):
        if unit and unit[-1] in boundary_chars:
            return index
    return None


async def split_stream(
    text_stream: AsyncIterator[str],
    mode: TtsChunkingMode = "full",
    min_chunk_words: int = 4,
    max_chunk_words: int = 18,
    lang: str = "en",
) -> AsyncIterator[str]:
    use_characters = lang in CJK_LANGUAGES
    joiner = "" if use_characters else " "

    if mode == "full":
        buffer = ""
        async for delta in text_stream:
            buffer += delta
        if buffer.strip():
            yield buffer.strip()
        return

    boundary_chars = _SENTENCE_BOUNDARY_CHARS if mode == "sentence" else _CLAUSE_BOUNDARY_CHARS
    raw_buffer = ""
    pending_units: list[str] = []

    async for delta in text_stream:
        raw_buffer += delta
        new_units, raw_buffer = _extract_complete_units(raw_buffer, use_characters)
        pending_units.extend(new_units)

        while True:
            boundary_index = _find_boundary_index(pending_units, boundary_chars)
            if boundary_index is not None and boundary_index + 1 >= min_chunk_words:
                yield joiner.join(pending_units[: boundary_index + 1])
                pending_units = pending_units[boundary_index + 1 :]
                continue
            if len(pending_units) >= max_chunk_words:
                yield joiner.join(pending_units[:max_chunk_words])
                pending_units = pending_units[max_chunk_words:]
                continue
            break

    remainder = pending_units + ([raw_buffer] if raw_buffer else [])
    if remainder:
        yield joiner.join(remainder)
