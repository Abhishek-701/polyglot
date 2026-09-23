"""Chunking. See SPEC.md Section 8.7.

Tokenized with BGE-M3's own tokenizer (loaded once, cached module-level) so
chunk sizes are measured in the units the embedding model actually sees.
"""

from dataclasses import dataclass

from transformers import AutoTokenizer, PreTrainedTokenizerBase

_TOKENIZER: PreTrainedTokenizerBase | None = None


def get_tokenizer() -> PreTrainedTokenizerBase:
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    return _TOKENIZER


@dataclass
class Chunk:
    text: str
    section: str
    chunk_index: int
    token_count: int


def chunk_section(
    text: str,
    section: str,
    tokenizer: PreTrainedTokenizerBase | None = None,
    min_tokens: int = 300,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
) -> list[Chunk]:
    tokenizer = tokenizer or get_tokenizer()
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []

    n = len(token_ids)
    stride = max_tokens - overlap_tokens
    windows: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(start + max_tokens, n)
        windows.append((start, end))
        if end == n:
            break
        start += stride

    # A too-small trailing window gets folded into the previous one instead
    # of standing alone as a tiny chunk.
    if len(windows) > 1:
        last_start, last_end = windows[-1]
        if last_end - last_start < min_tokens:
            prev_start, _prev_end = windows[-2]
            windows[-2] = (prev_start, last_end)
            windows.pop()

    chunks = []
    for index, (win_start, win_end) in enumerate(windows):
        chunk_ids = token_ids[win_start:win_end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(
            Chunk(text=chunk_text, section=section, chunk_index=index, token_count=len(chunk_ids))
        )
    return chunks
