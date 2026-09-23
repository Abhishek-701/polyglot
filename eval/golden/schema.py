"""golden.jsonl schema. See SPEC.md Section 10.3.

expected_passage_ids reference RetrievalStore chunk ids ("<doc_id>#<chunk_index>",
see polyglot/retrieval/store.py) — so a real golden set can only be built
after a real corpus has been ingested (chunk boundaries need to exist first).
"""

from typing import Literal

from pydantic import BaseModel


class Speaker(BaseModel):
    native_lang: str
    recorded_by: Literal["human", "tts", "dataset"]


class GoldenItem(BaseModel):
    id: str
    lang: str
    audio_path: str
    reference_transcript: str
    code_switched: bool = False
    speaker: Speaker
    question_type: str
    expected_doc_ids: list[str]
    expected_passage_ids: list[str]
    reference_answer: str
    answerable: bool = True
