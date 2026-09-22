"""Core data models. See SPEC.md Section 7.1.

Pydantic v2 models shared by every pipeline stage. No transport or model-library
imports here: this module must stay importable with only pydantic installed.
"""

from typing import Any, Literal

from pydantic import BaseModel


class AudioFrame(BaseModel):
    pcm: bytes
    sample_rate: int = 16000
    t_ms: int


class VADEvent(BaseModel):
    kind: Literal["speech_start", "speech_end"]
    t_ms: int
    prob: float


class TranscriptPartial(BaseModel):
    text: str
    stable_prefix: str
    lang: str
    lang_prob: float
    is_final: bool
    t_start_ms: int
    t_end_ms: int
    no_speech_prob: float | None = None
    avg_logprob: float | None = None


class Passage(BaseModel):
    id: str
    doc_id: str
    text: str
    source_url: str
    score: float


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    lang: str | None = None
    interrupted: bool = False


class LLMDelta(BaseModel):
    text: str | None = None
    tool_call: dict[str, Any] | None = None
    finish_reason: str | None = None


class TurnRecord(BaseModel):
    turn_id: str
    session_id: str
    lang: str
    user_text: str
    assistant_text_generated: str
    assistant_text_spoken: str
    intent: str | None
    passages: list[Passage]
    tool_calls: list[dict[str, Any]]
    code_switched: bool
    interrupted: bool
    timings: dict[str, int]
