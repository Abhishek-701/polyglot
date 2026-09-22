"""Event log. See SPEC.md Section 7.3 and Section 12.

Every pipeline step appends a typed Event with t_ms from the injected Clock.
All metrics are computed from these logs; no ad hoc timers (CLAUDE.md
architecture rules). Not yet wired to a real pipeline (that's M1).
"""

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

EventKind = Literal[
    "vad",
    "asr_partial",
    "asr_final",
    "eou",
    "retrieval_start",
    "retrieval_end",
    "llm_first_token",
    "tts_first_frame",
    "audio_out_first_frame",
    "bargein",
    "tool_call",
    "disclosure",
]

# Span names used as keys in TurnRecord.timings (SPEC.md Section 12).
SpanName = Literal[
    "turn",
    "vad.speech_end",
    "asr.first_partial",
    "asr.final",
    "eou.decision",
    "retrieval.speculative",
    "retrieval.final",
    "policy.graph",
    "llm.ttft",
    "llm.total",
    "tts.ttfb",
    "audio.first_frame_out",
    "bargein",
    "compaction",
]


class Event(BaseModel):
    kind: EventKind
    t_ms: int
    session_id: str
    turn_id: str | None = None
    data: dict[str, Any] = {}


class EventLog:
    """Appends typed Events as JSONL to a per-session file."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: Event) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(event.model_dump_json() + "\n")

    def read_all(self) -> list[Event]:
        if not self._path.exists():
            return []
        with self._path.open("r", encoding="utf-8") as f:
            return [Event.model_validate(json.loads(line)) for line in f if line.strip()]
