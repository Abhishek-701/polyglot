"""Per-turn perceived latency from event logs. See SPEC.md Section 12.

perceived latency = audio.first_frame_out.t_ms - vad.speech_end.t_ms, per
committed turn. Only a basic per-turn number for now (M1); percentiles,
per-stage breakdown, and the waterfall chart are M5/M7 work.

CLAUDE.md rule: never report latency from simulated-clock replay runs, only
from realtime mode on real hardware. This module just computes the delta
from whatever events it's given; the caller is responsible for only trusting
the result when the events came from a realtime run.
"""

from collections import defaultdict

from polyglot.core.events import Event


def perceived_latency_ms_by_turn(events: list[Event]) -> dict[str, int]:
    by_turn: dict[str, list[Event]] = defaultdict(list)
    for event in events:
        if event.turn_id is not None:
            by_turn[event.turn_id].append(event)

    result: dict[str, int] = {}
    for turn_id, turn_events in by_turn.items():
        speech_end = next(
            (e for e in turn_events if e.kind == "vad" and e.data.get("vad_kind") == "speech_end"),
            None,
        )
        audio_out = next((e for e in turn_events if e.kind == "audio_out_first_frame"), None)
        if speech_end is not None and audio_out is not None:
            result[turn_id] = audio_out.t_ms - speech_end.t_ms

    return result
