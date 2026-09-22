# ADR 001: Own the pipeline, keep LiveKit as a transport adapter only

## Status

Accepted (M0).

## Context

LiveKit Agents' `AgentSession` bundles transport, endpointing, and interruption
handling into one abstraction. Using it directly would be less code, but it
would also make the pipeline hard to test deterministically and hard to
ablate component by component: the latency breakdown and leave-one-out
ablation table are the project's main deliverable (SPEC.md Section 1.3), and
they require running the exact same pipeline logic in both a live LiveKit
session and an offline replay harness.

## Decision

`polyglot/core/` contains the pipeline (`Pipeline`, VAD, ASR, turn detection,
retrieval, dialogue policy, LLM, TTS, barge-in) and does not import LiveKit or
any other transport library. Every component depends on the `Protocol`
definitions in `core/interfaces.py`, not on concrete classes. LiveKit only
appears in `polyglot/transport/livekit_adapter.py`, which feeds `AudioFrame`s
into the same `Pipeline` object that `polyglot/transport/replay_adapter.py`
drives in eval mode. Both adapters share one injected `Clock` (`core/clock.py`):
`RealClock` for live sessions, `SimulatedClock` for deterministic replay.

## Consequences

- We reimplement endpointing glue and interruption handling that
  `AgentSession` would otherwise provide. This reimplementation is itself
  part of the project's value: it is what makes barge-in and turn detection
  independently measurable and swappable behind config flags
  (SPEC.md Section 9).
- The replay harness (`eval/replay.py`) can run the production pipeline
  against recorded audio with no LiveKit server running, which is what makes
  CI regression testing and the naive-vs-tuned ablation possible.
- Any future transport (e.g. a SIP adapter) only needs to implement the same
  `AudioFrame` in / out contract; no pipeline code changes.
