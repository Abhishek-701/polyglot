# Progress

## M1: Pipeline skeleton and replay harness

**Status:** complete, ready for human review.

### What was built

- `polyglot/core/pipeline.py` — `Pipeline` orchestrator implementing the turn
  lifecycle (SPEC.md Section 4.3) with injected fakes: VAD tap on the frame
  stream, ASR partial/final handling, turn-detector commit (threshold or
  `is_final`), retrieval at commit, LLM streaming into a single accumulated
  string, TTS synthesis, and a `TurnRecord` + `TurnResult` (record + audio
  frames) per turn. Wrapped in `tracer.span("turn", ...)`.
- `polyglot/fakes.py` — moved from `tests/fakes.py` so both unit tests and
  `eval/replay.py` can use it without eval depending on the test tree. Added
  `ScriptedFakeASREngine`, which yields a different scripted turn per call
  (one per scenario turn). Fixed a real bug found while wiring this up (see
  below): the ASR fakes now drain the `frames` iterator they're given.
- `polyglot/transport/replay_adapter.py` — `ReplayAdapter` drives a `Pipeline`
  from a `Scenario` (own minimal M1 schema: `id`, `lang`, `turns: [{user_text}]`),
  maintaining history across turns. `ScenarioTurn`/`Scenario` pydantic models
  live here since they're the adapter's input contract.
- `eval/replay.py` — CLI (`make replay SCENARIO=... PROFILE=... MODE=...`):
  loads a scenario YAML, loads the naive/tuned config, wires an all-fakes
  `Pipeline` (scripts the fake ASR from each turn's `user_text`), runs it via
  `asyncio.run`, writes `reports/<run_id>/events.jsonl` and `turns.jsonl`.
- `eval/scenarios/m1-smoke.yaml` — two-turn smoke scenario.
- `eval/metrics/latency.py` — `perceived_latency_ms_by_turn(events)`: groups
  events by `turn_id`, computes `audio_out_first_frame.t_ms - vad(speech_end).t_ms`
  per SPEC.md Section 12's formula.
- `tests/integration/test_pipeline.py` — a normal turn produces every event
  kind the M1 pipeline emits, and `TurnRecord.timings` covers the key spans.
- `tests/integration/test_replay_determinism.py` — SPEC.md Section 11.2: same
  scenario run twice via fresh `SimulatedClock`/`EventLog`/`Pipeline` produces
  identical transcripts, passages, and assistant text (turn_id/session_id
  excluded from the comparison; timings are trivially identical here since
  fakes never call `clock.advance()`).
- `reports/.gitkeep` so the (gitignored-contents) output directory exists.

### Test and lint status

- `uv run ruff check .`, `uv run ruff format --check .` — clean.
- `uv run mypy polyglot/core` (strict) — clean, 7 source files.
- `uv run pytest` — 30 passed (27 from M0 + 3 new integration tests; the M0
  count also includes fixes below).
- Manually ran `uv run python -m eval.replay --scenario eval/scenarios/m1-smoke.yaml
  --profile naive --mode simulated`: 2 turns, 18 events, output written under
  `reports/m1-smoke-simulated-<ts>/`. Inspected both `turns.jsonl` and the
  event kind sequence by hand.

### Bugs found and fixed while building this

- **`FakeASREngine`/`ScriptedFakeASREngine` never consumed the `frames`
  argument.** `Pipeline.run_turn` wraps the frame stream in a `tapped_frames()`
  generator that runs `vad.process(frame)` as a side effect while forwarding
  frames to the ASR engine — but the ASR fakes just ignored `frames` and
  yielded their scripted partials directly, so the wrapping generator was
  never iterated and VAD events never fired. Caught by
  `tests/integration/test_pipeline.py` asserting on the full expected event
  kind set. Fixed by having both fakes drain `frames` (`async for _ in frames:
  pass`) before yielding, matching how a real streaming ASR engine would
  actually consume the stream.
- **Flaky `test_real_clock_sleep_advances_time`** (from M0): asserted
  `now_ms() >= 20` after `sleep_ms(20)`; Windows timer resolution can wake
  `asyncio.sleep` a few ms early, so it intermittently failed with e.g. `15 >=
  20`. Loosened to `>= 10` — the test only needs to confirm real time passed,
  not measure exact sleep precision.
- **Ruff was reformatting `SPEC.md`'s embedded Python code fences** on
  `ruff format`, which would have silently rewritten the spec document (owned
  by the human, must not be touched). Added `extend-exclude = ["*.md"]` to
  `[tool.ruff]` in `pyproject.toml`.

### Decisions made / spec interpretations

- **M1 scenario schema is deliberately smaller than SPEC.md Section 10.5.**
  Section 10.5 scenarios reference real `user_audio` WAV files and support
  barge-in timing (`after_agent_start_ms`); neither exists yet (no real audio,
  no real ASR, no `BargeInMonitor`). M1's `Scenario`/`ScenarioTurn` only has
  `user_text`, consumed by `eval/replay.py` to script the fake ASR, and
  `ReplayAdapter` only supports sequential turns. The real Section 10.5 schema
  (audio files, barge-in timing) will be built out starting M2 (real audio)
  and M6 (barge-in), extending rather than replacing this.
- **`core/config.py`'s `load_settings(profile)` is loaded by `eval/replay.py`
  but doesn't yet change fake behavior.** The naive/tuned flags don't have
  anything to switch between until real components exist (M5 is where most of
  them start mattering). For now the CLI just prints the resolved flags to
  confirm config loading is wired end to end.
- **M1's event-kind coverage check excludes `bargein` and `tool_call`.**
  Those stages don't exist until M6 and M4 respectively; SPEC.md Section 7.3's
  full event kind list can't be fully produced until then. Documented in
  `tests/integration/test_pipeline.py`'s docstring.
- **LangIDTracker, ClauseSplitter, DialoguePolicy (LangGraph), history
  compaction, speculative retrieval, and BargeInMonitor are all absent from
  `Pipeline` in M1** — language is passed straight from ASR's reported `lang`,
  the full LLM output is sent to TTS as one chunk, retrieval happens once at
  turn commit (not speculatively), and there is no barge-in handling. Each is
  explicitly scoped to a later milestone in SPEC.md Section 15 (M4/M5/M6).
- **`make` is not installed in this dev environment** (Windows, no `make` in
  PATH). Verified the `replay` target's actual command directly via
  `uv run python -m eval.replay ...` instead. GitHub Actions' `ubuntu-latest`
  runners have `make` preinstalled, so CI is unaffected; flagging so a human
  either installs `make` locally (e.g. via a package manager) or we
  reconsider Makefile as the primary interface later.

### Unverified / deferred

- `eval/replay.py --mode realtime` is implemented (constructs a `RealClock`)
  but not exercised yet — nothing in the M1 pipeline actually waits on real
  time (fakes are instant), so realtime vs simulated mode currently produce
  the same all-zero timings. This becomes meaningful once real components
  exist.
- No WAV output yet (Section 11.1 says replay should also produce "agent
  audio WAV" per session) — `TurnResult.audio_frames` are collected but never
  written to disk. Deferred until there's real synthesized audio worth saving
  (fake TTS just encodes text as UTF-8 bytes, not real PCM).

### Next milestone

M2: Real audio front end (Silero VAD, faster-whisper streaming with
LocalAgreement, hallucination guard, language tracker) — not started.

## M0: Scaffold and contracts

**Status:** complete, ready for human review.

### What was built

- Repo skeleton: `pyproject.toml` (uv, Python pinned to 3.12), `Makefile`,
  `.pre-commit-config.yaml`, `.gitignore`, `.env.example`, `docker-compose.yml`
  (service definitions only, nothing wired to code yet), `README.md` stub,
  `.github/workflows/ci.yml` (lint + test on push/PR).
- `polyglot/core/types.py` — all seven Pydantic v2 models from SPEC.md 7.1.
- `polyglot/core/interfaces.py` — all eight `Protocol`s from SPEC.md 7.2,
  marked `@runtime_checkable`.
- `polyglot/core/clock.py` — `RealClock` (real `time.monotonic`/`asyncio.sleep`)
  and `SimulatedClock` (manual `advance()`, waiters resolved only on advance).
  This is the only file under `polyglot/` allowed to call
  `time.time`/`time.monotonic`/`asyncio.sleep`.
- `polyglot/core/events.py` — `Event` model, `EventLog` (JSONL append/read).
- `polyglot/core/config.py` — loads `config/default.yaml` overlaid by
  `config/profiles/{naive,tuned}.yaml` into a `Settings` (pydantic-settings)
  object; env overrides via `POLYGLOT_` prefix.
- `config/default.yaml`, `config/profiles/naive.yaml`, `config/profiles/tuned.yaml`
  — the Section 9 flag table, naive all-off / tuned all-on.
- `tests/fakes.py` — `FakeClock` (alias of `SimulatedClock`), `FakeVAD`,
  `FakeASREngine`, `FakeTurnDetector`, `FakeRetriever`, `FakeLLMClient`,
  `FakeTTSEngine`, `FakeTracer`.
- `tests/unit/`: `test_types.py`, `test_interfaces.py` (isinstance checks against
  Protocols), `test_clock.py`, `test_clock_usage.py` (static grep guard),
  `test_config.py`, `test_events.py`. 27 tests total.
- `docs/decisions/001-own-pipeline.md` — ADR for the transport/pipeline
  separation (SPEC.md 4.1).

### Test and lint status

- `uv run ruff check .` — clean.
- `uv run ruff format --check .` — clean.
- `uv run mypy polyglot/core` (strict) — clean, 6 source files.
- `uv run pytest` — 27 passed.
- Manually verified `test_clock_usage.py` actually catches a violation: temporarily
  added a `time.monotonic()` call to `events.py`, confirmed the test failed,
  reverted, confirmed green again.
- `make setup`, `make lint`, `make test` map to the above and were run directly
  via `uv run` (not yet run through `make` itself in this session — trivial
  wrapper, same commands).

### Decisions made

- **Python pinned to 3.12** via `uv python pin 3.12`, even though the system's
  default Python is 3.13.6. Rationale: safer compatibility once torch/vLLM/
  faster-whisper are added in later milestones; revisit if any of those
  publish 3.13 wheels before M2.
- **`core/config.py` location**: SPEC.md Section 6's repo tree doesn't name an
  exact file for the Section 9 config loader. Placed it under `polyglot/core/`
  since it's transport-independent and needed by every stage.
- **`LLMDelta.tool_call` / `TurnRecord.tool_calls` typed as `dict[str, Any]`**
  instead of bare `dict` as literally written in SPEC.md 7.1, because
  `mypy --strict` (required on `core/`) rejects unparameterized generics.
  Same intent (arbitrary JSON-ish payload), just satisfies strict mode.
- **`AbstractContextManager[Any]` instead of `ContextManager`** for
  `Tracer.span`'s return type: `collections.abc` has no `ContextManager`
  member (that was a mistake to fix, not a deliberate deviation) —
  `contextlib.AbstractContextManager` is the correct modern equivalent.
- **GPU provider: recommended Modal** (matches SPEC.md Section 5's primary
  choice; scale-to-zero fits a solo project's budget). Not exercised in M0;
  needed starting M2 (ASR/embeddings) and confirmed again before M4/M5.
- **Hosted LLM fallback**: OpenAI-compatible endpoint, no URL/key configured
  yet — add when `polyglot/llm/client.py` is built in M4.
- **Fourth language: Tagalog (`tl`)** — reflected in `config/default.yaml`.

### Unverified / deferred

- `docker-compose.yml` services (postgres, redis, langfuse, livekit) are
  defined but never started or smoke-tested in M0 — nothing depends on them
  yet. First real check happens in M3 (postgres/redis) and M4 (livekit/langfuse).
- `.github/workflows/ci.yml` has not run on an actual GitHub Actions runner
  (no remote pushed yet); the `astral-sh/setup-uv@v3` action tag should be
  re-verified against the current latest tag before the first real CI run.
- No `config/sources.yaml`, `config/languages.yaml`, or `config/thresholds.yaml`
  yet — out of scope for M0.

### Open questions (from SPEC.md Section 16, still unanswered)

1. Knowledge base URLs for `config/sources.yaml` — human fills in before M3.
2. Who records golden audio, and for which languages — needed before M3/M4
   golden set work.
3. Public deployment target/provider details — noted as "public eventually"
   but no specific provider chosen; relevant starting M8.

### Next milestone

M1: Pipeline skeleton and replay harness (`core/pipeline.py`, `replay_adapter.py`,
`eval/replay.py`, event-log-based `latency.py`, determinism test) — not started.
