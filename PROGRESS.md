# Progress

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
