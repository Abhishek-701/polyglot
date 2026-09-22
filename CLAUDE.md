# CLAUDE.md

Project instructions for Claude Code. Read this and `SPEC.md` at the start of every session. Check `PROGRESS.md` to see which milestone is current.

## Project

Polyglot is a real-time cross-lingual voice agent (English, Spanish, Hindi, plus one more language) answering airline disruption questions from an English policy corpus. The headline deliverables are a measured latency ablation (naive vs tuned, per stage) and a deterministic replay evaluation harness. Full detail in `SPEC.md`.

## Workflow rules

1. Work one milestone at a time, in the order in `SPEC.md` Section 15. Stop at the end of each milestone for human review.
2. Before stopping, update `PROGRESS.md` with: what was built, test and lint status, what is unverified, decisions made, and open questions.
3. Verify third-party APIs against the installed package (read the source or docs in the environment). LiveKit, faster-whisper, vLLM, and TTS libraries change often. Pin exact versions after verifying.
4. Never write a metric, benchmark number, or claim about model quality into any file unless it came from a script run whose output is saved under `reports/`.
5. If a task needs something you cannot do (GPU access, microphone, logging in to a gated dataset, recording audio, listening to output), stop and ask. Do not simulate it and present the result as real.
6. Do not invent URLs for the knowledge base. `config/sources.yaml` is filled in by the human.
7. Keep diffs focused. Do not refactor unrelated code while implementing a milestone.

## Architecture rules

- `polyglot/core/` must not import LiveKit or any transport. Transports are adapters in `polyglot/transport/`.
- All time goes through the injected `Clock`. Never call `time.time`, `time.monotonic`, or `asyncio.sleep` outside `core/clock.py`. A test enforces this.
- All metrics are computed from event logs. Do not add ad hoc timers.
- Every latency optimization sits behind a flag in `config/profiles/`. The naive profile must keep working forever.
- Nothing blocking on the audio path. CPU-heavy model calls run in an executor or a separate service.
- Components depend on the Protocols in `core/interfaces.py`, not on concrete classes.
- The audio pipeline is plain asyncio. LangGraph is used only for the dialogue policy.

## Code conventions

- Python 3.11+, managed with `uv`. Type hints everywhere. `mypy --strict` on `polyglot/core/`.
- Pydantic v2 for data models and config.
- `ruff` for lint and format.
- Async-first. Use `asyncio.TaskGroup` and explicit cancellation. Every cancellable task handles `CancelledError` cleanly, since barge-in depends on it.
- Structured logging (JSON) with `session_id` and `turn_id` on every line.
- Tests: `pytest`, `pytest-asyncio`, `hypothesis` for the splitter, local agreement, and playback ledger. Network tests marked `@pytest.mark.network`.

## Commands

```
make setup
make lint
make test
make up / make down
make ingest
make agent
make replay SCENARIO=<path> PROFILE=naive|tuned MODE=simulated|realtime
make eval-smoke
make eval-full
make ablation
make report RUN=<run_id>
```

Run `make lint` and `make test` before declaring any task done.

## Data rules

- Never commit datasets, model weights, or golden audio. Use download scripts; golden audio is gitignored or in Git LFS.
- Record the license of every dataset and model used in `docs/data_licenses.md`.
- Do not use XTTS-v2 (non-commercial license).
- Eval runs use mock tools only.

## Writing style for docs, READMEs, and reports

- Plain, direct, human tone. No marketing language.
- Do not use em dashes. Use commas, colons, parentheses, or separate sentences.
- Report per-language results, not just averages. State weaknesses plainly.
- Every results table names the hardware, model versions, config profile, and git commit it came from.
