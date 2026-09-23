# Progress

## M3: Knowledge base and retrieval

**Status: infrastructure complete, real corpus ingested, still blocked on
the golden set.** `config/sources.yaml` is now filled in with 5 verified
real URLs (below) and the real corpus has been ingested into pgvector.
M3's formal accept criteria (a real recall@k table, an ADR naming the
chosen default bridge) still can't be produced honestly — they need a
human-reviewed golden set, which needs a working LLM to draft candidates
from (M4) or manual authoring — see "Blocked" below.

### config/sources.yaml filled in, real ingestion run

The human asked for KB URLs to be searched rather than provided directly.
Per CLAUDE.md rule 6 (Claude Code doesn't invent KB URLs) every candidate
was verified by actually fetching it and running it through
`polyglot/retrieval/extract.py` — not just checked by eye — before being
added. Rejected candidates and why:

- **EUR-Lex's raw EU261 legal text** (`eur-lex.europa.eu/legal-content/...`)
  returns HTTP 202 with an empty body to a plain GET, consistently, across
  several URL variants. Used the EU's official "Your Europe" citizen-portal
  summary instead (SPEC.md 10.1 explicitly allows "EU261 text **or an
  official summary**") — extracts cleanly into 47 real sections.
- **United's Contract of Carriage page** renders 0 usable sections — a JS
  app with no server-rendered content, confirmed by fetching and running
  our own extractor against it, not by assumption.
- **A dedicated DOT refund-rule page** (`transportation.gov/airconsumer/refundsfinalruleapril2024`)
  extracts to ~1400 characters, almost all boilerplate — the actual rule
  text isn't in the page shell. Used DOT's "Fly Rights" guide instead,
  which extracts to 20 real sections including "Delayed and Cancelled
  Flights" and "Contract Terms" and covers refunds within them.

Final 5 sources (`config/sources.yaml`): `dot-fly-rights`,
`eu261-air-passenger-rights`, `american-airlines-conditions-of-carriage`,
`delta-contract-of-carriage-domestic`, `southwest-contract-of-carriage`.

Real `python -m polyglot.retrieval.ingest` run: 246 chunks total (61 + 40 +
41 + 47 + 57). Spot-checked retrieval quality with `MultilingualBridge`
against the real ingested corpus — a Spanish query ("¿Puedo obtener un
reembolso por mi vuelo cancelado?") correctly retrieves DOT/American/EU261
refund and cancellation passages with cosine scores 0.70-0.75; an English
EU-delay question correctly surfaces the EU261 compensation-amount table
section. Not a golden-set recall@k number, but real evidence the corpus and
bridges work together correctly.

Three real bugs found and fixed while doing this (all in
`polyglot/retrieval/ingest.py`):

- **Chunk ids collided across sections.** `chunk_section()` restarts
  `chunk_index` at 0 for every section it's called on; `ingest_document`
  was calling it once per extracted section and using each chunk's own
  (section-local) index directly as the store's global chunk index —
  section 2's first chunk got the same id as section 1's first chunk
  (`doc_id#0`), and `upsert_document` hit a real `UniqueViolation` on the
  first live ingest run. Fixed by renumbering chunks globally per document
  in `ingest_document` (`enumerate()` over the combined chunk list) rather
  than trusting `chunk.chunk_index`.
- **`transportation.gov` 403'd our original UA string** (`"polyglot-ingest/0.1"`
  alone). A `"Mozilla/5.0 (Windows NT 10.0; Win64; x64) polyglot-ingest/0.1"`
  string works reliably; a fuller Chrome/AppleWebKit-spoofing UA string
  intermittently got 403'd on a *different* source (American) — picked the
  simpler string that passed all 5 sources consistently rather than the
  more elaborate one.
- **Content-hash idempotency was hashing raw response bytes**, which some
  sites (confirmed on American's page) vary on every request — likely
  nonces/timestamps in unrelated `<script>` tags — even though the real
  policy text is byte-identical. Every re-ingest was reported as "changed"
  for that source. Fixed by hashing the *extracted* text instead of the raw
  bytes; verified two fetches of American's page produce different raw
  bytes but identical extracted text.

A fourth bug — this one in the test suite — surfaced after ingesting the
real corpus: `tests/integration/test_retrieval_bridges.py`'s synthetic
fixture used realistic-sounding refund language ("Passengers whose flights
are cancelled..."), which worked fine against an empty test DB but started
losing to the now-present real KB content in the top-k for a
realistic-sounding query — correct retrieval behavior, broken test
assumption. Fixed by rewriting the fixture and query with invented
brand/policy terms ("Zyloport", "Flexipass", "Category Nine") that can't
semantically collide with anything real, regardless of what else is in the
shared dev database. All 7 network tests pass again with the real KB loaded.

### What was built

- `polyglot/retrieval/extract.py` — HTML section extraction (BeautifulSoup:
  strip nav/header/footer/script/style/aside/form, split on h1-h6 into
  `(heading, text)` pairs, "document" fallback if no headings) and PDF
  extraction (pypdf, one section per page — PDFs don't have reliable heading
  markup without layout analysis).
- `polyglot/retrieval/chunking.py` — `chunk_section`: 300-500 token windows
  with 50 token overlap (SPEC.md Section 8.7), tokenized with BGE-M3's own
  tokenizer so chunk sizes match what the embedding model actually sees. A
  trailing window smaller than `min_tokens` is folded into the previous
  chunk instead of standing alone (occasionally producing a chunk bigger
  than `max_tokens` — a deliberate tradeoff over emitting tiny fragments).
- `polyglot/retrieval/store.py` — `RetrievalStore`: Postgres + pgvector,
  HNSW index on cosine distance, `documents`/`chunks` tables, content-hash
  based idempotency. ADR: `docs/decisions/002-pgvector-store.md`.
- `polyglot/retrieval/ingest.py` — downloads a source, extracts, chunks,
  embeds (BGE-M3), upserts; skips re-ingesting unchanged documents (content
  hash). Refuses to run against an empty `config/sources.yaml` (see
  "Blocked").
- `polyglot/retrieval/bridges/multilingual.py` — embed the query directly
  with BGE-M3, dense search.
- `polyglot/retrieval/bridges/mt_bridge.py` — `Translator` (NLLB-200-distilled-600M)
  + BGE-M3 embed + dense search.
- `polyglot/retrieval/bridges/hybrid.py` — reciprocal rank fusion (k=60) of
  mt + multilingual, optional rerank with `bge-reranker-v2-m3`.
- `eval/metrics/retrieval.py` — recall@1/5/10, MRR, grouped by
  language/bridge.
- `eval/golden/schema.py` — `GoldenItem`/`Speaker` pydantic models, exactly
  SPEC.md Section 10.3's schema. `expected_passage_ids` reference
  `RetrievalStore` chunk ids (`<doc_id>#<chunk_index>`).
- `eval/golden/validate.py` — validates `golden.jsonl` against the schema
  and reports pass/fail against Section 10.3's coverage targets (200 items,
  >=40/language, >=20% code-switched for es/hi, >=15% unanswerable).
- `eval/retrieval_eval.py` — the real recall@k/MRR report CLI, ready to run
  the moment sources + a golden set exist (see "Blocked").
- `config/sources.yaml` — empty template (`documents: []`) with the expected
  shape commented in; `ingest.py` refuses to run while it's empty.
- Tests: `test_chunking.py` (fake word-level tokenizer, no network — window/
  overlap/merge logic swept across many input sizes), `test_extract.py`
  (in-memory HTML fixtures + a blank-page PDF), `test_retrieval_metrics.py`,
  `test_golden_validate.py`, and `tests/integration/test_retrieval_bridges.py`
  (`@pytest.mark.network`: a synthetic one-document corpus ingested into the
  real Postgres/pgvector instance, all three bridges — including the real
  NLLB translation and BGE-M3/reranker models — correctly retrieve the
  relevant passage for a Spanish query).

### Environment work

- **Docker Desktop wasn't running; started it.** Low-risk (starting an
  application), needed to get Postgres up.
- **Port 5432 conflict: this machine already runs a native Windows
  PostgreSQL service on 5432**, unrelated to this project. Both it and our
  container's port mapping bound `0.0.0.0:5432`; the native service silently
  won host-originated connections (Docker's own `docker exec` into the
  container worked fine — only host->container routing was broken). Didn't
  touch the pre-existing native service; remapped our compose file to host
  port 5433 instead (`docker-compose.yml`, `.env.example`).
  `POLYGLOT_DB_DSN` env var overrides the default if needed.
- Verified real model downloads/APIs work on this CPU-only Windows box:
  `BAAI/bge-m3` (1024-dim, sentence-transformers), `facebook/nllb-200-distilled-600M`
  (translation, direct `AutoModelForSeq2SeqLM.generate()`), `BAAI/bge-reranker-v2-m3`
  (`CrossEncoder`), pgvector 0.8.3 + psycopg3 + pgvector-python.

### Test and lint status

- `uv run ruff check .`, `uv run ruff format --check .` — clean.
- `uv run mypy polyglot/core` (strict) — clean, unchanged (retrieval code is
  outside `core/`, same scoping decision as M2).
- `uv run pytest` (network deselected) — 66 passed.
- `uv run pytest -m network` — 7 passed (4 from M2 + 3 new retrieval bridge
  tests: multilingual, mt, and hybrid-with-rerank all correctly find the
  relevant passage for a real Spanish query against a real synthetic corpus).

### Bugs found and fixed while building this

- **`transformers==5.17.0` removed the generic `pipeline("translation", ...)`
  task** (`KeyError: Unknown task translation`). Used
  `AutoModelForSeq2SeqLM.generate()` directly instead — more explicit control
  over `forced_bos_token_id` anyway.
- **Trafilatura (originally planned for HTML extraction) collapses headings
  into flat paragraph text** even in its XML/HTML output modes with
  `include_formatting=True` — confirmed by direct testing, not just reading
  docs. Since heading-preserving sections are a hard SPEC.md 8.7 requirement,
  switched to BeautifulSoup with direct heading-tag splitting instead. Not
  a bug exactly (trafilatura does what it's designed to do — boilerplate
  removal, not structure preservation), but a real planning correction made
  before writing dependent code, not after.

### Decisions made / spec interpretations

- **NLLB language codes confirmed against the installed tokenizer's vocab**:
  `eng_Latn`, `spa_Latn`, `hin_Deva`, `tgl_Latn`. Note NLLB uses `tgl_Latn`
  for Tagalog while FLEURS (M2) uses `fil_ph` — different datasets' naming
  for the same language, not a bug, just worth remembering when cross-
  referencing the two.
- **Chunk token counts use BGE-M3's own tokenizer**, not a generic one
  (tiktoken, word count) — SPEC.md doesn't specify which tokenizer "300 to
  500 tokens" refers to; using the embedding model's own tokenizer keeps
  chunk sizing meaningful relative to what it actually encodes.
- **`eval/golden/validate.py`'s coverage checks are informational (pass/
  fail printout), not a hard gate.** SPEC.md Section 10.3 says "Claude Code
  builds the schema, validators, and a script that drafts candidate
  questions... for human review" — the human still finalizes the set, so
  this validates structure/coverage, it doesn't approve content.
- **The golden-question-drafting generator (Section 10.3: "drafts candidate
  questions and reference answers from the corpus") was not built.** It
  needs a working LLM (Section 5: Qwen2.5-7B via vLLM, or the hosted
  OpenAI-compatible fallback from the M0 open questions) to actually
  generate candidate Q&A pairs — no LLM client exists yet (that's M4). Built
  the schema and validator now since they don't need one; the drafting
  script is deferred to when M4's LLM client exists, or sooner if a hosted
  API key is provided.

### Blocked

- **There's still no real golden set**, so still no real recall@k table —
  M3's actual accept criterion. SPEC.md Section 10.3's `expected_passage_ids`
  now *can* reference real chunk ids (the corpus is ingested), but building
  ~100-200 human-reviewed questions needs either an LLM to draft candidates
  from the real corpus (SPEC.md 10.3's "Claude Code builds... a script that
  drafts candidate questions" — no LLM client exists until M4) or manual
  human authoring now. `eval/retrieval_eval.py` is ready and will produce a
  real report the moment `eval/golden/golden.jsonl` exists.
- **The ADR recording "the chosen default bridge and why"** can't be written
  honestly without that recall@k data. Will follow directly from the first
  real `eval/retrieval_eval.py` run.
- Everything else that real recall@k data depends on is now proven working:
  real corpus ingested (246 chunks, 5 documents), retrieval spot-checks look
  correct, all three bridges pass their integration tests against real
  models. What's left is purely the golden-set question: get an LLM wired
  up early (bring M4's LLM client forward, or provide a hosted API key now)
  to draft candidates for human review, or author them by hand.

### Next milestone

M4 (first full voice turn) prerequisites — LLM client, LangGraph policy,
TTS, LiveKit adapter — don't depend on the KB and could proceed in parallel.
Bringing the LLM client forward from M4 would also unblock the golden-set
drafting script above, closing out M3's remaining accept criteria at the
same time.

## M2: Real audio front end

**Status:** complete, ready for human review.

### What was built

- `polyglot/audio/frames.py` — `pcm16_to_float32`/`float32_to_pcm16`,
  `resample` (via torchaudio, already a dependency), `SampleBuffer` for
  accumulating streaming samples.
- `polyglot/audio/vad.py` — `SileroVAD`. Verified against installed
  `silero-vad==6.2.2`: the model requires exact 512-sample (32ms) windows at
  16kHz (raises `ValueError` below that). Reimplemented the speech
  start/end state machine ourselves instead of using `silero_vad.VADIterator`,
  because `VADIterator` only exposes `min_silence_duration_ms` — SPEC.md
  Section 8.1 also wants `min_speech_ms` enforced, which needed our own
  hysteresis logic.
- `polyglot/asr/local_agreement.py` — `local_agreement_prefix` (word-level
  longest common prefix) + stateful `LocalAgreement` wrapper. Hypothesis
  property test in `tests/unit/test_local_agreement.py`.
- `polyglot/asr/hallucination_guard.py` — `check_hallucination`, pure
  function per SPEC.md Section 8.3 (no_speech_prob, avg_logprob,
  compression_ratio, blocklist checks, in that order). Deliberately doesn't
  do the "outside VAD speech region" check (needs VAD context this function
  doesn't have) or event logging (no EventLog access) — both are for
  whatever wires this into the real pipeline in M4.
- `polyglot/asr/faster_whisper_engine.py` — `FasterWhisperEngine`
  implementing the `ASREngine` protocol. Verified against installed
  `faster-whisper==1.2.1`: `WhisperModel.transcribe()` takes a float32
  `np.ndarray` directly, and its default `no_speech_threshold`/
  `log_prob_threshold`/`compression_ratio_threshold` (0.6, -1.0, 2.4) are
  exactly SPEC.md 8.3's guard thresholds — the spec was written against
  faster-whisper's own defaults. Decodes every `asr_step_ms` (default 200ms)
  in the default executor (blocking CPU call, per CLAUDE.md's "nothing
  blocking on the audio path" rule).
- `polyglot/langid/tracker.py` — `LangIDTracker` (fastText LID over
  sliding word windows, per-language share + code-switch detection, SPEC.md
  Section 8.4). Verified `fasttext-wheel==0.9.2` against installed
  `numpy==2.5.3`: `_FastText.predict(str, ...)` is broken on numpy>=2
  (`np.array(probs, copy=False)` raises `ValueError`). The multi-string
  input path doesn't hit that code, so this always calls `predict([text],
  ...)` and unwraps, even for one window.
- `config/languages.yaml` — seeded `hallucination_blocklist` per language
  (en/es/hi/tl). Other sections (VAD/turn thresholds, backchannels) left for
  M5/M6 when they're tuned.
- `scripts/download_datasets.py` — downloads (to gitignored `data/cache/`,
  never committed): fastText LID model, a FLEURS subset per language (via
  `datasets`, decoded with `soundfile` — see bug note below), the ESC-10
  subset of ESC-50 (real ambient noise for the hallucination test), and
  generated near-silence clips.
- `eval/metrics/asr.py` / `eval/metrics/hallucination.py` — pure metric
  functions (WER/CER via `jiwer` with our own text normalization;
  non-empty-rate + suppression-reason breakdown).
- `eval/asr_eval.py` / `eval/hallucination_eval.py` — CLIs producing the two
  M2 accept-criteria reports under `reports/<run_id>/`.
- Tests: `test_frames.py`, `test_local_agreement.py` (+ Hypothesis property
  test), `test_hallucination_guard.py`, `test_vad.py` (synthetic
  near-silence only — no network needed, silero-vad ships its weights in the
  pip package), and `tests/integration/test_real_audio_components.py`
  (`@pytest.mark.network`: real Silero VAD + faster-whisper + LangIDTracker
  against a real downloaded FLEURS clip, skipped if the data isn't present).
- `docs/data_licenses.md` — every dataset/model used so far, with license
  and source.

### The two M2 accept-criteria reports (real runs, saved under reports/)

**ASR WER/CER, FLEURS subset, faster-whisper `small` int8 CPU** (`reports/asr-wer-1790119676/`):

| Language | Clips | WER | CER |
|---|---|---|---|
| en | 8 | 0.058 | 0.024 |
| es | 8 | 0.043 | 0.008 |
| hi | 8 | 0.418 | 0.161 |
| tl | 8 | 0.241 | 0.061 |

Only 8 clips/language (small subset, no threshold yet per spec). en/es are
strong; hi/tl are much weaker — this is a known limitation of Whisper
`small` on Hindi and (especially) Tagalog, not a bug. Larger models
(`large-v3-turbo`, GPU) should close most of this gap; worth re-running this
exact report once GPU access is available, to see the real gap size.

**Hallucination guard, silence + ESC-10 noise, same model** (`reports/hallucination-1790120120/`):

| Category | Clips | Non-empty rate (post-guard) | Suppressions by reason |
|---|---|---|---|
| noise | 20 | 0.0% | avg_logprob: 9, no_speech_prob: 3 |
| silence | 5 | 0.0% | no_speech_prob: 1 |

0% non-empty rate post-guard on both — comfortably under the 1% target in
SPEC.md Section 3. Small sample (25 clips total); worth re-running on a
larger noise set before trusting this number for anything beyond "the guard
isn't obviously broken."

### Test and lint status

- `uv run ruff check .`, `uv run ruff format --check .` — clean, all new
  modules included.
- `uv run mypy polyglot/core` (strict) — clean, unchanged (M2 code lives
  outside `core/`, so it isn't mypy-strict-checked, matching the M0 scoping
  decision to keep strict mode to `core/` only).
- `uv run pytest` (default, network tests deselected) — 48 passed.
- `uv run pytest -m network` (real downloaded data/models) — 4 passed:
  Silero VAD detects real speech, faster-whisper transcribes it, LangIDTracker
  detects English and a real code-switched en/es sentence.

### Bugs found and fixed while building this

- **`fasttext-wheel==0.9.2`'s single-string `predict()` is broken on
  numpy>=2.0** (`np.array(probs, copy=False)` — the `copy=False` contract
  changed in numpy 2.0). Worked around by always calling `predict([text],
  k=...)` (the list-input path returns the raw C++ binding output and never
  hits the broken line) and unwrapping the single result.
- **`ruff format` was rewriting Python code fences inside `.md` files**,
  which would have silently mutated `SPEC.md`/`CLAUDE.md`/`PROGRESS.md`.
  Already excluded via `extend-exclude = ["*.md"]` from the M1 fix; confirmed
  it still holds with the new files.
- Not a bug, but a real environment gap worth recording: `datasets`' `Audio`
  feature needs `torchcodec` to auto-decode, and `torchcodec` fails to load
  its native library on this Windows machine (`OSError: Could not load...
  libtorchcodec_image.dll`, almost certainly missing FFmpeg). Worked around
  by loading the `audio` column with `Audio(decode=False)` (raw bytes) and
  decoding with `soundfile` instead, which needs no FFmpeg. `torchcodec` was
  removed from dependencies since nothing else needs it.

### Decisions made / spec interpretations

- **Silero VAD requires 512-sample (32ms) windows at 16kHz** — confirmed by
  triggering the model's own `ValueError` on a smaller chunk. SPEC.md Section
  8.1 says "20ms or 32ms windows (whatever the installed version requires)";
  32ms it is.
- **Noise dataset: ESC-10 subset of ESC-50, not the full ESC-50 set.** The
  full ESC-50 dataset is CC BY-NC 3.0 (non-commercial); its ESC-10 subset is
  separately licensed CC BY 3.0 (attribution only, commercial-compatible),
  confirmed by reading ESC-50's LICENSE file. SPEC.md Section 10.2 delegates
  picking "an openly licensed noise dataset" to whoever builds this — chose
  ESC-10 for the clean, unambiguous license. Only 20 clips downloaded (2 per
  ESC-10 category) to keep the download light; can pull more via
  `scripts/download_datasets.py --only noise` if a larger sample is wanted.
- **Pipeline (M1) is untouched — still fakes-only.** M2 delivers standalone,
  independently-tested real components (VAD, ASR, hallucination guard,
  LangIDTracker) plus eval scripts, not a rewired conversational `Pipeline`.
  Wiring VAD-gated real ASR into `Pipeline` happens in M4 ("first full voice
  turn"), alongside the LLM, TTS, and LiveKit adapter — doing it piecemeal
  now would mean touching `Pipeline` twice for the same milestone's worth of
  change, and M2's own accept criteria (WER report, hallucination report)
  don't need a working `Pipeline` turn at all.
- **`make eval-smoke` was deliberately NOT wired to `eval/asr_eval.py`/
  `eval/hallucination_eval.py`**, even though it looks like a natural fit.
  SPEC.md Section 15 explicitly assigns "CI smoke eval on PRs" to M7, and
  that CI-facing version likely needs small *committed* fixtures for
  determinism/speed rather than network-downloaded FLEURS data — different
  enough from these dev-facing report scripts that wiring it now would
  probably need re-wiring again in M7. Left as the M0 placeholder.
- **M2's real-component tests only exercise English for the network-marked
  integration test** (`test_real_audio_components.py`), not all four
  languages — kept it to one clip to keep that test fast; the *actual*
  per-language quality signal is the WER report above, run separately via
  `eval/asr_eval.py --languages en es hi tl` (the default).
- **fastText LID model: `lid.176.ftz`** (compressed, ~950KB), not the full
  `lid.176.bin` (~126MB) — same predictions, much lighter for a CPU dev
  loop. License is CC BY-SA 3.0, recorded in `docs/data_licenses.md`.

### Unverified / deferred

- **No GPU tested.** Everything above ran on CPU (`small`, int8) per the dev
  profile described in SPEC.md Section 5. The `large-v3-turbo` GPU path is
  unverified — needs the GPU decision (Modal recommended, still pending)
  before it can be tried.
- **`config/languages.yaml`'s hallucination blocklist is a seed, not tuned.**
  0% non-empty rate on this small sample came mostly from the no_speech_prob/
  avg_logprob checks, not the blocklist — the blocklist phrases haven't
  actually been exercised against a hallucination yet. Revisit once a larger
  noise/silence set surfaces real blocklist-worthy phrases.
- **`FasterWhisperEngine`'s internal hallucination guard doesn't emit
  suppression events** — no `EventLog` access from inside `ASREngine`
  (Protocol doesn't include it). `eval/hallucination_eval.py` calls
  `WhisperModel.transcribe()` + `check_hallucination()` directly instead of
  going through `FasterWhisperEngine`, so this doesn't block the M2 report,
  but real per-suppression event logging (SPEC.md 8.3: "log every suppression
  as an event with the reason") is deferred to M4's pipeline wiring.
- Common Voice and EdAcc/L2-ARCTIC (SPEC.md Section 10.2) not downloaded —
  not needed for M2's accept criteria.

### Next milestone

M3: Knowledge base and retrieval — needs `config/sources.yaml` filled in by
the human first (CLAUDE.md rule 6: do not invent KB URLs).

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
