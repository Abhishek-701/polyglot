# Polyglot: Technical Specification

Real-time cross-lingual voice agent with a deterministic evaluation harness.

Version 1.0. Owner: Abhishek Walvekar.

---

## 0. How to use this spec (read first, Claude Code)

- Work **one milestone at a time** (Section 15). Do not start a milestone until the previous one meets its acceptance criteria.
- At the end of each milestone: run lint and tests, update `PROGRESS.md`, summarize what was built, list anything unverified, and **stop for human review**.
- Library APIs in this space change quickly (LiveKit Agents, faster-whisper, vLLM, TTS models). **Verify every API against the installed version** (read the package source or docs) instead of relying on memory. Pin exact versions once verified.
- **Never fabricate metrics.** Every number in a report must come from a script run that is logged in `reports/`. If something cannot be measured in this environment (no GPU, no mic, gated dataset), say so and leave a TODO.
- Every latency optimization must sit behind a config flag (Section 9) so the naive baseline stays reproducible. The ablation table is the main deliverable of this project; it is impossible without flags.
- When a requirement here is ambiguous or seems wrong once you see real behavior, write the question into `PROGRESS.md` under "Open questions" and choose the simplest reasonable option.

---

## 1. Overview

### 1.1 What it is

A voice agent that a caller can phone or open in a browser, speak to in English, Spanish, Hindi, or a fourth lower-resource language, and get spoken answers grounded in an **English-only** knowledge base. Target perceived latency is under 800ms at p95.

### 1.2 Scenario

Airline irregular-operations support. A caller whose flight was cancelled or delayed asks about refunds, rebooking, compensation, and baggage. The knowledge base is public policy text: US DOT aviation consumer protection rules on refunds and delays, EU261 passenger rights, and three airline Contracts of Carriage. A mock booking and flight-status tool supports the conversational flow.

### 1.3 Goals

1. End-to-end streaming voice pipeline (VAD, ASR, dialogue policy, retrieval, LLM, TTS) with barge-in.
2. Measured latency reduction from a naive baseline to a tuned configuration, with a per-stage breakdown and a leave-one-out ablation.
3. Cross-lingual retrieval over an English corpus with three bridge strategies benchmarked.
4. A deterministic replay harness that runs recorded conversations through the exact production pipeline, used for regression testing in CI.
5. Per-language honesty: every quality metric reported per language, never only averaged.

### 1.4 Non-goals

- Speech-to-speech models. The cascaded pipeline is intentional for observability and component-level ablation.
- Real airline integrations or real bookings. Tools are mocked, with an optional real flight-status adapter.
- Supporting more than four languages.
- Fine-tuning ASR or LLM weights. (Fine-tuning a turn detector is a stretch goal only.)
- A polished consumer UI. A minimal web client plus a dashboard is enough.

---

## 2. Scope

| Item | Value |
|---|---|
| Languages | `en`, `es`, `hi`, plus one of `tl` / `vi` / `bn` (configurable, human decides in M0) |
| Input channels | Browser via WebRTC (required), phone via SIP (stretch, M8) |
| Knowledge base | ~5 to 10 public policy documents, English, a few thousand chunks |
| Concurrency target | 5 concurrent sessions on one GPU node (measured, not assumed) |
| Session length | Up to 10 minutes, up to 40 turns |

---

## 3. Non-functional requirements

| Requirement | Target | Notes |
|---|---|---|
| Perceived latency p50 | at most 500ms | Last user speech frame to first agent audio frame |
| Perceived latency p95 | at most 800ms (stretch 650ms) | Measured in realtime replay mode on the reference hardware |
| False endpoint rate | at most 8% per language | Agent starts speaking while user is mid-thought |
| ASR hallucination on silence | at most 1% of silent or noise-only segments produce text | |
| Retrieval recall@5 (cross-lingual) | at least 0.80 for `es`, `hi` | Report the fourth language honestly even if lower |
| Faithfulness (RAGAS) | at least 0.85 | |
| Dialogue policy overhead | at most 15ms p95 | Time spent in the LangGraph policy layer excluding model calls |
| Cost | Report cost per conversation minute | Self-hosted vs hosted comparison |

Reference hardware (record actual hardware in every latency report): one GPU in the L4 / A10G class for ASR, embeddings, and TTS; LLM on a second GPU via vLLM or on a hosted OpenAI-compatible endpoint.

---

## 4. Architecture

### 4.1 Key design decision: transport is separate from the pipeline

LiveKit handles WebRTC transport, rooms, and job dispatch only. The pipeline lives in `polyglot/core/` and does not import LiveKit. Both the live LiveKit adapter and the replay harness drive the **same** `Pipeline` object with injected components and an injected `Clock`.

Why: deterministic replay, component-level ablation, and swapping any stage without touching transport. The tradeoff is that we reimplement some things LiveKit's `AgentSession` provides for free (endpointing glue, interruption handling). That reimplementation is part of the project's value. Document this tradeoff in `docs/decisions/001-own-pipeline.md`.

### 4.2 Component diagram

```
                +---------------------- Transport adapters ----------------------+
                |  LiveKitAdapter (live)             ReplayAdapter (eval)        |
                +-------------------------------+--------------------------------+
                                                | AudioFrame stream in / out
                                                v
+------------------------------------------ core.Pipeline ------------------------------------------+
|                                                                                                     |
|  VAD --> ASR (streaming partials) --> LangIDTracker --> TurnDetector ----> DialoguePolicy (LangGraph)|
|                    |                                        ^                    |                   |
|                    +--> SpeculativeRetriever ---------------+                    v                   |
|                           (fires on stable partials,                     LLM (stream) + Tools        |
|                            caches by normalized prefix)                          |                   |
|                                                                                  v                   |
|                                                         ClauseSplitter --> TTSRouter --> audio out  |
|                                                                                  ^                   |
|  BargeInMonitor (VAD + ASR during agent speech) ---- cancel / flush / truncate --+                   |
|                                                                                                     |
|  Clock (real or simulated)   EventLog (JSONL)   Tracer (OpenTelemetry -> Langfuse)                   |
+-----------------------------------------------------------------------------------------------------+
        |                         |                          |
     Redis                  Postgres + pgvector           Langfuse
 (session state,          (corpus chunks, embeddings,   (traces per turn)
  semantic cache)          call records, eval results)
```

### 4.3 Turn lifecycle

1. Audio frames (20ms, 16kHz mono PCM) arrive from the adapter.
2. VAD marks speech onset and offset.
3. Streaming ASR emits `TranscriptPartial` events with a stable prefix (LocalAgreement policy).
4. `LangIDTracker` updates the session language and flags code-switching.
5. `SpeculativeRetriever` fires retrieval when the stable prefix passes the trigger rule (Section 8.6) and caches results.
6. `TurnDetector` produces an end-of-turn probability from transcript text, language, trailing silence, and recent history. When it crosses the per-language threshold, the turn commits.
7. `DialoguePolicy` (LangGraph) classifies intent and routes: answer from retrieval, call a tool, ask a clarifying question, or hand off.
8. LLM streams tokens. `ClauseSplitter` emits speakable chunks as soon as a clause boundary appears.
9. `TTSRouter` picks the TTS engine for the language and streams audio frames out.
10. `BargeInMonitor` watches for user speech during agent playback. On a confirmed interruption it cancels LLM and TTS, flushes the output buffer, and truncates the assistant message to what was actually played.
11. After the turn, history compaction runs off the critical path.

---

## 5. Tech stack

Verify each item's current version and API before use. Pin versions in `pyproject.toml`.

| Concern | Choice | Alternative to benchmark or fallback |
|---|---|---|
| Language / tooling | Python 3.11+, `uv`, `ruff`, `mypy` (strict on `core/`), `pytest`, `pytest-asyncio` | |
| Transport | LiveKit server (self-hosted via Docker) + LiveKit Python SDK / Agents worker for dispatch | Pipecat (not required) |
| VAD | Silero VAD | |
| Turn detection | Open-weight smart-turn style ONNX model behind `TurnDetector` (portable to replay) | LiveKit's multilingual turn detector, benchmarked as a comparison if it can run outside `AgentSession` |
| ASR | `faster-whisper` with `large-v3-turbo`, INT8 on GPU; `small` INT8 on CPU for dev and CI smoke | NVIDIA Parakeet TDT as English-only fast path (M5, flag `asr_router_parakeet`) |
| Language ID | Whisper's detected language per utterance + fastText LID on text windows for code-switch detection | |
| Zero-shot intent | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (verify) | LLM-based classification, compare latency |
| Embeddings | `BAAI/bge-m3` | `intfloat/multilingual-e5-large` |
| Reranker (optional) | `BAAI/bge-reranker-v2-m3` | none |
| Machine translation (bridge A) | `facebook/nllb-200-distilled-600M` | LLM translation |
| LLM | `Qwen2.5-7B-Instruct` (or current equivalent) served by vLLM with prefix caching, via OpenAI-compatible API | Hosted OpenAI-compatible fast inference endpoint, same interface |
| Dialogue policy | LangGraph (policy graph only; audio path is plain asyncio) | |
| TTS | CosyVoice2-0.5B where it supports the language; Kokoro-82M; route per language | Piper as a CPU fallback. **Check every TTS license and language list in M4.** Do not use XTTS-v2 (non-commercial license). |
| Vector store / DB | Postgres 16 + pgvector | |
| Cache / session state | Redis | |
| Observability | OpenTelemetry spans exported to self-hosted Langfuse | Grafana dashboard is a stretch |
| API | FastAPI (health, token issuing for the web client, admin endpoints) | |
| Web client | Minimal static page using LiveKit's JS client, plus a latency waterfall view | |
| Eval libs | `jiwer` (WER/CER), `ragas`, `unbabel-comet`, a UTMOS predictor (verify an available implementation), `matplotlib` for report charts | |
| GPU serving | Modal (or RunPod) for ASR/TTS/LLM services, scale to zero | Local GPU if available |
| CI | GitHub Actions: lint, unit tests, CPU smoke eval on every PR; full GPU eval on PR label `eval-full` or manual dispatch | |

HuggingFace task coverage (for the README): Automatic Speech Recognition, Voice Activity Detection, Translation, Text-to-Speech, Zero-Shot Classification, Sentence Similarity, Feature Extraction, Text Ranking, Text Generation.

---

## 6. Repository layout

```
polyglot/
  CLAUDE.md
  SPEC.md
  PROGRESS.md                  # maintained by Claude Code, one section per milestone
  README.md                    # results tables at the top (written in M8)
  pyproject.toml
  Makefile
  docker-compose.yml
  .env.example
  config/
    default.yaml
    languages.yaml             # per-language model routing and thresholds
    profiles/
      naive.yaml               # all optimizations off
      tuned.yaml               # all optimizations on
    thresholds.yaml            # CI eval gates
    sources.yaml               # knowledge base source URLs (human fills in)
  polyglot/
    core/
      types.py                 # data models (Section 7.1)
      interfaces.py            # Protocols (Section 7.2)
      clock.py                 # RealClock, SimulatedClock
      events.py                # EventLog, event types
      pipeline.py              # the orchestrator
      tracing.py               # OpenTelemetry setup, span helpers
    audio/
      vad.py
      frames.py                # resampling, framing, buffers
    asr/
      faster_whisper_engine.py
      local_agreement.py       # streaming stabilization
      parakeet_engine.py       # M5, optional
      router.py
      hallucination_guard.py
    langid/
      tracker.py
    turn/
      silence_detector.py      # naive baseline
      semantic_detector.py     # ONNX smart-turn style model
    retrieval/
      ingest.py
      chunking.py
      store.py                 # pgvector access
      bridges/
        mt_bridge.py
        multilingual.py
        hybrid.py              # reciprocal rank fusion
      speculative.py
      semantic_cache.py
    policy/
      graph.py                 # LangGraph dialogue policy
      intents.py
      prompts.py               # prefix-cache friendly prompt assembly
      compaction.py
    llm/
      client.py                # OpenAI-compatible streaming client
    tools/
      booking.py               # mock booking lookup
      flight_status.py         # mock + optional real adapter
    tts/
      router.py
      cosyvoice_engine.py
      kokoro_engine.py
      clause_splitter.py
    bargein/
      monitor.py
      playback_ledger.py       # tracks what text was actually played
    compliance/
      disclosure.py
    transport/
      livekit_adapter.py
      replay_adapter.py
    api/
      app.py
  eval/
    replay.py                  # CLI entry
    scenarios/                 # multi-turn conversation scripts (YAML)
    golden/
      golden.jsonl
      turns.jsonl              # labeled turn boundaries
      audio/                   # gitignored or Git LFS
    metrics/
      asr.py
      hallucination.py
      turn.py
      retrieval.py
      generation.py
      crosslingual.py
      tts.py
      latency.py
      cost.py
    ablation.py
    report.py
    gate.py                    # compares a run to thresholds.yaml, exits non-zero on failure
  scripts/
    download_datasets.py
    build_golden_audio.py
  web/
    index.html
    waterfall.html
  infra/
    modal_app.py
  docs/
    decisions/                 # ADRs
    architecture.md
  reports/                     # generated, one folder per run id
  tests/
    unit/
    integration/
```

---

## 7. Core contracts

Implement these first (M0). All other code depends on them. Use Pydantic v2 models for data and `typing.Protocol` for interfaces. Adjust signatures if real usage demands it, and record the change in `PROGRESS.md`.

### 7.1 Data models (`core/types.py`)

```python
class AudioFrame(BaseModel):
    pcm: bytes                  # 16-bit little-endian mono
    sample_rate: int = 16000
    t_ms: int                   # clock time of the frame start

class VADEvent(BaseModel):
    kind: Literal["speech_start", "speech_end"]
    t_ms: int
    prob: float

class TranscriptPartial(BaseModel):
    text: str
    stable_prefix: str          # committed text that will not change
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
    interrupted: bool = False   # set when barge-in truncated this message

class LLMDelta(BaseModel):
    text: str | None = None
    tool_call: dict | None = None
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
    tool_calls: list[dict]
    code_switched: bool
    interrupted: bool
    timings: dict[str, int]     # span name -> ms, see Section 12
```

### 7.2 Interfaces (`core/interfaces.py`)

```python
class Clock(Protocol):
    def now_ms(self) -> int: ...
    async def sleep_ms(self, ms: int) -> None: ...

class VAD(Protocol):
    def process(self, frame: AudioFrame) -> list[VADEvent]: ...

class ASREngine(Protocol):
    def stream(self, frames: AsyncIterator[AudioFrame], lang_hint: str | None) -> AsyncIterator[TranscriptPartial]: ...

class TurnDetector(Protocol):
    def end_of_turn_prob(self, text: str, lang: str, trailing_silence_ms: int, history: list[Message]) -> float: ...

class Retriever(Protocol):
    async def retrieve(self, query: str, lang: str, k: int) -> list[Passage]: ...

class LLMClient(Protocol):
    def stream(self, messages: list[Message], tools: list[dict] | None) -> AsyncIterator[LLMDelta]: ...

class TTSEngine(Protocol):
    languages: set[str]
    def synthesize(self, chunks: AsyncIterator[str], lang: str, voice: str) -> AsyncIterator[AudioFrame]: ...

class Tracer(Protocol):
    def span(self, name: str, **attrs) -> ContextManager: ...
```

Every component takes a `Clock` if it measures or waits on time. No component calls `time.time()` or `asyncio.sleep()` directly. A unit test greps for this.

### 7.3 Event log

Every pipeline step appends a typed event (`vad`, `asr_partial`, `asr_final`, `eou`, `retrieval_start`, `retrieval_end`, `llm_first_token`, `tts_first_frame`, `audio_out_first_frame`, `bargein`, `tool_call`, etc.) with `t_ms` from the injected clock. The event log is written as JSONL per session. All metrics are computed from event logs, never from ad hoc timers.

---

## 8. Component specifications

### 8.1 VAD
- Silero VAD at 16kHz with 20ms or 32ms windows (whatever the installed version requires; resample and reframe in `audio/frames.py`).
- Config: `threshold`, `min_speech_ms`, `min_silence_ms`, per language in `languages.yaml`.
- ASR only receives audio inside VAD speech regions plus a configurable padding (default 200ms before, 300ms after).

### 8.2 Streaming ASR
- `faster-whisper` with a sliding window over buffered speech audio, re-decoded every `asr_step_ms` (default 200ms).
- `local_agreement.py`: commit the longest common prefix of the last two hypotheses (LocalAgreement-2). Emit `TranscriptPartial` with `stable_prefix`.
- Language handling modes (config `asr.lang_mode`): `auto` (detect per utterance), `session` (detect on first utterance, then force), `forced` (from config). Default `session`.
- On turn commit, run one final decode on the full utterance for `is_final=True`.

### 8.3 Hallucination guard
Drop or blank an ASR result when any of these hold (thresholds configurable):
- Segment lies outside VAD speech.
- `no_speech_prob > 0.6`.
- `avg_logprob < -1.0`.
- Compression ratio of the text above 2.4 (repetition).
- Text matches a per-language blocklist in `config/languages.yaml` (seed with known Whisper silence phrases such as subtitle credits and "thank you for watching" equivalents).
Log every suppression as an event with the reason. The hallucination metric feeds pure silence, room noise, and music clips through the guard.

### 8.4 Language ID and code-switching
- Session language comes from ASR detection (mode above).
- `langid/tracker.py` runs fastText LID over sliding windows of 4 to 6 words on the final transcript and marks the turn `code_switched=True` when more than one language exceeds a share threshold.
- Response language policy: respond in the session's primary non-English language when the user code-switches, unless the user explicitly asks for English. Record the decision in the event log.
- This is expected to be imperfect. The goal is measurement: WER and task success split by code-switched vs monolingual turns.

### 8.5 Turn detection
- `silence_detector.py` (naive): commit the turn after `min_silence_ms` (default 500ms) of VAD silence.
- `semantic_detector.py` (tuned): after at least `min_silence_ms_semantic` (default 120ms) of silence, score the stable transcript with the ONNX turn model. Commit if probability exceeds the per-language threshold. Hard cap: commit after `max_silence_ms` (default 1200ms) regardless.
- Thresholds live in `languages.yaml` and are tuned in M5 against `eval/golden/turns.jsonl`.
- Emit `eou` event with the probability, silence duration, and which rule fired.

### 8.6 Speculative retrieval
- Trigger when the stable prefix has at least `spec_min_words` (default 4) words and has changed meaningfully since the last speculative query (cosine distance of embeddings above `spec_delta`, default 0.08).
- Run the configured bridge retrieval asynchronously. Store results keyed by the query embedding.
- At turn commit: if the final transcript embedding has cosine similarity at least `spec_reuse_sim` (default 0.92) with the latest speculative query, reuse its passages; otherwise retrieve again.
- Metrics: speculative hit rate, wasted retrieval calls per turn, latency saved.
- At most `spec_max_inflight` (default 2) concurrent speculative calls per session; cancel stale ones.

### 8.7 Cross-lingual retrieval bridges
All three implement `Retriever`. Selected by flag `retrieval_bridge`.
- `mt`: translate query to English with NLLB, embed with BGE-M3, dense search.
- `multilingual`: embed the original-language query with BGE-M3 directly, dense search.
- `hybrid`: run both, fuse with reciprocal rank fusion (k=60), optional rerank with `bge-reranker-v2-m3`.
- Chunking: 300 to 500 tokens with 50 token overlap, preserving section headings as metadata. Store `doc_id`, `section`, `source_url`, `chunk_index`.
- pgvector: HNSW index on the dense vector. Document why pgvector suffices at this scale in an ADR.

### 8.8 Semantic cache
- Redis cache keyed by (language, normalized question embedding bucket). Hit when cosine similarity above 0.95 and the cached answer came from retrieval without tool calls.
- Never cache answers that depended on tool output or session-specific data.
- Flag `semantic_cache`.

### 8.9 Dialogue policy (LangGraph)
Nodes:
- `classify_intent`: zero-shot multilingual classifier with labels `refund`, `rebooking`, `compensation`, `baggage`, `flight_status`, `booking_lookup`, `smalltalk`, `other`. Below a confidence floor, route to `clarify`.
- `retrieve`: uses cached speculative passages if valid.
- `tool_call`: `get_flight_status(flight_number, date)`, `lookup_booking(confirmation_code, last_name)`.
- `clarify`: ask one short question in the user's language.
- `handoff`: polite escalation message; logs a handoff event.
- `generate`: builds the prompt and streams from the LLM.
- `guard`: inline check that the response does not state refund amounts or deadlines absent from retrieved passages; if it fails, fall back to a safe hedge.
Measure policy overhead (time in graph minus model and tool time). Target at most 15ms p95.

When a tool call is expected to exceed `filler_threshold_ms` (default 400ms), speak a short filler phrase in the user's language ("let me check that") while the tool runs. Flag `filler_on_tool_call`.

### 8.10 Prompt assembly and history
- Prompt order for prefix caching (flag `prefix_cache_prompt_order`): static system prompt and policy instructions, then compacted conversation summary, then recent raw turns, then retrieved passages, then the current user turn. With the flag off, use a naive order (retrieved passages first, then system prompt, then full history).
- System prompt rules: answer only from passages, cite the source document name when stating a rule, say plainly when the policy text does not cover the question, keep spoken responses under 3 sentences unless asked for detail, respond in the session language, no markdown or lists in output.
- Compaction (flag `history_compaction`): keep the last 6 turns raw; after each turn, asynchronously summarize older turns into a running summary. Never on the critical path.

### 8.11 LLM client
- OpenAI-compatible streaming client, base URL from config, works with vLLM and hosted endpoints.
- Temperature 0 in eval runs. Record model name, server, and version in every report.
- Timeouts and one retry for the first token only; if TTFT exceeds `llm_ttft_timeout_ms`, fall back to the secondary endpoint if configured.

### 8.12 Clause splitter and TTS
- `clause_splitter.py` (flag `tts_chunking`: `full` | `sentence` | `clause`): in `clause` mode, emit a chunk at clause punctuation for the language (including the Devanagari danda for Hindi) once the chunk has at least `min_chunk_words` (default 4) words; force an emit at `max_chunk_words` (default 18).
- `tts/router.py`: pick the engine by language from `languages.yaml`. In M4, produce `docs/tts_language_matrix.md` listing each candidate engine, its supported languages, license, and measured TTFB on the reference hardware. Choose per language from that matrix.
- Output resampled to the transport's sample rate.

### 8.13 Barge-in
- While agent audio is playing, VAD and ASR keep running on user input.
- Interruption is confirmed when user speech lasts at least `bargein_min_speech_ms` (default 250ms) **and** the partial transcript is not a backchannel (per-language list in `languages.yaml`: "mhm", "ok", "sí", "haan", etc.). Echo of the agent's own audio must not trigger barge-in; for browser clients rely on WebRTC echo cancellation and add a guard that ignores partials highly similar to the text currently being spoken.
- On confirmation: cancel LLM stream, cancel TTS, flush queued audio, and emit `bargein` event.
- `playback_ledger.py` records which text chunks were fully played and the play position within the current chunk. The assistant `Message` stored in history is truncated to the spoken text and marked `interrupted=True`. This is the most error-prone part of the system and must have property-based tests.

### 8.14 Compliance
- The greeting discloses that the caller is speaking with an AI assistant, in the caller's language once known (greet in English plus a short multilingual prompt at start).
- Log a `disclosure` event per session. Log recording consent if audio is stored.
- Do not store raw caller audio outside eval runs unless `store_audio: true`.

### 8.15 Tools
- `lookup_booking` and `get_flight_status` return fixture data from `tests/fixtures/bookings.json` and `flights.json`.
- Optional real adapter for flight status behind `tools.flight_status.provider: mock | opensky`. Mock is default and is always used in eval for determinism.

---

## 9. Configuration

Pydantic-settings loading YAML with environment overrides. Profiles overlay `default.yaml`.

Optimization flags (all in the profile files):

| Flag | naive | tuned |
|---|---|---|
| `turn_detection` | `silence` | `semantic` |
| `speculative_retrieval` | false | true |
| `prefix_cache_prompt_order` | false | true |
| `history_compaction` | false | true |
| `tts_chunking` | `full` | `clause` |
| `semantic_cache` | false | true |
| `filler_on_tool_call` | false | true |
| `asr_router_parakeet` | false | true (if implemented) |
| `retrieval_bridge` | `mt` | winner from M3 |

`eval/ablation.py` runs: naive, tuned, and tuned with each flag reverted one at a time (leave-one-out). Output: a table of p50/p95 perceived latency and quality metrics per configuration.

---

## 10. Data

### 10.1 Knowledge base
- `config/sources.yaml` lists source URLs and document names. **The human fills this in**; do not invent URLs. Expected: DOT refund and delay rules, EU261 text or official summary, three airline Contracts of Carriage.
- `retrieval/ingest.py` downloads, extracts text (HTML and PDF), chunks, embeds, and upserts. Idempotent: re-running with unchanged sources makes no changes. Store a content hash per document.

### 10.2 Public speech datasets (downloaded by script, never committed)
- FLEURS for multilingual read speech in all four languages.
- Common Voice for speaker variety (requires accepting terms on HuggingFace; ask the human to log in).
- EdAcc or L2-ARCTIC for accented English.
- Silence and noise clips for the hallucination test (generate silence; source noise from an openly licensed noise dataset and record its license in `docs/data_licenses.md`).

### 10.3 Golden set (`eval/golden/golden.jsonl`)
One line per single-turn item:

```json
{
  "id": "hi-refund-007",
  "lang": "hi",
  "audio_path": "eval/golden/audio/hi-refund-007.wav",
  "reference_transcript": "...",
  "code_switched": true,
  "speaker": {"native_lang": "hi", "recorded_by": "human|tts|dataset"},
  "question_type": "refund",
  "expected_doc_ids": ["dot-refunds"],
  "expected_passage_ids": ["dot-refunds#12"],
  "reference_answer": "...",
  "answerable": true
}
```

Targets: 200 items total, at least 40 per language, at least 20% code-switched for `hi` and `es`, at least 15% unanswerable from the corpus (tests the "policy does not cover this" behavior).

Claude Code builds the schema, validators, and a script that drafts candidate questions and reference answers from the corpus for **human review**. Audio for the golden set comes from human recordings where possible. TTS-generated audio is allowed only as a clearly labeled fallback (`recorded_by: "tts"`) and must be reported separately, since it makes ASR look better than reality.

### 10.4 Turn boundary labels (`eval/golden/turns.jsonl`)
100+ clips with labeled pauses: each pause marked `end_of_turn` or `mid_turn`. Used to tune and evaluate turn detection. Build a small labeling helper script; the human labels.

### 10.5 Scenarios (`eval/scenarios/*.yaml`)
Multi-turn scripted conversations for replay:

```yaml
id: es-cancelled-flight-bargein
lang: es
turns:
  - user_audio: eval/golden/audio/es-scn1-t1.wav
    start: after_agent_done      # or {after_agent_start_ms: 1200} to simulate barge-in
  - user_audio: eval/golden/audio/es-scn1-t2.wav
    start: {after_agent_start_ms: 900}
expect:
  intents: [refund, compensation]
  bargein_count: 1
  final_history_contains_interrupted: true
```

At least 20 scenarios, including at least 6 with barge-in, 3 with tool calls, and 3 with code-switching.

---

## 11. Evaluation harness

### 11.1 Replay adapter
- Reads a scenario, streams user audio in 20ms frames according to the scenario timing, and collects agent output frames.
- Two modes:
  - `simulated`: `SimulatedClock` advances only when components yield; fast and deterministic, used for quality metrics and CI. Model compute time is not reflected in simulated time, so **never report latency from simulated mode**.
  - `realtime`: `RealClock`, audio paced at real speed; used for all latency numbers, on reference hardware.
- Output per session: event log JSONL, `TurnRecord` list, agent audio WAV.

### 11.2 Determinism test
Running the same scenario twice in simulated mode with the same config must produce identical transcripts, intents, passages, and assistant text. Small numeric differences from GPU nondeterminism are allowed only in scores, within a tolerance set in the test.

### 11.3 Metrics

| Module | Metrics |
|---|---|
| `asr.py` | WER and CER per language, split by code-switched vs not, by speaker nativeness, and by `recorded_by` |
| `hallucination.py` | Non-empty output rate on silence, noise, and music; suppressions by reason |
| `turn.py` | False endpoint rate, missed endpoint rate, EOU delay p50/p95, per language |
| `retrieval.py` | recall@1/5/10, MRR, per language, per bridge |
| `generation.py` | RAGAS faithfulness and answer relevance; unanswerable-handling accuracy; guard trigger rate |
| `crosslingual.py` | COMET and chrF of responses against reference answers in the target language |
| `tts.py` | UTMOS; round-trip intelligibility (ASR on TTS output, WER against intended text); TTFB |
| `latency.py` | Per-stage and perceived latency p50/p95/p99 from event logs; waterfall chart |
| `cost.py` | GPU seconds and tokens per conversation minute, converted with prices from config |
| scenario checks | Intent sequence, barge-in count, history truncation correctness, task success judged by an LLM with a 50-item human-agreement check reported alongside |

### 11.4 Reports
`eval/report.py` writes `reports/<run_id>/` containing `summary.md` (all tables), `metrics.json`, charts as PNG, the resolved config, hardware info, model versions, and git commit hash.

### 11.5 Gates
`eval/gate.py` compares `metrics.json` to `config/thresholds.yaml` and to the last baseline on `main`. Fail when: WER regresses more than 0.5 points in any language, faithfulness drops more than 0.02, recall@5 drops more than 0.03, hallucination rate rises above 1%, or (full eval only) p95 perceived latency rises more than 100ms.

---

## 12. Observability

Span names (also used as keys in `TurnRecord.timings`):

`turn`, `vad.speech_end`, `asr.first_partial`, `asr.final`, `eou.decision`, `retrieval.speculative`, `retrieval.final`, `policy.graph`, `llm.ttft`, `llm.total`, `tool.<name>`, `tts.ttfb`, `audio.first_frame_out`, `bargein`, `compaction`.

Perceived latency is `audio.first_frame_out.t_ms - vad.speech_end.t_ms` for the committed turn.

- Export spans to Langfuse via OpenTelemetry. One trace per turn, grouped by session.
- Integration test: every completed turn has all required spans.
- `web/waterfall.html`: reads a session's event log (from the API) and draws a per-turn latency waterfall. This is the key demo visual.

---

## 13. Infrastructure

`docker-compose.yml` services: `postgres` (pgvector image), `redis`, `langfuse` (and its database), `livekit` (server in dev mode), `api`, `agent`. GPU services (`vllm`, `asr`, `tts`) run under a compose profile `gpu` locally or on Modal via `infra/modal_app.py`.

Makefile targets:

```
make setup        # uv sync, pre-commit install
make lint         # ruff + mypy
make test         # unit + integration (CPU)
make up / down    # docker compose
make ingest       # build the knowledge base
make agent        # run the live agent worker
make replay SCENARIO=... PROFILE=... MODE=simulated|realtime
make eval-smoke   # CPU, small models, subset; used in CI
make eval-full    # GPU, full golden set and scenarios
make ablation     # leave-one-out across flags (realtime, GPU)
make report RUN=...
```

CPU dev mode: every component has a CPU path (Whisper `small`, a small instruct model or hosted endpoint, a CPU TTS) so the whole pipeline runs without a GPU for development and CI smoke tests.

---

## 14. Testing strategy

- Unit tests with fakes for every interface (`FakeASR` replays scripted partials, `FakeLLM` streams scripted tokens, `FakeTTS` emits silence frames of computed duration).
- Property-based tests (Hypothesis) for `local_agreement.py`, `clause_splitter.py`, and `playback_ledger.py` truncation.
- Pipeline integration tests in simulated mode with fakes, covering: normal turn, barge-in mid-chunk, barge-in during filler, tool call, code-switched turn, speculative hit and miss.
- A test that fails if any module under `polyglot/` calls `time.time`, `time.monotonic`, or `asyncio.sleep` directly (outside `clock.py`).
- No test depends on network access except those marked `@pytest.mark.network`, skipped in CI.

---

## 15. Milestones

Each milestone ends with: lint and tests green, `PROGRESS.md` updated, and a stop for human review.

### M0: Scaffold and contracts
- Repo layout, `pyproject.toml`, Makefile, pre-commit, CI workflow (lint + unit tests).
- `core/types.py`, `core/interfaces.py`, `core/clock.py`, `core/events.py`, config loading with profiles.
- Fakes for every interface.
- **Accept:** `make lint` and `make test` pass; config loads both profiles; clock-usage test exists and passes.
- **Human decides:** fourth language, GPU provider, hosted LLM fallback.

### M1: Pipeline skeleton and replay harness
- `core/pipeline.py` wiring all stages with fakes; `replay_adapter.py`; `eval/replay.py` CLI; event log output; basic `latency.py` from event logs.
- **Accept:** a scripted scenario runs end to end with fakes; determinism test passes; event log contains all required events.

### M2: Real audio front end
- Silero VAD, faster-whisper streaming with LocalAgreement, hallucination guard, language tracker.
- Dataset download script; ASR and hallucination metrics.
- **Accept:** per-language WER report on a FLEURS subset (reported, no threshold yet); hallucination report on silence and noise clips.

### M3: Knowledge base and retrieval
- Ingestion pipeline (needs `sources.yaml` from human), pgvector store, three bridges, retrieval metrics.
- Golden question drafting script; human reviews and finalizes at least 100 items.
- **Accept:** recall@k table per language per bridge; ADR recording the chosen default bridge and why.

### M4: First full voice turn (naive profile)
- LLM client, LangGraph policy with intents and mock tools, prompt assembly, guard, TTS router and engines, compliance greeting.
- `docs/tts_language_matrix.md`.
- LiveKit adapter and minimal web client.
- **Accept:** the human can hold a spoken conversation in English and Spanish in the browser; replay runs real components in realtime mode; **naive baseline latency report recorded and committed** before any optimization.
- **Human checkpoint:** talk to it in every language, note problems in `PROGRESS.md`.

### M5: Latency work
- Semantic turn detector with per-language threshold tuning; speculative retrieval; prefix-cache prompt order and vLLM prefix caching; clause chunking; history compaction; semantic cache; filler on tool calls; optional Parakeet router.
- **Accept:** tuned profile report; leave-one-out ablation table; p95 perceived latency at or near target on reference hardware (report honestly if not met, with the bottleneck identified from traces).

### M6: Barge-in
- Barge-in monitor, backchannel filter, echo guard, playback ledger and history truncation.
- **Accept:** all barge-in scenarios pass their expectations; property tests pass; false barge-in rate reported on scenarios with backchannels.

### M7: Full evaluation and CI gates
- Remaining metrics (turn, generation, cross-lingual, TTS, cost, scenario checks), report generator, gate script, thresholds, CI smoke eval on PRs, `eval-full` workflow on label.
- **Accept:** one complete report for the tuned profile; a deliberately regressed branch fails the gate.

### M8: Observability, packaging, deploy
- Langfuse traces verified; waterfall page; deploy to Modal (or chosen provider); README with results tables at the top, architecture diagram, HF task list, how to reproduce every number; ADRs complete.
- Stretch: SIP phone number via LiveKit SIP; Grafana dashboard; fine-tuned turn detector for one language.
- **Accept:** a fresh clone can run `make setup && make up && make eval-smoke` successfully following the README.

### Cut order if behind schedule
1. Fourth language
2. SIP
3. MT bridge (keep multilingual embeddings only)
4. TTS quality metrics (keep TTFB)
5. Parakeet router

Irreducible core: two languages, streaming end to end, semantic turn detection, barge-in with correct truncation, replay harness, naive vs tuned latency ablation.

---

## 16. Open questions for the human (answer before or during M0)

1. Fourth language: Tagalog, Vietnamese, or Bengali?
2. GPU provider and budget ceiling for the month.
3. Hosted LLM fallback: which provider, if any?
4. Knowledge base URLs for `config/sources.yaml`.
5. Who records golden audio, and for which languages?
6. Public deployment or private demo only?
