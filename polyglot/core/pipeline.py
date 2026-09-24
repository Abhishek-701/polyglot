"""Turn orchestrator. See SPEC.md Section 4.3 (turn lifecycle) and Section 12
(span names).

M4: VAD -> ASR -> TurnDetector -> DialoguePolicy (LangGraph: classify_intent
-> retrieve/tool_call/clarify/handoff -> generate -> guard) -> ClauseSplitter
-> TTS. The dialogue policy graph is built once per Pipeline (not per turn)
and treated as an opaque `ainvoke(dict) -> dict` callable here — LangGraph
itself is only imported by polyglot/policy/graph.py, matching CLAUDE.md's
"LangGraph is used only for the dialogue policy" rule. Retrieval and LLM
generation now happen inside that graph rather than inline here; the graph's
`on_event` hook is how this module still gets `retrieval_start`/
`retrieval_end`/`llm_first_token`/`tool_call` events and their timings
(SPEC.md Section 7.3/12), since a per-turn callback can't be captured at
graph-build time — `_active_turn_id`/`_active_turn_start_ms`/`_active_timings`
carry the current turn's identity into that one shared callback instead.

Not yet included, each deferred to the milestone that builds it for real:
LangIDTracker (M2, not wired in — language still comes straight from ASR),
speculative retrieval and history compaction (M5), barge-in (M6).
"""

from collections.abc import AsyncIterator, Callable
from typing import Any

from pydantic import BaseModel

from polyglot.core.events import Event, EventKind, EventLog
from polyglot.core.interfaces import (
    VAD,
    ASREngine,
    Clock,
    LLMClient,
    Retriever,
    Tracer,
    TTSEngine,
    TurnDetector,
)
from polyglot.core.types import AudioFrame, Message, TurnRecord
from polyglot.policy.graph import DEFAULT_CONFIDENCE_FLOOR, build_policy_graph
from polyglot.policy.intents import IntentResult
from polyglot.policy.intents import classify_intent as default_classify_intent
from polyglot.tts.clause_splitter import TtsChunkingMode, split_stream

# Event kinds fired by policy/graph.py's on_event hook, and the
# TurnRecord.timings span each corresponds to (SPEC.md Section 12) — None
# where the event doesn't correspond to a single span (e.g. retrieval_start
# is the start of the retrieval.final span, not a span of its own).
_POLICY_EVENT_SPANS: dict[str, str | None] = {
    "retrieval_start": None,
    "retrieval_end": "retrieval.final",
    "llm_first_token": "llm.ttft",
    "tool_call": None,
}


class TurnResult(BaseModel):
    record: TurnRecord
    audio_frames: list[AudioFrame]


class Pipeline:
    def __init__(
        self,
        *,
        vad: VAD,
        asr: ASREngine,
        turn_detector: TurnDetector,
        llm: LLMClient,
        retriever: Retriever,
        tts: TTSEngine,
        tracer: Tracer,
        clock: Clock,
        event_log: EventLog,
        session_id: str,
        k_retrieval: int = 5,
        voice: str = "",
        eou_threshold: float = 0.5,
        tts_chunking: TtsChunkingMode = "full",
        confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR,
        prefix_cache_prompt_order: bool = False,
        intent_classifier: Callable[[str], IntentResult] = default_classify_intent,
    ) -> None:
        self.vad = vad
        self.asr = asr
        self.turn_detector = turn_detector
        self.tts = tts
        self.tracer = tracer
        self.clock = clock
        self.event_log = event_log
        self.session_id = session_id
        self.voice = voice
        self.eou_threshold = eou_threshold
        self.tts_chunking = tts_chunking
        self._turn_counter = 0

        self._active_turn_id: str | None = None
        self._active_turn_start_ms = 0
        self._active_timings: dict[str, int] = {}

        self._policy_graph = build_policy_graph(
            llm,
            retriever,
            k=k_retrieval,
            confidence_floor=confidence_floor,
            prefix_cache_prompt_order=prefix_cache_prompt_order,
            intent_classifier=intent_classifier,
            on_event=self._emit_policy_event,
        )

    def _emit_policy_event(self, kind: str) -> None:
        assert self._active_turn_id is not None, "policy event fired outside a turn"
        self.event_log.append(
            Event(
                kind=kind,  # type: ignore[arg-type]
                t_ms=self.clock.now_ms(),
                session_id=self.session_id,
                turn_id=self._active_turn_id,
                data={},
            )
        )
        span = _POLICY_EVENT_SPANS.get(kind)
        if span is not None:
            self._active_timings[span] = self.clock.now_ms() - self._active_turn_start_ms

    async def run_turn(
        self,
        frames: AsyncIterator[AudioFrame],
        lang_hint: str | None,
        history: list[Message],
    ) -> TurnResult:
        turn_id = f"{self.session_id}-turn-{self._turn_counter}"
        self._turn_counter += 1

        with self.tracer.span("turn", turn_id=turn_id):
            return await self._run_turn(turn_id, frames, lang_hint, history)

    async def _run_turn(
        self,
        turn_id: str,
        frames: AsyncIterator[AudioFrame],
        lang_hint: str | None,
        history: list[Message],
    ) -> TurnResult:
        turn_start_ms = self.clock.now_ms()
        self._active_turn_id = turn_id
        self._active_turn_start_ms = turn_start_ms
        self._active_timings = {}
        timings = self._active_timings

        speech_end_logged = False
        speech_end_t_ms: int | None = None

        def log(kind: EventKind, **data: object) -> None:
            self.event_log.append(
                Event(
                    kind=kind,
                    t_ms=self.clock.now_ms(),
                    session_id=self.session_id,
                    turn_id=turn_id,
                    data=data,
                )
            )

        async def tapped_frames() -> AsyncIterator[AudioFrame]:
            nonlocal speech_end_logged, speech_end_t_ms
            async for frame in frames:
                for vad_event in self.vad.process(frame):
                    log("vad", vad_kind=vad_event.kind, prob=vad_event.prob)
                    if vad_event.kind == "speech_end":
                        speech_end_t_ms = vad_event.t_ms
                        if not speech_end_logged:
                            speech_end_logged = True
                            timings["vad.speech_end"] = self.clock.now_ms() - turn_start_ms
                    elif vad_event.kind == "speech_start":
                        speech_end_t_ms = None
                yield frame

        lang = lang_hint or "en"
        final_partial = None
        committed = False

        async for partial in self.asr.stream(tapped_frames(), lang_hint):
            asr_event_kind: EventKind = "asr_final" if partial.is_final else "asr_partial"
            log(
                asr_event_kind,
                text=partial.text,
                stable_prefix=partial.stable_prefix,
                lang=partial.lang,
            )
            if "asr.first_partial" not in timings:
                timings["asr.first_partial"] = self.clock.now_ms() - turn_start_ms
            lang = partial.lang or lang
            final_partial = partial
            if partial.is_final:
                timings["asr.final"] = self.clock.now_ms() - turn_start_ms

            if not committed:
                trailing_silence_ms = (
                    self.clock.now_ms() - speech_end_t_ms if speech_end_t_ms is not None else 0
                )
                prob = self.turn_detector.end_of_turn_prob(
                    partial.stable_prefix, lang, trailing_silence_ms, history
                )
                if prob >= self.eou_threshold or partial.is_final:
                    committed = True
                    log("eou", prob=prob, rule="final" if partial.is_final else "threshold")
                    timings["eou.decision"] = self.clock.now_ms() - turn_start_ms
                    break

        if final_partial is None:
            raise RuntimeError(f"{turn_id}: ASR produced no transcript for this turn")

        final_text = final_partial.stable_prefix or final_partial.text

        with self.tracer.span("policy.graph", turn_id=turn_id):
            policy_result: dict[str, Any] = await self._policy_graph.ainvoke(
                {"user_text": final_text, "lang": lang, "history": history}
            )

        assistant_text = policy_result.get("assistant_text", "")
        passages = policy_result.get("passages", [])
        intent = policy_result.get("intent")
        tool_result = policy_result.get("tool_result")
        interrupted = False

        async def text_stream() -> AsyncIterator[str]:
            yield assistant_text

        chunks = split_stream(text_stream(), mode=self.tts_chunking, lang=lang)

        audio_frames: list[AudioFrame] = []
        async for frame in self.tts.synthesize(chunks, lang, self.voice):
            if not audio_frames:
                log("tts_first_frame")
                log("audio_out_first_frame")
                timings["tts.ttfb"] = self.clock.now_ms() - turn_start_ms
                timings["audio.first_frame_out"] = self.clock.now_ms() - turn_start_ms
            audio_frames.append(frame)

        timings["turn"] = self.clock.now_ms() - turn_start_ms

        record = TurnRecord(
            turn_id=turn_id,
            session_id=self.session_id,
            lang=lang,
            user_text=final_text,
            assistant_text_generated=assistant_text,
            assistant_text_spoken=assistant_text,
            intent=intent,
            passages=passages,
            tool_calls=[tool_result] if tool_result is not None else [],
            code_switched=False,
            interrupted=interrupted,
            timings=timings,
        )
        self._active_turn_id = None
        return TurnResult(record=record, audio_frames=audio_frames)
