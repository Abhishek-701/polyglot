"""Turn orchestrator. See SPEC.md Section 4.3 (turn lifecycle) and Section 12
(span names).

M1 scope: wires VAD -> ASR -> TurnDetector -> Retriever -> LLM -> TTS with
injected components (Protocols from core/interfaces.py). Not yet included,
each deferred to the milestone that builds it for real: LangIDTracker
(placeholder: language comes straight from ASR), speculative retrieval (M5),
LangGraph dialogue policy and tools (M4), clause splitter (M4), history
compaction (M5), barge-in (M6). Nothing here imports a transport.
"""

from collections.abc import AsyncIterator

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
        retriever: Retriever,
        llm: LLMClient,
        tts: TTSEngine,
        tracer: Tracer,
        clock: Clock,
        event_log: EventLog,
        session_id: str,
        k_retrieval: int = 5,
        voice: str = "default",
        eou_threshold: float = 0.5,
    ) -> None:
        self.vad = vad
        self.asr = asr
        self.turn_detector = turn_detector
        self.retriever = retriever
        self.llm = llm
        self.tts = tts
        self.tracer = tracer
        self.clock = clock
        self.event_log = event_log
        self.session_id = session_id
        self.k_retrieval = k_retrieval
        self.voice = voice
        self.eou_threshold = eou_threshold
        self._turn_counter = 0

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
        timings: dict[str, int] = {}
        speech_end_logged = False

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
            nonlocal speech_end_logged
            async for frame in frames:
                for vad_event in self.vad.process(frame):
                    log("vad", vad_kind=vad_event.kind, prob=vad_event.prob)
                    if vad_event.kind == "speech_end" and not speech_end_logged:
                        speech_end_logged = True
                        timings["vad.speech_end"] = self.clock.now_ms() - turn_start_ms
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
                prob = self.turn_detector.end_of_turn_prob(partial.stable_prefix, lang, 0, history)
                if prob >= self.eou_threshold or partial.is_final:
                    committed = True
                    log("eou", prob=prob, rule="final" if partial.is_final else "threshold")
                    timings["eou.decision"] = self.clock.now_ms() - turn_start_ms
                    break

        if final_partial is None:
            raise RuntimeError(f"{turn_id}: ASR produced no transcript for this turn")

        final_text = final_partial.stable_prefix or final_partial.text

        log("retrieval_start", query=final_text)
        passages = await self.retriever.retrieve(final_text, lang, self.k_retrieval)
        log("retrieval_end", count=len(passages))
        timings["retrieval.final"] = self.clock.now_ms() - turn_start_ms

        messages = list(history)
        if passages:
            messages.append(Message(role="system", content="\n".join(p.text for p in passages)))
        messages.append(Message(role="user", content=final_text, lang=lang))

        assistant_text = ""
        first_token_logged = False
        async for delta in self.llm.stream(messages, None):
            if delta.text:
                assistant_text += delta.text
                if not first_token_logged:
                    first_token_logged = True
                    log("llm_first_token")
                    timings["llm.ttft"] = self.clock.now_ms() - turn_start_ms

        async def single_chunk() -> AsyncIterator[str]:
            yield assistant_text

        audio_frames: list[AudioFrame] = []
        async for frame in self.tts.synthesize(single_chunk(), lang, self.voice):
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
            intent=None,
            passages=passages,
            tool_calls=[],
            code_switched=False,
            interrupted=False,
            timings=timings,
        )
        return TurnResult(record=record, audio_frames=audio_frames)
