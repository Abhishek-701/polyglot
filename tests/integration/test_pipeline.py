"""Pipeline integration test: a normal turn end to end with fakes,
including the LangGraph dialogue policy (M4) — intent classification is
faked too, so this stays fast and network-free; the graph's own routing
logic is covered separately and thoroughly in test_policy_graph.py.

bargein is excluded from EXPECTED_EVENT_KINDS: that stage doesn't exist
until M6. tool_call isn't hit by this test's refund-intent path (it would
route to a different graph branch); see test_policy_graph.py for that.
"""

from pathlib import Path

from polyglot.core.clock import SimulatedClock
from polyglot.core.events import EventLog
from polyglot.core.pipeline import Pipeline
from polyglot.core.types import LLMDelta, Passage, TranscriptPartial, VADEvent
from polyglot.fakes import (
    FakeLLMClient,
    FakeRetriever,
    FakeTracer,
    FakeTTSEngine,
    FakeTurnDetector,
    FakeVAD,
    ScriptedFakeASREngine,
)
from polyglot.policy.intents import IntentResult
from polyglot.transport.replay_adapter import ReplayAdapter, Scenario, ScenarioTurn

EXPECTED_EVENT_KINDS = {
    "vad",
    "asr_partial",
    "asr_final",
    "eou",
    "retrieval_start",
    "retrieval_end",
    "llm_first_token",
    "tts_first_frame",
    "audio_out_first_frame",
}


def _fake_refund_classifier(text: str) -> IntentResult:
    return IntentResult(label="refund", confidence=0.9)


def _build_pipeline(tmp_path: Path) -> tuple[Pipeline, EventLog]:
    event_log = EventLog(tmp_path / "events.jsonl")
    interim_partial = TranscriptPartial(
        text="my flight was",
        stable_prefix="my flight was",
        lang="en",
        lang_prob=0.9,
        is_final=False,
        t_start_ms=0,
        t_end_ms=400,
    )
    final_partial = TranscriptPartial(
        text="my flight was cancelled",
        stable_prefix="my flight was cancelled",
        lang="en",
        lang_prob=1.0,
        is_final=True,
        t_start_ms=0,
        t_end_ms=800,
    )
    pipeline = Pipeline(
        vad=FakeVAD(
            events=[
                VADEvent(kind="speech_start", t_ms=0, prob=0.9),
                VADEvent(kind="speech_end", t_ms=800, prob=0.9),
            ]
        ),
        asr=ScriptedFakeASREngine([[interim_partial, final_partial]]),
        turn_detector=FakeTurnDetector(prob=0.0),
        retriever=FakeRetriever(
            passages=[Passage(id="p1", doc_id="dot-refunds", text="...", source_url="x", score=0.9)]
        ),
        llm=FakeLLMClient(deltas=[LLMDelta(text="Here is your answer.")]),
        tts=FakeTTSEngine(),
        tracer=FakeTracer(),
        clock=SimulatedClock(),
        event_log=event_log,
        session_id="s1",
        intent_classifier=_fake_refund_classifier,
    )
    return pipeline, event_log


async def test_normal_turn_produces_all_expected_events(tmp_path: Path) -> None:
    pipeline, event_log = _build_pipeline(tmp_path)
    scenario = Scenario(id="s1", lang="en", turns=[ScenarioTurn(user_text="irrelevant")])
    adapter = ReplayAdapter(pipeline)

    results = await adapter.run_scenario(scenario)

    assert len(results) == 1
    record = results[0].record
    assert record.intent == "refund"
    assert record.assistant_text_spoken == "Here is your answer."
    assert record.passages[0].doc_id == "dot-refunds"
    assert results[0].audio_frames

    kinds = {event.kind for event in event_log.read_all()}
    assert EXPECTED_EVENT_KINDS <= kinds


async def test_turn_timings_cover_key_spans(tmp_path: Path) -> None:
    pipeline, _ = _build_pipeline(tmp_path)
    scenario = Scenario(id="s1", lang="en", turns=[ScenarioTurn(user_text="irrelevant")])
    adapter = ReplayAdapter(pipeline)

    results = await adapter.run_scenario(scenario)

    timings = results[0].record.timings
    for span in ("asr.first_partial", "eou.decision", "retrieval.final", "llm.ttft", "turn"):
        assert span in timings
