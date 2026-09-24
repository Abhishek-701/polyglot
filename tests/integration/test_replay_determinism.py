"""Determinism test. See SPEC.md Section 11.2.

Running the same scenario twice in simulated mode with the same (fake)
config must produce identical transcripts, intents, passages, and assistant
text. turn_id/session_id are excluded from the comparison since they are
opaque identifiers, not content.
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
from polyglot.transport.replay_adapter import ReplayAdapter, Scenario


def _fake_classifier(text: str) -> IntentResult:
    return IntentResult(label="refund", confidence=0.9)


def _make_scenario() -> Scenario:
    return Scenario.model_validate(
        {
            "id": "determinism-check",
            "lang": "en",
            "turns": [
                {"user_text": "My flight got cancelled, can I get a refund?"},
                {"user_text": "What about compensation for the delay?"},
            ],
        }
    )


def _build_pipeline(tmp_path: Path, run_name: str) -> Pipeline:
    scenario = _make_scenario()
    scripts = [
        [
            TranscriptPartial(
                text=turn.user_text,
                stable_prefix=turn.user_text,
                lang=scenario.lang,
                lang_prob=1.0,
                is_final=True,
                t_start_ms=0,
                t_end_ms=800,
            )
        ]
        for turn in scenario.turns
    ]
    return Pipeline(
        vad=FakeVAD(
            events=[
                VADEvent(kind="speech_start", t_ms=0, prob=0.9),
                VADEvent(kind="speech_end", t_ms=800, prob=0.9),
            ]
        ),
        asr=ScriptedFakeASREngine(scripts),
        turn_detector=FakeTurnDetector(prob=1.0),
        retriever=FakeRetriever(
            passages=[Passage(id="p1", doc_id="dot-refunds", text="...", source_url="x", score=0.9)]
        ),
        llm=FakeLLMClient(deltas=[LLMDelta(text="Here is your answer.")]),
        tts=FakeTTSEngine(),
        tracer=FakeTracer(),
        clock=SimulatedClock(),
        event_log=EventLog(tmp_path / f"{run_name}.jsonl"),
        session_id="determinism-check",
        intent_classifier=_fake_classifier,
    )


async def test_replay_is_deterministic(tmp_path: Path) -> None:
    scenario = _make_scenario()

    pipeline_a = _build_pipeline(tmp_path, "run-a")
    results_a = await ReplayAdapter(pipeline_a).run_scenario(scenario)

    pipeline_b = _build_pipeline(tmp_path, "run-b")
    results_b = await ReplayAdapter(pipeline_b).run_scenario(scenario)

    assert len(results_a) == len(results_b) == 2

    for result_a, result_b in zip(results_a, results_b, strict=True):
        record_a, record_b = result_a.record, result_b.record
        assert record_a.lang == record_b.lang
        assert record_a.user_text == record_b.user_text
        assert record_a.assistant_text_generated == record_b.assistant_text_generated
        assert record_a.assistant_text_spoken == record_b.assistant_text_spoken
        assert record_a.intent == record_b.intent
        assert record_a.passages == record_b.passages
        assert record_a.code_switched == record_b.code_switched
        assert record_a.interrupted == record_b.interrupted
