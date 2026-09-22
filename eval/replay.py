"""CLI entry point: make replay SCENARIO=<path> PROFILE=naive|tuned MODE=simulated|realtime.

See SPEC.md Section 11.1. M1 scope: the pipeline is wired entirely with fakes
(polyglot/fakes.py) since no real ASR/LLM/TTS exist yet. --profile is loaded
and validated but does not yet change fake behavior — the naive/tuned flags
only start affecting real component choices from M5 onward. Output: an event
log JSONL and a turn records JSONL, written under reports/<run_id>/.
"""

import argparse
import asyncio
import time
from pathlib import Path

import yaml

from polyglot.core.clock import RealClock, SimulatedClock
from polyglot.core.config import load_settings
from polyglot.core.events import EventLog
from polyglot.core.interfaces import Clock
from polyglot.core.pipeline import Pipeline
from polyglot.core.types import LLMDelta, TranscriptPartial, VADEvent
from polyglot.fakes import (
    FakeLLMClient,
    FakeRetriever,
    FakeTracer,
    FakeTTSEngine,
    FakeTurnDetector,
    FakeVAD,
    ScriptedFakeASREngine,
)
from polyglot.transport.replay_adapter import ReplayAdapter, Scenario

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_scenario(path: Path) -> Scenario:
    with path.open("r", encoding="utf-8") as f:
        return Scenario.model_validate(yaml.safe_load(f))


def build_fake_pipeline(scenario: Scenario, clock: Clock, event_log: EventLog) -> Pipeline:
    scripts = [
        [
            TranscriptPartial(
                text=turn.user_text,
                stable_prefix=turn.user_text,
                lang=scenario.lang,
                lang_prob=1.0,
                is_final=True,
                t_start_ms=0,
                t_end_ms=1000,
            )
        ]
        for turn in scenario.turns
    ]
    return Pipeline(
        vad=FakeVAD(
            events=[
                VADEvent(kind="speech_start", t_ms=0, prob=0.95),
                VADEvent(kind="speech_end", t_ms=800, prob=0.95),
            ]
        ),
        asr=ScriptedFakeASREngine(scripts),
        turn_detector=FakeTurnDetector(prob=1.0),
        retriever=FakeRetriever(passages=[]),
        llm=FakeLLMClient(
            deltas=[LLMDelta(text="Here is a placeholder answer grounded in policy text.")]
        ),
        tts=FakeTTSEngine(),
        tracer=FakeTracer(),
        clock=clock,
        event_log=event_log,
        session_id=scenario.id,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a scenario through the replay harness.")
    parser.add_argument("--scenario", required=True, type=Path)
    parser.add_argument("--profile", choices=["naive", "tuned"], default="naive")
    parser.add_argument("--mode", choices=["simulated", "realtime"], default="simulated")
    args = parser.parse_args()

    settings = load_settings(args.profile)
    scenario = load_scenario(args.scenario)

    run_id = f"{scenario.id}-{args.mode}-{int(time.time())}"
    run_dir = REPO_ROOT / "reports" / run_id
    event_log = EventLog(run_dir / "events.jsonl")

    clock: Clock = RealClock() if args.mode == "realtime" else SimulatedClock()
    pipeline = build_fake_pipeline(scenario, clock, event_log)
    adapter = ReplayAdapter(pipeline)

    results = asyncio.run(adapter.run_scenario(scenario))

    turns_path = run_dir / "turns.jsonl"
    with turns_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(result.record.model_dump_json() + "\n")

    print(f"run_id={run_id} profile={args.profile} flags={settings.flags.model_dump()}")
    print(f"turns={len(results)} events={len(event_log.read_all())}")
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
