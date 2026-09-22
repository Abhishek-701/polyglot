"""Replay transport adapter. See SPEC.md Section 11.1.

Drives a Pipeline from a scripted scenario instead of a live transport. Both
this adapter and the (future) LiveKit adapter feed the same Pipeline object
(docs/decisions/001-own-pipeline.md).

M1 scope: turns are sequential (`start: after_agent_done` only); no real
audio files. Each turn carries `user_text` used only to build the fake ASR
script (in eval/replay.py) — the frames this adapter streams are a single
placeholder AudioFrame per turn, since VAD/ASR are fakes in M1 and don't
inspect pcm content. Real WAV loading (`user_audio`, barge-in timing via
`after_agent_start_ms`) lands in M2+ once real audio and a real ASR exist;
tracked as an open question in PROGRESS.md.
"""

from collections.abc import AsyncIterator

from pydantic import BaseModel

from polyglot.core.pipeline import Pipeline, TurnResult
from polyglot.core.types import AudioFrame, Message


class ScenarioTurn(BaseModel):
    user_text: str


class Scenario(BaseModel):
    id: str
    lang: str
    turns: list[ScenarioTurn]


async def _single_placeholder_frame(t_ms: int) -> AsyncIterator[AudioFrame]:
    yield AudioFrame(pcm=b"", t_ms=t_ms)


class ReplayAdapter:
    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

    async def run_scenario(self, scenario: Scenario) -> list[TurnResult]:
        history: list[Message] = []
        results: list[TurnResult] = []

        for _turn in scenario.turns:
            frames = _single_placeholder_frame(self.pipeline.clock.now_ms())
            result = await self.pipeline.run_turn(frames, scenario.lang, history)

            history.append(
                Message(role="user", content=result.record.user_text, lang=scenario.lang)
            )
            history.append(
                Message(
                    role="assistant",
                    content=result.record.assistant_text_spoken,
                    lang=scenario.lang,
                    interrupted=result.record.interrupted,
                )
            )
            results.append(result)

        return results
