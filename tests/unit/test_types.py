from polyglot.core.types import (
    AudioFrame,
    LLMDelta,
    Message,
    Passage,
    TranscriptPartial,
    TurnRecord,
    VADEvent,
)


def test_audio_frame_defaults() -> None:
    frame = AudioFrame(pcm=b"\x00\x01", t_ms=0)
    assert frame.sample_rate == 16000


def test_vad_event() -> None:
    event = VADEvent(kind="speech_start", t_ms=100, prob=0.9)
    assert event.kind == "speech_start"


def test_transcript_partial_optional_fields() -> None:
    partial = TranscriptPartial(
        text="hola",
        stable_prefix="hola",
        lang="es",
        lang_prob=0.98,
        is_final=False,
        t_start_ms=0,
        t_end_ms=500,
    )
    assert partial.no_speech_prob is None
    assert partial.avg_logprob is None


def test_passage() -> None:
    passage = Passage(id="p1", doc_id="dot-refunds", text="...", source_url="https://x", score=0.8)
    assert passage.score == 0.8


def test_message_defaults() -> None:
    message = Message(role="user", content="hi")
    assert message.interrupted is False
    assert message.lang is None


def test_llm_delta_tool_call() -> None:
    delta = LLMDelta(tool_call={"name": "get_flight_status", "args": {"flight_number": "AA1"}})
    assert delta.tool_call is not None
    assert delta.tool_call["name"] == "get_flight_status"


def test_turn_record() -> None:
    record = TurnRecord(
        turn_id="t1",
        session_id="s1",
        lang="en",
        user_text="where is my refund",
        assistant_text_generated="...",
        assistant_text_spoken="...",
        intent="refund",
        passages=[],
        tool_calls=[],
        code_switched=False,
        interrupted=False,
        timings={"llm.ttft": 120},
    )
    assert record.timings["llm.ttft"] == 120
