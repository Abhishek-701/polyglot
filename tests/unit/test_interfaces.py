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
from polyglot.fakes import (
    FakeASREngine,
    FakeClock,
    FakeLLMClient,
    FakeRetriever,
    FakeTracer,
    FakeTTSEngine,
    FakeTurnDetector,
    FakeVAD,
)


def test_fake_clock_satisfies_protocol() -> None:
    assert isinstance(FakeClock(), Clock)


def test_fake_vad_satisfies_protocol() -> None:
    assert isinstance(FakeVAD(), VAD)


def test_fake_asr_satisfies_protocol() -> None:
    assert isinstance(FakeASREngine(), ASREngine)


def test_fake_turn_detector_satisfies_protocol() -> None:
    assert isinstance(FakeTurnDetector(), TurnDetector)


def test_fake_retriever_satisfies_protocol() -> None:
    assert isinstance(FakeRetriever(), Retriever)


def test_fake_llm_client_satisfies_protocol() -> None:
    assert isinstance(FakeLLMClient(), LLMClient)


def test_fake_tts_engine_satisfies_protocol() -> None:
    assert isinstance(FakeTTSEngine(), TTSEngine)


def test_fake_tracer_satisfies_protocol() -> None:
    assert isinstance(FakeTracer(), Tracer)
