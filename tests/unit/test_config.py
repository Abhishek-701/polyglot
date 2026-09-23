from polyglot.core.config import load_settings


def test_naive_profile_flags() -> None:
    settings = load_settings("naive")
    assert settings.flags.turn_detection == "silence"
    assert settings.flags.speculative_retrieval is False
    assert settings.flags.prefix_cache_prompt_order is False
    assert settings.flags.history_compaction is False
    assert settings.flags.tts_chunking == "full"
    assert settings.flags.semantic_cache is False
    assert settings.flags.filler_on_tool_call is False
    assert settings.flags.asr_router_parakeet is False
    assert settings.flags.retrieval_bridge == "mt"


def test_tuned_profile_flags() -> None:
    settings = load_settings("tuned")
    assert settings.flags.turn_detection == "semantic"
    assert settings.flags.speculative_retrieval is True
    assert settings.flags.prefix_cache_prompt_order is True
    assert settings.flags.history_compaction is True
    assert settings.flags.tts_chunking == "clause"
    assert settings.flags.semantic_cache is True
    assert settings.flags.filler_on_tool_call is True
    assert settings.flags.asr_router_parakeet is True


def test_languages_come_from_default() -> None:
    settings = load_settings("naive")
    assert set(settings.languages) == {"en", "es", "hi", "zh"}
