from polyglot.compliance.disclosure import GREETING_EN, build_greeting


def test_greeting_includes_english_disclosure() -> None:
    greeting = build_greeting()
    assert GREETING_EN in greeting
    assert "AI assistant" in greeting


def test_greeting_includes_all_other_languages() -> None:
    greeting = build_greeting()
    assert "español" in greeting
    assert "हिंदी" in greeting
    assert "Tagalog" in greeting
