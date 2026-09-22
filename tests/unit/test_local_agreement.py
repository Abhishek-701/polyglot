from hypothesis import given
from hypothesis import strategies as st

from polyglot.asr.local_agreement import LocalAgreement, local_agreement_prefix


def test_prefix_basic_cases() -> None:
    assert local_agreement_prefix("hello world", "hello world foo") == "hello world"
    assert local_agreement_prefix("hello world", "hello there") == "hello"
    assert local_agreement_prefix("", "hello") == ""
    assert local_agreement_prefix("hello", "") == ""
    assert local_agreement_prefix("a b c", "a b c") == "a b c"


def test_local_agreement_class_commits_growing_prefix() -> None:
    agreement = LocalAgreement()
    assert agreement.update("hello") == ""
    assert agreement.update("hello world") == "hello"
    assert agreement.update("hello world foo") == "hello world"
    agreement.reset()
    assert agreement.update("restarted") == ""


_WORD = st.sampled_from(["a", "b", "c", "hello", "world"])
_WORDS = st.lists(_WORD, min_size=0, max_size=5)


@given(_WORDS, _WORDS, _WORDS)
def test_prefix_is_shared_and_maximal(
    common: list[str], tail1: list[str], tail2: list[str]
) -> None:
    prev_words = common + tail1
    curr_words = common + tail2
    stable = local_agreement_prefix(" ".join(prev_words), " ".join(curr_words))
    stable_words = stable.split()
    n = len(stable_words)

    # The stable prefix really is a prefix of both hypotheses.
    assert prev_words[:n] == stable_words
    assert curr_words[:n] == stable_words

    # And it's maximal: if both have a next word, it must differ.
    if n < len(prev_words) and n < len(curr_words):
        assert prev_words[n] != curr_words[n]
