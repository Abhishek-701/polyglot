"""LocalAgreement-2 streaming stabilization. See SPEC.md Section 8.2.

Commits the longest common word-level prefix of the last two hypotheses.
"""


def local_agreement_prefix(previous: str, current: str) -> str:
    prev_words = previous.split()
    curr_words = current.split()
    common: list[str] = []
    for prev_word, curr_word in zip(prev_words, curr_words, strict=False):
        if prev_word != curr_word:
            break
        common.append(curr_word)
    return " ".join(common)


class LocalAgreement:
    """Stateful wrapper: feed successive hypotheses, get back the stable prefix."""

    def __init__(self) -> None:
        self._previous: str | None = None

    def update(self, hypothesis: str) -> str:
        stable = (
            "" if self._previous is None else local_agreement_prefix(self._previous, hypothesis)
        )
        self._previous = hypothesis
        return stable

    def reset(self) -> None:
        self._previous = None
