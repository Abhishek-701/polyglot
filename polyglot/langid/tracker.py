"""fastText-based language ID and code-switch detection. See SPEC.md Section 8.4.

Verified against installed fasttext-wheel==0.9.2 + numpy 2.5.3:
_FastText.predict(str, k=...) is broken on numpy>=2.0 (`np.array(probs,
copy=False)` raises `ValueError: Unable to avoid copy`). The multi-string
input path (`predict([str, ...], k=...)`) doesn't hit that code — it returns
the raw C++ binding output directly — so this module always calls predict
with a one-item list and unwraps the result, even for a single window.
"""

import fasttext


class LangIDTracker:
    def __init__(
        self,
        model_path: str,
        window_words: int = 5,
        share_threshold: float = 0.2,
    ) -> None:
        self._model = fasttext.load_model(model_path)
        self.window_words = window_words
        self.share_threshold = share_threshold

    def detect_window(self, text: str) -> str:
        labels, _probs = self._model.predict([text], k=1)
        label = labels[0][0]
        return label.removeprefix("__label__")

    def analyze(self, text: str) -> tuple[str, bool, dict[str, float]]:
        """Returns (primary_lang, code_switched, per_lang_share) for the final transcript."""
        words = text.split()
        if not words:
            return "en", False, {}

        windows = [
            " ".join(words[i : i + self.window_words])
            for i in range(0, len(words), self.window_words)
        ]
        counts: dict[str, int] = {}
        for window in windows:
            lang = self.detect_window(window)
            counts[lang] = counts.get(lang, 0) + 1

        total = sum(counts.values())
        shares = {lang: count / total for lang, count in counts.items()}
        primary = max(shares, key=lambda lang: shares[lang])
        code_switched = sum(1 for share in shares.values() if share >= self.share_threshold) > 1
        return primary, code_switched, shares
