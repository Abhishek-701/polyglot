"""Zero-shot multilingual intent classification. See SPEC.md Section 8.9.

Verified against installed transformers==5.17.0 +
MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7: the spec's bare
label words ("refund", "smalltalk", ...) score very poorly with this
model's default hypothesis template — e.g. an obvious refund request
scored "smalltalk" highest by a wide margin in testing. Descriptive label
phrases plus a domain-specific hypothesis_template measurably improved
accuracy (5/7 on a small hand-written check, versus clearly-wrong top-1s
before), but confidence stays modest (0.2-0.35) even when correct. This is
expected (SPEC.md 8.9: "This is expected to be imperfect. The goal is
measurement.") — confidence-floor routing to "clarify" (owned by
policy/graph.py, not this module) is what actually handles the remaining
error rate, not further prompt tweaking here.
"""

from dataclasses import dataclass
from typing import Literal

from transformers import Pipeline, pipeline

IntentLabel = Literal[
    "refund",
    "rebooking",
    "compensation",
    "baggage",
    "flight_status",
    "booking_lookup",
    "smalltalk",
    "other",
]

# Bare label words score poorly (see module docstring); descriptive phrases
# are fed to the classifier and mapped back to the SPEC.md 8.9 label set.
_LABEL_PHRASES: dict[IntentLabel, str] = {
    "refund": "a request for a refund",
    "rebooking": "a request to rebook onto a different flight",
    "compensation": "a request for compensation money for a delay or cancellation",
    "baggage": "a complaint about lost or damaged luggage",
    "flight_status": "a question about whether a flight is delayed or on time",
    "booking_lookup": "a request to look up a reservation",
    "smalltalk": "a greeting or casual remark with no travel request",
    "other": "an unrelated question",
}
_PHRASE_TO_LABEL = {phrase: label for label, phrase in _LABEL_PHRASES.items()}
_HYPOTHESIS_TEMPLATE = "This airline customer service message is {}."

_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
_classifier: Pipeline | None = None


def get_classifier() -> Pipeline:
    global _classifier
    if _classifier is None:
        _classifier = pipeline("zero-shot-classification", model=_MODEL_NAME)
    return _classifier


@dataclass
class IntentResult:
    label: IntentLabel
    confidence: float


def classify_intent(text: str, classifier: Pipeline | None = None) -> IntentResult:
    classifier = classifier or get_classifier()
    result = classifier(
        text, list(_LABEL_PHRASES.values()), hypothesis_template=_HYPOTHESIS_TEMPLATE
    )
    top_label = _PHRASE_TO_LABEL[result["labels"][0]]
    return IntentResult(label=top_label, confidence=result["scores"][0])
