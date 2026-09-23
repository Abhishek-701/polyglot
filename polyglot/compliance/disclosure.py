"""AI disclosure greeting. See SPEC.md Section 8.14: "The greeting discloses
that the caller is speaking with an AI assistant, in the caller's language
once known (greet in English plus a short multilingual prompt at start)."

Only builds the greeting text. Logging the "disclosure" event (already a
valid EventKind, see core/events.py) and any recording-consent event
happens wherever session start is orchestrated (Pipeline/transport), since
that's what has EventLog access — same pattern as hallucination_guard.py.
"""

GREETING_EN = (
    "Hello, you're speaking with an AI assistant that can help with flight "
    "refunds, rebooking, compensation, and baggage questions."
)

_MULTILINGUAL_PROMPTS = {
    "es": "Puedes hablar conmigo en español.",
    "hi": "आप मुझसे हिंदी में भी बात कर सकते हैं।",
    "tl": "Maaari ka ring makipag-usap sa akin sa Tagalog.",
}


def build_greeting() -> str:
    return " ".join([GREETING_EN, *_MULTILINGUAL_PROMPTS.values()])
