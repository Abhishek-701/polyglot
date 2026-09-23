"""Mock booking lookup. See SPEC.md Section 8.15.

Fixture data only, always used in eval for determinism; there's no real
adapter for this one (unlike flight_status, which has an optional real
provider).
"""

import json
from pathlib import Path
from typing import Any

FIXTURES_PATH = (
    Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures" / "bookings.json"
)


def lookup_booking(confirmation_code: str, last_name: str) -> dict[str, Any] | None:
    with FIXTURES_PATH.open(encoding="utf-8") as f:
        bookings: list[dict[str, Any]] = json.load(f)
    for booking in bookings:
        if (
            booking["confirmation_code"].upper() == confirmation_code.upper()
            and booking["last_name"].lower() == last_name.lower()
        ):
            return booking
    return None
