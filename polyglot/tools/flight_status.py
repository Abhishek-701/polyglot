"""Mock (default) and optional real flight-status adapter. See SPEC.md
Section 8.15: "Mock is default and is always used in eval for determinism."
"""

import json
from pathlib import Path
from typing import Any, Literal

FIXTURES_PATH = (
    Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures" / "flights.json"
)


def get_flight_status(
    flight_number: str, date: str, provider: Literal["mock", "opensky"] = "mock"
) -> dict[str, Any] | None:
    if provider == "opensky":
        raise NotImplementedError(
            "the opensky flight-status provider is not implemented — SPEC.md 8.15 "
            "marks it optional; mock is the default and is always used in eval."
        )
    with FIXTURES_PATH.open(encoding="utf-8") as f:
        flights: list[dict[str, Any]] = json.load(f)
    for flight in flights:
        if flight["flight_number"].upper() == flight_number.upper() and flight["date"] == date:
            return flight
    return None
