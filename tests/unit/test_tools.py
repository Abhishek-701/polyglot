import pytest

from polyglot.tools.booking import lookup_booking
from polyglot.tools.flight_status import get_flight_status


def test_lookup_booking_found() -> None:
    booking = lookup_booking("ABC123", "Garcia")
    assert booking is not None
    assert booking["passenger_name"] == "Maria Garcia"


def test_lookup_booking_case_insensitive() -> None:
    assert lookup_booking("abc123", "garcia") is not None


def test_lookup_booking_not_found() -> None:
    assert lookup_booking("ZZZ999", "Nobody") is None


def test_lookup_booking_wrong_last_name() -> None:
    assert lookup_booking("ABC123", "WrongName") is None


def test_get_flight_status_found() -> None:
    flight = get_flight_status("DL200", "2026-09-26")
    assert flight is not None
    assert flight["status"] == "cancelled"


def test_get_flight_status_not_found() -> None:
    assert get_flight_status("ZZ999", "2026-01-01") is None


def test_get_flight_status_opensky_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        get_flight_status("AA100", "2026-09-25", provider="opensky")
