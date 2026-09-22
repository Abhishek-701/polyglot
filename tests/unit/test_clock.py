import asyncio

from polyglot.core.clock import RealClock, SimulatedClock


def test_real_clock_starts_near_zero() -> None:
    clock = RealClock()
    assert clock.now_ms() >= 0
    assert clock.now_ms() < 50


async def test_real_clock_sleep_advances_time() -> None:
    clock = RealClock()
    await clock.sleep_ms(20)
    assert clock.now_ms() >= 20


def test_simulated_clock_starts_at_given_time() -> None:
    clock = SimulatedClock(start_ms=1000)
    assert clock.now_ms() == 1000


def test_simulated_clock_does_not_advance_on_its_own() -> None:
    clock = SimulatedClock()
    assert clock.now_ms() == 0
    clock.advance(500)
    assert clock.now_ms() == 500


async def test_simulated_clock_sleep_only_resolves_on_advance() -> None:
    clock = SimulatedClock()
    woke = False

    async def waiter() -> None:
        nonlocal woke
        await clock.sleep_ms(100)
        woke = True

    task = asyncio.ensure_future(waiter())
    await asyncio.sleep(0)
    assert not woke

    clock.advance(50)
    await asyncio.sleep(0)
    assert not woke

    clock.advance(50)
    await task
    assert woke
