"""RealClock and SimulatedClock. See SPEC.md Section 7.2 and 11.1.

This is the ONLY module under polyglot/ allowed to call time.time,
time.monotonic, or asyncio.sleep. tests/unit/test_clock_usage.py enforces
this with a static check over the source tree.
"""

import asyncio
import bisect
import time


class RealClock:
    """Wall-clock time via time.monotonic(), real asyncio.sleep for waits."""

    def __init__(self) -> None:
        self._start = time.monotonic()

    def now_ms(self) -> int:
        return int((time.monotonic() - self._start) * 1000)

    async def sleep_ms(self, ms: int) -> None:
        await asyncio.sleep(ms / 1000)


class SimulatedClock:
    """Manually advanced clock for deterministic replay (SPEC.md 11.1).

    sleep_ms() never sleeps for real; it registers a waiter and suspends
    until advance() moves simulated time past the waiter's deadline. Time
    only moves when advance() is called explicitly.
    """

    def __init__(self, start_ms: int = 0) -> None:
        self._now_ms = start_ms
        self._waiters: list[tuple[int, asyncio.Event]] = []

    def now_ms(self) -> int:
        return self._now_ms

    async def sleep_ms(self, ms: int) -> None:
        if ms <= 0:
            return
        deadline = self._now_ms + ms
        event = asyncio.Event()
        bisect.insort(self._waiters, (deadline, event), key=lambda w: w[0])
        await event.wait()

    def advance(self, ms: int) -> None:
        if ms < 0:
            raise ValueError("cannot advance a SimulatedClock backwards")
        self._now_ms += ms
        while self._waiters and self._waiters[0][0] <= self._now_ms:
            _, event = self._waiters.pop(0)
            event.set()
