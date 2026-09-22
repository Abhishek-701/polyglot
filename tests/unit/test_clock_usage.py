"""Enforces CLAUDE.md architecture rule: all time goes through the injected
Clock. Nothing under polyglot/ may call time.time, time.monotonic, or
asyncio.sleep directly, except polyglot/core/clock.py itself.
"""

import re
from pathlib import Path

FORBIDDEN = re.compile(r"\b(time\.time|time\.monotonic|asyncio\.sleep)\s*\(")
ALLOWED_FILE = Path("polyglot") / "core" / "clock.py"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_no_direct_time_or_sleep_calls_outside_clock() -> None:
    violations: list[str] = []
    for path in (REPO_ROOT / "polyglot").rglob("*.py"):
        relative = path.relative_to(REPO_ROOT)
        if relative == ALLOWED_FILE:
            continue
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if FORBIDDEN.search(line):
                violations.append(f"{relative}:{line_no}: {line.strip()}")

    assert not violations, "Direct time/sleep calls outside core/clock.py:\n" + "\n".join(
        violations
    )
