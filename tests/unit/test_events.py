from pathlib import Path

from polyglot.core.events import Event, EventLog


def test_event_log_round_trip(tmp_path: Path) -> None:
    log_path = tmp_path / "session.jsonl"
    log = EventLog(log_path)

    log.append(Event(kind="vad", t_ms=0, session_id="s1", data={"kind": "speech_start"}))
    log.append(
        Event(kind="asr_final", t_ms=500, session_id="s1", turn_id="t1", data={"text": "hi"})
    )

    events = log.read_all()
    assert len(events) == 2
    assert events[0].kind == "vad"
    assert events[1].turn_id == "t1"
    assert events[1].data["text"] == "hi"


def test_event_log_creates_parent_dirs(tmp_path: Path) -> None:
    log_path = tmp_path / "nested" / "session.jsonl"
    EventLog(log_path)
    assert log_path.parent.exists()


def test_event_log_read_all_missing_file(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "missing.jsonl")
    assert log.read_all() == []
