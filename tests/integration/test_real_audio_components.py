"""M2 real-component integration tests. See SPEC.md Section 14: "no test
depends on network access except those marked @pytest.mark.network, skipped
in CI." Requires `uv run python scripts/download_datasets.py` to have been
run first (FLEURS + the fastText LID model); skipped gracefully otherwise.
"""

import json
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from polyglot.asr.faster_whisper_engine import FasterWhisperEngine
from polyglot.audio.frames import float32_to_pcm16
from polyglot.audio.vad import SileroVAD
from polyglot.core.types import AudioFrame
from polyglot.langid.tracker import LangIDTracker

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FLEURS_EN = REPO_ROOT / "data" / "cache" / "fleurs" / "en"
LID_MODEL = REPO_ROOT / "data" / "cache" / "models" / "lid.176.ftz"

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        not (FLEURS_EN / "manifest.jsonl").exists(),
        reason="run 'uv run python scripts/download_datasets.py --only fleurs' first",
    ),
]


def _load_en_clip() -> tuple[np.ndarray, str]:
    manifest = FLEURS_EN / "manifest.jsonl"
    entry = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])
    data, sample_rate = sf.read(REPO_ROOT / entry["path"])
    assert sample_rate == 16000
    return data.astype(np.float32), entry["transcription"]


async def _frames(samples: np.ndarray, chunk_size: int = 320) -> AsyncIterator[AudioFrame]:
    for i in range(0, len(samples), chunk_size):
        chunk = samples[i : i + chunk_size]
        yield AudioFrame(pcm=float32_to_pcm16(chunk), t_ms=round(i / 16000 * 1000))


async def test_silero_vad_detects_real_speech() -> None:
    samples, _reference = _load_en_clip()
    vad = SileroVAD()

    saw_speech_start = False
    async for frame in _frames(samples):
        for event in vad.process(frame):
            if event.kind == "speech_start":
                saw_speech_start = True

    assert saw_speech_start


async def test_faster_whisper_transcribes_real_speech() -> None:
    samples, _reference = _load_en_clip()
    # A huge asr_step_ms means no interim decode fires; only the final
    # full-buffer decode runs once frames are exhausted, keeping this fast.
    engine = FasterWhisperEngine(model_size="small", asr_step_ms=999_999)

    partials = [p async for p in engine.stream(_frames(samples), "en")]

    assert len(partials) == 1
    final = partials[0]
    assert final.is_final
    assert len(final.text) > 0
    assert final.lang == "en"


@pytest.mark.skipif(
    not LID_MODEL.exists(),
    reason="run 'uv run python scripts/download_datasets.py --only lid' first",
)
def test_langid_tracker_detects_english() -> None:
    tracker = LangIDTracker(str(LID_MODEL))
    lang, code_switched, _shares = tracker.analyze("my flight was cancelled yesterday")
    assert lang == "en"
    assert not code_switched


@pytest.mark.skipif(
    not LID_MODEL.exists(),
    reason="run 'uv run python scripts/download_datasets.py --only lid' first",
)
def test_langid_tracker_detects_code_switch() -> None:
    tracker = LangIDTracker(str(LID_MODEL))
    # Long runs of each language so per-window fastText calls aren't ambiguous.
    text = (
        "hola como estas hoy espero que muy bien y de buen humor "
        "my flight got cancelled and I really need help with a refund"
    )
    _lang, code_switched, shares = tracker.analyze(text)
    assert code_switched
    assert len(shares) > 1
