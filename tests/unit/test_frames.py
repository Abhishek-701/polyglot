import numpy as np

from polyglot.audio.frames import SampleBuffer, float32_to_pcm16, pcm16_to_float32, resample


def test_pcm16_float32_round_trip() -> None:
    original = np.array([0, 1000, -1000, 32767, -32768], dtype="<i2")
    floats = pcm16_to_float32(original.tobytes())
    back = np.frombuffer(float32_to_pcm16(floats), dtype="<i2")
    np.testing.assert_allclose(back, original, atol=1)


def test_resample_identity_when_same_rate() -> None:
    samples = np.random.default_rng(0).normal(0, 0.1, 1000).astype(np.float32)
    result = resample(samples, 16000, 16000)
    assert result is samples


def test_resample_changes_length_and_rate() -> None:
    samples = np.zeros(1600, dtype=np.float32)
    result = resample(samples, 16000, 8000)
    assert abs(len(result) - 800) <= 2


def test_sample_buffer_pop_window_waits_for_enough_samples() -> None:
    buf = SampleBuffer()
    buf.push(np.zeros(300, dtype=np.float32))
    assert buf.pop_window(512) is None
    assert len(buf) == 300

    buf.push(np.zeros(300, dtype=np.float32))
    window = buf.pop_window(512)
    assert window is not None
    assert len(window) == 512
    assert len(buf) == 88


def test_sample_buffer_pop_window_leaves_remainder() -> None:
    buf = SampleBuffer()
    buf.push(np.arange(600, dtype=np.float32))
    window = buf.pop_window(512)
    assert window is not None
    np.testing.assert_array_equal(window, np.arange(512, dtype=np.float32))
    np.testing.assert_array_equal(buf.peek_all(), np.arange(512, 600, dtype=np.float32))
