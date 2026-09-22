"""PCM conversion, resampling, and buffering. See SPEC.md Section 6.

Not under polyglot/core/ (no mypy --strict requirement), but no transport
imports either — real audio math only, usable by the live and replay
adapters alike.
"""

import numpy as np


def pcm16_to_float32(pcm: bytes) -> np.ndarray:
    """16-bit little-endian mono PCM -> float32 samples in [-1, 1]."""
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def float32_to_pcm16(samples: np.ndarray) -> bytes:
    """float32 samples in [-1, 1] -> 16-bit little-endian mono PCM."""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample float32 mono samples with torchaudio (already a dependency)."""
    if orig_sr == target_sr:
        return samples
    import torch
    import torchaudio.functional as ta_functional

    tensor = torch.from_numpy(samples).unsqueeze(0)
    resampled = ta_functional.resample(tensor, orig_sr, target_sr)
    return resampled.squeeze(0).numpy()


class SampleBuffer:
    """Accumulates float32 mono samples; pops fixed-size windows off the front."""

    def __init__(self) -> None:
        self._samples = np.zeros(0, dtype=np.float32)

    def push(self, samples: np.ndarray) -> None:
        self._samples = np.concatenate([self._samples, samples])

    def pop_window(self, size: int) -> np.ndarray | None:
        if len(self._samples) < size:
            return None
        window = self._samples[:size]
        self._samples = self._samples[size:]
        return window

    def peek_all(self) -> np.ndarray:
        return self._samples

    def __len__(self) -> int:
        return len(self._samples)
