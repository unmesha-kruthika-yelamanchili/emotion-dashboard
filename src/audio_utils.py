"""Audio loading + chunking. Mirrors the preprocessing used during fine-tuning."""
from __future__ import annotations

import numpy as np
import librosa

from .config import (SAMPLE_RATE, CHUNK_SEC, MIN_CHUNK_SEC,
                     MAX_CHUNKS_PER_CLIP, SILENCE_RMS)


def load_audio(path_or_file, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load any audio as 16 kHz mono float32.

    NOTE: training did NOT normalise the waveform (plain librosa.load), so we
    don't either. Adding normalisation here would shift inputs away from what
    the model saw.
    """
    wav, _ = librosa.load(path_or_file, sr=sr, mono=True)
    return wav.astype(np.float32)


def split_into_chunks(wav: np.ndarray, sr: int = SAMPLE_RATE):
    """Cut a waveform into consecutive CHUNK_SEC windows.

    Returns a list of (start_sec, end_sec, samples, is_silent).
    """
    size, min_len = int(CHUNK_SEC * sr), int(MIN_CHUNK_SEC * sr)
    out = []
    for start in range(0, len(wav), size):
        seg = wav[start:start + size]
        if len(seg) < min_len and out:          # drop a tiny leftover tail
            break
        if len(seg) < 400:                      # same minimum length as training
            seg = np.pad(seg, (0, 400 - len(seg)))
        rms = float(np.sqrt(np.mean(seg ** 2)))
        out.append((start / sr, (start + len(seg)) / sr, seg, rms < SILENCE_RMS))
        if len(out) >= MAX_CHUNKS_PER_CLIP:
            break
    if not out:                                 # empty file
        out.append((0.0, 400 / sr, np.zeros(400, dtype=np.float32), True))
    return out
