from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOISE_GATE_RMS = 0.01


def compute_rms(frame: np.ndarray) -> float:
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)))


def compute_fft(frame: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    if frame.size == 0:
        return np.array([]), np.array([])
    window = np.hanning(len(frame))
    spectrum = np.fft.rfft(frame * window)
    freqs = np.fft.rfftfreq(len(frame), 1.0 / sr)
    magnitude = np.abs(spectrum)
    return freqs, magnitude


def estimate_f0(frame: np.ndarray, sr: int, fmin: float = 50.0, fmax: float = 1000.0) -> Optional[float]:
    if frame.size == 0:
        return None
    frame = frame.astype(np.float32)
    frame = frame - np.mean(frame)
    rms = compute_rms(frame)
    if rms < NOISE_GATE_RMS:
        return None
    if np.max(np.abs(frame)) < 1e-4:
        return None

    autocorr = np.correlate(frame, frame, mode="full")
    autocorr = autocorr[len(autocorr) // 2:]
    if autocorr[0] == 0:
        return None

    lag_min = int(sr / fmax)
    lag_max = int(sr / fmin)
    lag_max = min(lag_max, len(autocorr) - 1)
    if lag_max <= lag_min:
        return None

    segment = autocorr[lag_min:lag_max]
    peak_index = np.argmax(segment)
    lag = lag_min + peak_index
    peak_value = autocorr[lag] / autocorr[0]
    if peak_value < 0.2:
        return None
    return float(sr / lag)


def note_from_f0(f0: Optional[float]) -> Tuple[str, float]:
    if not f0 or f0 <= 0:
        return "--", 0.0
    midi = 69 + 12 * math.log2(f0 / 440.0)
    rounded = int(round(midi))
    cents = (midi - rounded) * 100
    name = NOTE_NAMES[rounded % 12]
    octave = rounded // 12 - 1
    return f"{name}{octave}", float(cents)


def f0_to_midi(f0: Optional[float]) -> Optional[float]:
    if not f0 or f0 <= 0:
        return None
    return 69 + 12 * math.log2(f0 / 440.0)


def midi_to_note(midi: Optional[float]) -> str:
    if midi is None:
        return "--"
    rounded = int(round(midi))
    name = NOTE_NAMES[rounded % 12]
    octave = rounded // 12 - 1
    return f"{name}{octave}"


def note_to_freq(note: str) -> Optional[float]:
    note = note.strip().upper()
    if len(note) < 2:
        return None
    if note[1] == "#":
        name = note[:2]
        octave_str = note[2:]
    else:
        name = note[:1]
        octave_str = note[1:]
    if name not in NOTE_NAMES:
        return None
    try:
        octave = int(octave_str)
    except ValueError:
        return None
    midi = (octave + 1) * 12 + NOTE_NAMES.index(name)
    return 440.0 * 2 ** ((midi - 69) / 12)


def compute_f0_contour(audio: np.ndarray, sr: int, frame_size: int = 1024, hop: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    if audio.size == 0:
        return np.array([]), np.array([])
    frames = []
    for start in range(0, len(audio) - frame_size + 1, hop):
        frames.append(audio[start:start + frame_size])
    if not frames:
        return np.array([]), np.array([])
    f0s = [estimate_f0(frame, sr) or 0.0 for frame in frames]
    times = np.arange(len(f0s)) * (hop / sr)
    return times, np.array(f0s, dtype=np.float32)


def compute_mel_spectrogram(
    audio: np.ndarray, sr: int, n_fft: int = 1024, hop: int = 256, n_mels: int = 64
) -> Tuple[np.ndarray, np.ndarray]:
    if audio.size == 0:
        return np.array([[]]), np.array([])
    window = np.hanning(n_fft)
    frames = []
    for start in range(0, len(audio) - n_fft + 1, hop):
        frame = audio[start:start + n_fft]
        spectrum = np.fft.rfft(frame * window)
        frames.append(np.abs(spectrum) ** 2)
    if not frames:
        return np.array([[]]), np.array([])
    power = np.stack(frames, axis=1)
    mel_fb = _mel_filterbank(sr, n_fft, n_mels)
    mel = mel_fb @ power
    mel_db = 10 * np.log10(np.maximum(mel, 1e-10))
    mel_db = mel_db[::-1, :]
    times = np.arange(mel_db.shape[1]) * (hop / sr)
    return mel_db, times


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    def hz_to_mel(hz: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10 ** (mel / 2595.0) - 1.0)

    fmin = 0.0
    fmax = sr / 2.0
    mels = np.linspace(hz_to_mel(np.array([fmin]))[0], hz_to_mel(np.array([fmax]))[0], n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center == left:
            center += 1
        if right == center:
            right += 1
        for j in range(left, center):
            fb[i, j] = (j - left) / max(center - left, 1)
        for j in range(center, right):
            fb[i, j] = (right - j) / max(right - center, 1)
    return fb
