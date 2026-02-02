from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
from PyQt6 import QtCore

from audio.ring_buffer import RingBuffer
from audio.dsp import note_to_freq


class AudioEngine(QtCore.QObject):
    error = QtCore.pyqtSignal(str)
    status = QtCore.pyqtSignal(str)

    def __init__(self, sample_rate: int = 44100, channels: int = 1, block_size: int = 1024):
        super().__init__()
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = block_size

        self.stream: Optional[sd.Stream] = None
        self.recording = False
        self.preview = False
        self.bgm_during_recording = False
        self.bgm_gain = 0.5
        self.overlay_gain = 0.5
        self.monitor_gain = 0.0
        self.pre_roll_samples = 0

        self._ring = RingBuffer(size=sample_rate * 5)
        self._record_file: Optional[sf.SoundFile] = None
        self._record_lock = threading.Lock()
        self._bgm_data: Optional[np.ndarray] = None
        self._bgm_playlist: list[np.ndarray] = []
        self._bgm_index = 0
        self._bgm_pos = 0
        self._bgm_overlay: Optional[np.ndarray] = None
        self._bgm_overlay_pos = 0
        self._overlay_enabled = True
        self._record_final_path: Optional[Path] = None
        self._record_temp_path: Optional[Path] = None
        self._recorded_chunks: list[np.ndarray] = []
        self._recorded_samples = 0
        self._recorded_cache: Optional[np.ndarray] = None
        self._recorded_cache_samples = 0
        self._last_recorded: Optional[np.ndarray] = None
        self._record_vis_stride = 4
        self._out_buffer: Optional[np.ndarray] = None
        self._bgm_buffer: Optional[np.ndarray] = None
        self._overlay_buffer: Optional[np.ndarray] = None
        self._monitor_buffer: Optional[np.ndarray] = None
        self.device: Optional[tuple[Optional[int], Optional[int]]] = None
        self._last_bgm_chunk: Optional[np.ndarray] = None
        self.trim_tail_ms = 50

    def set_bgm_gain(self, value: float) -> None:
        self.bgm_gain = float(max(0.0, min(1.0, value)))

    def set_overlay_gain(self, value: float) -> None:
        self.overlay_gain = float(max(0.0, min(1.0, value)))

    def set_monitor_gain(self, value: float) -> None:
        self.monitor_gain = float(max(0.0, min(1.0, value)))

    def set_pre_roll_ms(self, ms: int) -> None:
        self.pre_roll_samples = int(self.sample_rate * (ms / 1000.0))

    def load_bgm_wav(self, path: Path) -> None:
        data, sr = sf.read(str(path), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != self.sample_rate:
            data = self._resample(data, sr, self.sample_rate)
        self._bgm_data = data
        self._bgm_playlist = [data]
        self._bgm_index = 0
        self._bgm_pos = 0
        self.status.emit(f"BGM loaded: {path.name}")

    def load_bgm_playlist(self, paths: list[Path]) -> None:
        playlist: list[np.ndarray] = []
        for path in paths:
            data, sr = sf.read(str(path), dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            if sr != self.sample_rate:
                data = self._resample(data, sr, self.sample_rate)
            playlist.append(data.astype(np.float32))
        if not playlist:
            raise ValueError("No WAV files in playlist")
        self._bgm_playlist = playlist
        self._bgm_index = 0
        self._bgm_data = self._bgm_playlist[0]
        self._bgm_pos = 0
        self.status.emit(f"BGM playlist loaded: {len(playlist)} files")

    def generate_bgm(
        self,
        note: str,
        duration_sec: float = 2.0,
        pre_silence_sec: float = 0.0,
        post_silence_sec: float = 0.0,
    ) -> None:
        freq = note_to_freq(note)
        if not freq:
            raise ValueError("Invalid note format. Use like A4 or C#3.")
        tone = self._tone(note, duration_sec)
        pre = np.zeros(int(self.sample_rate * pre_silence_sec), dtype=np.float32)
        post = np.zeros(int(self.sample_rate * post_silence_sec), dtype=np.float32)
        self._bgm_data = np.concatenate((pre, tone, post))
        self._bgm_playlist = [self._bgm_data]
        self._bgm_index = 0
        self._bgm_pos = 0
        self.status.emit(f"BGM generated: {note}")

    def generate_bgm_mora(
        self,
        note: str,
        bpm: float,
        mora_count: int,
        gap_sec: float = 0.03,
        pre_silence_sec: float = 0.03,
        post_silence_sec: float = 0.03,
    ) -> None:
        if bpm <= 0 or mora_count <= 0:
            raise ValueError("Invalid BPM or mora count")
        mora_dur = 60.0 / bpm
        gap = max(0.0, min(gap_sec, mora_dur * 0.5))
        tone_dur = max(0.01, mora_dur - gap)
        parts = []
        for i in range(mora_count):
            parts.append(self._tone(note, tone_dur))
            if i < mora_count - 1:
                parts.append(np.zeros(int(self.sample_rate * gap), dtype=np.float32))
        tone = np.concatenate(parts) if parts else np.array([], dtype=np.float32)
        pre = np.zeros(int(self.sample_rate * pre_silence_sec), dtype=np.float32)
        post = np.zeros(int(self.sample_rate * post_silence_sec), dtype=np.float32)
        self._bgm_data = np.concatenate((pre, tone, post))
        self._bgm_playlist = [self._bgm_data]
        self._bgm_index = 0
        self._bgm_pos = 0
        self.status.emit(f"BGM generated (mora): {note}")

    def generate_metronome(
        self,
        bpm: float,
        duration_sec: float,
        click_hz: float = 2000.0,
        click_len_ms: float = 20.0,
        accent_every: int = 4,
    ) -> None:
        if bpm <= 0 or duration_sec <= 0:
            raise ValueError("Invalid BPM or duration")
        beat_sec = 60.0 / bpm
        total_samples = int(self.sample_rate * duration_sec)
        audio = np.zeros(total_samples, dtype=np.float32)

        click_len = max(1, int(self.sample_rate * (click_len_ms / 1000.0)))
        t = np.linspace(0, click_len / self.sample_rate, click_len, endpoint=False)
        base_click = np.sin(2 * np.pi * click_hz * t).astype(np.float32)
        env = np.minimum(1.0, t / 0.002) * np.minimum(1.0, (t[-1] - t) / 0.005 if t[-1] > 0 else 1.0)
        env = np.clip(env, 0.0, 1.0).astype(np.float32)
        base_click *= env

        beat = 0
        pos = 0.0
        while True:
            start = int(pos * self.sample_rate)
            if start >= total_samples:
                break
            end = min(total_samples, start + click_len)
            click = base_click[: end - start].copy()
            if accent_every > 0 and beat % accent_every == 0:
                click *= 1.5
            audio[start:end] += click
            beat += 1
            pos += beat_sec

        audio = np.clip(audio, -1.0, 1.0)
        self._bgm_data = audio
        self._bgm_playlist = [audio]
        self._bgm_index = 0
        self._bgm_pos = 0
        self.status.emit("BGM generated (metronome)")

    def _tone(self, note: str, duration_sec: float) -> np.ndarray:
        freq = note_to_freq(note)
        if not freq:
            raise ValueError("Invalid note format. Use like A4 or C#3.")
        tone_len = int(self.sample_rate * duration_sec)
        t = np.linspace(0, duration_sec, tone_len, endpoint=False)
        wave = 0.2 * np.sin(2 * np.pi * freq * t)
        env = np.minimum(1.0, t / 0.02) * np.minimum(1.0, (duration_sec - t) / 0.05)
        env = np.clip(env, 0.0, 1.0)
        return (wave * env).astype(np.float32)

    def set_bgm_overlay(self, note: str, duration_sec: float = 2.0) -> None:
        freq = note_to_freq(note)
        if not freq:
            raise ValueError("Invalid note format. Use like A4 or C#3.")
        t = np.linspace(0, duration_sec, int(self.sample_rate * duration_sec), endpoint=False)
        wave = 0.2 * np.sin(2 * np.pi * freq * t)
        env = np.minimum(1.0, t / 0.02) * np.minimum(1.0, (duration_sec - t) / 0.05)
        env = np.clip(env, 0.0, 1.0)
        self._bgm_overlay = (wave * env).astype(np.float32)
        self._bgm_overlay_pos = 0
        self.status.emit(f"BGM overlay set: {note}")

    def clear_bgm_overlay(self) -> None:
        self._overlay_enabled = True

    def set_overlay_enabled(self, enabled: bool) -> None:
        self._overlay_enabled = bool(enabled)

    def start_recording(self, out_path: Path, bgm_during: bool = True) -> None:
        if self.recording:
            return
        if self._bgm_data is None and bgm_during:
            self.status.emit("No BGM loaded, recording without BGM.")
            bgm_during = False
        self.bgm_during_recording = bgm_during
        self.recording = True
        self.preview = False
        self._bgm_pos = -self.pre_roll_samples if bgm_during else 0
        if self._bgm_overlay is not None:
            self._bgm_overlay_pos = 0
        self._recorded_chunks = []
        self._recorded_samples = 0
        self._recorded_cache = None
        self._recorded_cache_samples = 0
        self._last_recorded = None

        subtype = "PCM_16"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._record_final_path = out_path
        self._record_temp_path = out_path.with_name(f"{out_path.stem}.tmp{out_path.suffix}")
        self._record_file = sf.SoundFile(
            str(self._record_temp_path),
            mode="w",
            samplerate=self.sample_rate,
            channels=self.channels,
            subtype=subtype,
        )
        self._ensure_stream()
        self.status.emit("Recording started")

    def stop_recording(self) -> None:
        if not self.recording:
            return
        self.recording = False
        with self._record_lock:
            if self._record_file:
                self._record_file.close()
                self._record_file = None
        if self._record_temp_path and self._record_final_path:
            try:
                data, sr = sf.read(str(self._record_temp_path), dtype="float32")
                trim_samples = int(self.trim_tail_ms * sr / 1000.0)
                if trim_samples > 0 and len(data) > trim_samples:
                    data = data[:-trim_samples]
                sf.write(
                    str(self._record_final_path),
                    data,
                    sr,
                    subtype="PCM_16",
                )
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                self._last_recorded = data.astype(np.float32, copy=False)
                try:
                    self._record_temp_path.unlink(missing_ok=True)
                except Exception:
                    pass
            except Exception:
                try:
                    self._record_temp_path.replace(self._record_final_path)
                except Exception:
                    self.status.emit("Failed to finalize recording file")
        self.status.emit("Recording stopped")
        if not self.preview:
            self._stop_stream_if_idle()

    def play_bgm(self) -> None:
        if self._bgm_data is None:
            raise ValueError("No BGM loaded")
        self.preview = True
        self.recording = False
        self._bgm_pos = 0
        if self._bgm_overlay is not None:
            self._bgm_overlay_pos = 0
        self._ensure_stream()
        self.status.emit("BGM preview")

    def stop_bgm(self) -> None:
        if not self.preview:
            return
        self.preview = False
        self._stop_stream_if_idle()
        self.status.emit("BGM stopped")

    def is_active(self) -> bool:
        return self.recording or self.preview

    def get_latest_audio(self, length: int) -> np.ndarray:
        return self._ring.get(length)

    def get_waveform_audio(self) -> np.ndarray:
        if self.recording:
            return self._get_recorded_concat()
        if self._last_recorded is not None:
            return self._last_recorded
        return self._ring.get(self.sample_rate)

    def get_waveform_sample_rate(self) -> float:
        if self.recording:
            return float(self.sample_rate) / float(self._record_vis_stride)
        return float(self.sample_rate)

    def get_bgm_data(self) -> Optional[np.ndarray]:
        if self._bgm_data is None:
            return None
        return self._bgm_data.copy()

    def get_preview_audio(self) -> np.ndarray:
        if self._last_bgm_chunk is not None:
            return self._last_bgm_chunk
        return self._ring.get(self.sample_rate)

    def _ensure_stream(self) -> None:
        if self.stream:
            return
        self.stream = sd.Stream(
            samplerate=self.sample_rate,
            channels=max(1, self.channels),
            blocksize=self.block_size,
            dtype="float32",
            device=self.device,
            callback=self._callback,
        )
        self.stream.start()

    def set_devices(self, input_device: Optional[int], output_device: Optional[int]) -> None:
        if input_device is None and output_device is None:
            self.device = None
        else:
            self.device = (input_device, output_device)
        if self.stream:
            was_active = self.recording or self.preview
            self.stream.stop()
            self.stream.close()
            self.stream = None
            if was_active:
                self._ensure_stream()

    def _stop_stream_if_idle(self) -> None:
        if self.recording or self.preview:
            return
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def _callback(self, indata, outdata, frames, time, status) -> None:
        if status:
            self.status.emit(str(status))
        mono_in = indata[:, 0] if indata.ndim > 1 else indata
        self._ring.push(mono_in)

        if self.recording:
            with self._record_lock:
                if self._record_file:
                    self._record_file.write(indata)
            if self._record_vis_stride > 1:
                vis = mono_in[:: self._record_vis_stride].copy()
            else:
                vis = mono_in.copy()
            self._recorded_chunks.append(vis)
            self._recorded_samples += len(vis)
            self._recorded_cache = None

        self._ensure_buffers(frames)
        out = self._out_buffer
        if self.preview or (self.recording and self.bgm_during_recording):
            self._fill_bgm_chunk(self._bgm_buffer)
            mix = self._bgm_buffer
            if self.bgm_gain != 1.0:
                np.multiply(mix, self.bgm_gain, out=mix)
            if self._overlay_enabled:
                self._fill_overlay_chunk(self._overlay_buffer)
                if self.overlay_gain != 1.0:
                    np.multiply(self._overlay_buffer, self.overlay_gain, out=self._overlay_buffer)
                np.add(mix, self._overlay_buffer, out=mix)
            if self._last_bgm_chunk is None or len(self._last_bgm_chunk) != frames:
                self._last_bgm_chunk = np.zeros(frames, dtype=np.float32)
            np.copyto(self._last_bgm_chunk, mix)
            for ch in range(self.channels):
                out[:, ch] += mix
        if self.monitor_gain > 0.0:
            gain = self.monitor_gain
            if self._monitor_buffer is None or len(self._monitor_buffer) != frames:
                self._monitor_buffer = np.zeros(frames, dtype=np.float32)
            np.multiply(mono_in, gain, out=self._monitor_buffer)
            for ch in range(self.channels):
                out[:, ch] += self._monitor_buffer
        outdata[:] = out

    def _ensure_buffers(self, frames: int) -> None:
        if self._out_buffer is None or self._out_buffer.shape != (frames, self.channels):
            self._out_buffer = np.zeros((frames, self.channels), dtype=np.float32)
        else:
            self._out_buffer.fill(0.0)
        if self._bgm_buffer is None or len(self._bgm_buffer) != frames:
            self._bgm_buffer = np.zeros(frames, dtype=np.float32)
        if self._overlay_buffer is None or len(self._overlay_buffer) != frames:
            self._overlay_buffer = np.zeros(frames, dtype=np.float32)
        if self._monitor_buffer is None or len(self._monitor_buffer) != frames:
            self._monitor_buffer = np.zeros(frames, dtype=np.float32)

    def _fill_bgm_chunk(self, buffer: np.ndarray) -> None:
        buffer.fill(0.0)
        if self._bgm_data is None:
            return
        frames = len(buffer)
        if self._bgm_pos < 0:
            advance = min(frames, -self._bgm_pos)
            self._bgm_pos += advance
            if advance < frames:
                self._read_bgm_into(buffer, advance, frames - advance)
            return
        self._read_bgm_into(buffer, 0, frames)

    def _read_bgm_into(self, buffer: np.ndarray, offset: int, frames: int) -> None:
        if self._bgm_data is None:
            return
        remaining = frames
        write_pos = offset
        while remaining > 0:
            start = self._bgm_pos
            if start >= len(self._bgm_data):
                if self._bgm_playlist:
                    self._bgm_index = (self._bgm_index + 1) % len(self._bgm_playlist)
                    self._bgm_data = self._bgm_playlist[self._bgm_index]
                    self._bgm_pos = 0
                    start = 0
                else:
                    buffer[write_pos:write_pos + remaining] = 0.0
                    self._bgm_pos = start + remaining
                    return
            available = len(self._bgm_data) - start
            take = min(available, remaining)
            buffer[write_pos:write_pos + take] = self._bgm_data[start:start + take]
            self._bgm_pos = start + take
            write_pos += take
            remaining -= take
            if remaining > 0 and not self._bgm_playlist:
                buffer[write_pos:write_pos + remaining] = 0.0
                self._bgm_pos += remaining
                return

    def _fill_overlay_chunk(self, buffer: np.ndarray) -> None:
        buffer.fill(0.0)
        if self._bgm_overlay is None:
            return
        frames = len(buffer)
        start = self._bgm_overlay_pos
        end = start + frames
        if start >= len(self._bgm_overlay):
            return
        available = len(self._bgm_overlay) - start
        take = min(available, frames)
        buffer[:take] = self._bgm_overlay[start:start + take]
        self._bgm_overlay_pos = end

    @staticmethod
    def _resample(data: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
        if sr_in == sr_out:
            return data
        ratio = sr_out / sr_in
        new_len = int(len(data) * ratio)
        x_old = np.linspace(0, 1, len(data))
        x_new = np.linspace(0, 1, new_len)
        return np.interp(x_new, x_old, data).astype(np.float32)

    def _get_recorded_concat(self) -> np.ndarray:
        if self._recorded_cache is not None and self._recorded_cache_samples == self._recorded_samples:
            return self._recorded_cache
        if not self._recorded_chunks:
            self._recorded_cache = np.array([], dtype=np.float32)
            self._recorded_cache_samples = 0
            return self._recorded_cache
        self._recorded_cache = np.concatenate(self._recorded_chunks)
        self._recorded_cache_samples = self._recorded_samples
        return self._recorded_cache
