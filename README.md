# UTAU Voicebank Recorder (MVP)

Desktop recorder for UTAU voicebank sessions built with Python 3.11+ and PyQt6.
It supports real recording workflows: sessions, reclists, voicebank import, BGM preview/recording, live analysis, and per‑item tasks.

## Features
- Session management (save/open/autosave) with JSON state
- Reclist import (dedupe, replace or append)
- Voicebank import (multi‑bank support, oto.ini aware, prefix/suffix handling)
- BGM preview and BGM during recording
- BGM overlay note (clean reference tone)
- Crash‑safe recording (temp file → atomic rename)
- Portable project: copies reclist and BGM into session folder
- Live waveform/spectrum/power/F0 + recorded F0 & mel spectrogram
- Click waveform to play with playhead
- Romaji auto‑generation (mora segmented)
- Multi‑language UI (English / Русский / 日本語)
- Event log for session actions
- Optional VST batch processing with plugin chains, presets, and backups

## Requirements
- Python 3.11+
- PortAudio (required by sounddevice)

Python packages (see `requirements.txt`):
- PyQt6
- numpy
- sounddevice
- soundfile
- pyqtgraph

Optional (VST batch processing host):
- JUCE (C++), used by the offline VST hosts in `vst_host/`

## Install
```bash
pip install -r requirements.txt
```

## Run
```bash
python main.py
```

## VST Batch Processing (Optional)
1) Build the VST hosts from `vst_host/` (see `vst_host/README.txt`).
2) Copy binaries to `tools/`:
   - `tools/utau_vst_host(.exe)` (CLI)
   - `tools/utau_vst_host_gui(.exe)` (GUI)
3) Set paths in **Settings -> VST Tools**.
4) Open **Tools -> Apply VST Plugins...** and build your chain.
5) Use **Open plugin UI** to edit a plugin and save its preset.
6) Set **Parallel jobs** for batch speed (more threads = faster, up to CPU limits).

Backups are created next to each recording folder:
`_backup_vst_YYYYMMDD_HHMMSS`


## Tests
```bash
python -m unittest tests/test_parsers.py
```

## Project Structure
```
app/        UI and controllers
audio/      audio engine and DSP
models/     session models and parsers
storage/    save/load utilities
resources/  assets
main.py     entry point
```

## Workflow
1) **File → New Session**
   - If path is empty, a default folder is created: `recordings/<singer>/<session>`
   - Output prefix/suffix can be set at creation

2) **Import**
   - **Reclist**: choose replace or append; duplicates are ignored
   - **Voicebank**: supports multiple banks; optionally use samples as BGM per alias
   - **BGM WAV**: copied into session folder for portability
   - **Generate BGM Note**: choose replace or overlay

3) **Record**
   - Select item → Record / Stop / Re‑record
   - BGM can play during recording

4) **Review**
   - Click waveform to play, playhead shows position
   - Selecting an item shows its waveform & analysis

## Session Folder (Portable)
```
<session>/
  session.json
  reclist.txt
  Recordings/
  BGM/
  event_log.txt
```

## Notes
- If `sounddevice` can’t find devices, ensure PortAudio is installed and audio drivers are working.
- Voicebank BGM mapping uses `oto.ini` when present, and falls back to WAV filenames.
- Overlay tone is a clean sine (no vibrato).
- The optional VST hosts are implemented with JUCE and must be built separately (see `vst_host/README.txt`).

## License
MIT
