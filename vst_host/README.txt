UTAU VST Host (CLI)

This is a minimal offline VST host used by the Python UI.
It processes audio files with a plugin chain and writes the result.

Requirements
- JUCE (for VST3 / AU / LV2 / LADSPA hosting)
- CMake 3.22+
- C++17 compiler

Build (with JUCE)
1) Configure:
   cmake -S . -B build -DJUCE_DIR=PATH_TO_JUCE

2) Build:
   cmake --build build --config Release

Where to find the binary
- Windows: build/Release/utau_vst_host.exe
- macOS/Linux: build/utau_vst_host (or build/Release/utau_vst_host if multi-config)

Example (Windows PowerShell)
  cmake -S . -B build -DJUCE_DIR=C:\path\to\JUCE
  cmake --build build --config Release

Example (macOS/Linux)
  cmake -S . -B build -DJUCE_DIR=/path/to/JUCE
  cmake --build build -j

VST2 (optional)
- Requires the legacy VST2 SDK. Enable with:
  cmake -S . -B build -DJUCE_DIR=PATH_TO_JUCE -DENABLE_VST2=ON -DVST2_SDK_PATH=PATH_TO_VST2_SDK

Usage
  utau_vst_host --input <file> --output <file> --chain <json> [--block <size>]

GUI host (for editing plugin UI / saving presets)
  utau_vst_host_gui --plugin <file> [--preset <file>] [--save <file>] [--block <size>]

Chain JSON format
{
  "version": 1,
  "plugins": [
    {
      "path": "C:/Plugins/MyPlugin.vst3",
      "preset": "C:/Presets/MyPreset.vstpreset",
      "bypass": false
    }
  ]
}

Notes
- Preset loading uses AudioProcessor::setStateInformation. Some plugin preset formats may not load.
- The host does offline processing only. No realtime/GUI.
