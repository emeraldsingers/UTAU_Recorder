cmake_minimum_required(VERSION 3.22)
project(utau_vst_host VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT DEFINED JUCE_DIR)
  message(FATAL_ERROR "Set JUCE_DIR to the path of your JUCE folder.")
endif()

add_subdirectory(${JUCE_DIR} juce)

option(ENABLE_VST2 "Enable VST2 hosting (requires VST2 SDK)" OFF)
set(VST2_SDK_PATH "" CACHE PATH "Path to VST2 SDK root (optional)")

juce_add_console_app(utau_vst_host
  PRODUCT_NAME "UTAU VST Host"
)

target_sources(utau_vst_host PRIVATE
  src/main.cpp
)

juce_generate_juce_header(utau_vst_host)

target_compile_definitions(utau_vst_host PRIVATE
  JUCE_PLUGINHOST_VST3=1
  JUCE_PLUGINHOST_AU=1
  JUCE_PLUGINHOST_LADSPA=1
  JUCE_PLUGINHOST_LV2=1
  JUCE_USE_CURL=0
)

if (ENABLE_VST2)
  target_compile_definitions(utau_vst_host PRIVATE JUCE_PLUGINHOST_VST=1)
  if (VST2_SDK_PATH)
    target_compile_definitions(utau_vst_host PRIVATE JUCE_VST2_SDK_LOCATION="${VST2_SDK_PATH}")
  endif()
endif()

target_link_libraries(utau_vst_host PRIVATE
  juce::juce_core
  juce::juce_audio_basics
  juce::juce_audio_formats
  juce::juce_audio_processors
)

juce_add_gui_app(utau_vst_host_gui
  PRODUCT_NAME "UTAU VST Host GUI"
)

target_sources(utau_vst_host_gui PRIVATE
  src/gui_main.cpp
)

juce_generate_juce_header(utau_vst_host_gui)

target_compile_definitions(utau_vst_host_gui PRIVATE
  JUCE_PLUGINHOST_VST3=1
  JUCE_PLUGINHOST_AU=1
  JUCE_PLUGINHOST_LADSPA=1
  JUCE_PLUGINHOST_LV2=1
  JUCE_USE_CURL=0
)

if (ENABLE_VST2)
  target_compile_definitions(utau_vst_host_gui PRIVATE JUCE_PLUGINHOST_VST=1)
  if (VST2_SDK_PATH)
    target_compile_definitions(utau_vst_host_gui PRIVATE JUCE_VST2_SDK_LOCATION="${VST2_SDK_PATH}")
  endif()
endif()

target_link_libraries(utau_vst_host_gui PRIVATE
  juce::juce_core
  juce::juce_audio_basics
  juce::juce_audio_formats
  juce::juce_audio_processors
  juce::juce_gui_basics
  juce::juce_gui_extra
)
