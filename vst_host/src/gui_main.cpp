#include <JuceHeader.h>

using namespace juce;

struct GuiArgs
{
    String pluginPath;
    String presetPath;
    String savePath;
    int blockSize = 512;
};

static void showUsage()
{
    NativeMessageBox::showMessageBoxAsync(AlertWindow::WarningIcon,
                                          "UTAU VST Host GUI",
                                          "Usage:\n  utau_vst_host_gui --plugin <file> [--preset <file>] [--save <file>] [--block <size>]");
}

static bool parseArgs(const StringArray& args, GuiArgs& out)
{
    for (int i = 0; i < args.size(); ++i)
    {
        const auto arg = args[i];
        if (arg == "--plugin" && i + 1 < args.size())
        {
            out.pluginPath = args[++i];
            continue;
        }
        if (arg == "--preset" && i + 1 < args.size())
        {
            out.presetPath = args[++i];
            continue;
        }
        if (arg == "--save" && i + 1 < args.size())
        {
            out.savePath = args[++i];
            continue;
        }
        if (arg == "--block" && i + 1 < args.size())
        {
            out.blockSize = jmax(64, args[++i].getIntValue());
            continue;
        }
    }

    return out.pluginPath.isNotEmpty();
}

static bool applyPreset(AudioPluginInstance& instance, const File& presetFile)
{
    if (!presetFile.existsAsFile())
        return false;
    MemoryBlock data;
    if (!presetFile.loadFileAsData(data))
        return false;
    instance.setStateInformation(data.getData(), static_cast<int>(data.getSize()));
    return true;
}

static bool savePreset(AudioPluginInstance& instance, const File& presetFile)
{
    MemoryBlock data;
    instance.getStateInformation(data);
    return presetFile.replaceWithData(data.getData(), data.getSize());
}

static std::unique_ptr<AudioPluginInstance> loadPlugin(const File& file,
                                                       AudioPluginFormatManager& formatManager,
                                                       double sampleRate,
                                                       int blockSize,
                                                       String& error)
{
    OwnedArray<PluginDescription> types;
    for (auto* format : formatManager.getFormats())
    {
        if (!format->fileMightContainThisPluginType(file.getFullPathName()))
            continue;
        format->findAllTypesForFile(types, file.getFullPathName());
    }

    if (types.isEmpty())
    {
        error = "No plugin types found for " + file.getFullPathName();
        return nullptr;
    }

    PluginDescription desc = *types[0];
    auto instance = formatManager.createPluginInstance(desc, sampleRate, blockSize, error);
    if (!instance)
        return nullptr;

    auto layout = instance->getBusesLayout();
    if (layout.inputBuses.size() > 0)
        layout.inputBuses.getReference(0) = AudioChannelSet::stereo();
    if (layout.outputBuses.size() > 0)
        layout.outputBuses.getReference(0) = AudioChannelSet::stereo();
    instance->setBusesLayout(layout);
    instance->prepareToPlay(sampleRate, blockSize);
    instance->reset();

    return instance;
}

class HostComponent : public Component
{
public:
    HostComponent(AudioPluginInstance& processor, const File& presetFile)
        : processor(processor), presetPath(presetFile)
    {
        if (processor.hasEditor())
        {
            editor.reset(processor.createEditor());
            addAndMakeVisible(editor.get());
        }
        else
        {
            infoLabel.setText("Plugin has no editor", dontSendNotification);
            infoLabel.setJustificationType(Justification::centred);
            addAndMakeVisible(infoLabel);
        }

        saveButton.setButtonText("Save Preset");
        saveAsButton.setButtonText("Save As...");
        saveButton.onClick = [this]() { handleSave(); };
        saveAsButton.onClick = [this]() { handleSaveAs(); };
        addAndMakeVisible(saveButton);
        addAndMakeVisible(saveAsButton);

        setSize(getPreferredWidth(), getPreferredHeight());
    }

    int getPreferredWidth() const
    {
        if (editor)
            return editor->getWidth();
        return 640;
    }

    int getPreferredHeight() const
    {
        if (editor)
            return editor->getHeight() + 48;
        return 480;
    }

    void resized() override
    {
        auto area = getLocalBounds();
        auto buttonArea = area.removeFromBottom(40).reduced(8, 4);
        saveButton.setBounds(buttonArea.removeFromLeft(buttonArea.getWidth() / 2).reduced(4, 0));
        saveAsButton.setBounds(buttonArea.reduced(4, 0));

        if (editor)
            editor->setBounds(area);
        else
            infoLabel.setBounds(area);
    }

private:
    void handleSave()
    {
        if (presetPath == File())
        {
            handleSaveAs();
            return;
        }
        if (!savePreset(processor, presetPath))
            AlertWindow::showMessageBoxAsync(AlertWindow::WarningIcon, "UTAU VST Host GUI", "Failed to save preset.");
    }

    void handleSaveAs()
    {
        fileChooser = std::make_unique<FileChooser>(
            "Save preset",
            presetPath,
            "*.vstpreset;*.fxp;*.fxb;*.bin"
        );
        auto flags = FileBrowserComponent::saveMode | FileBrowserComponent::canSelectFiles;
        fileChooser->launchAsync(flags, [this](const FileChooser& chooser)
        {
            const auto result = chooser.getResult();
            fileChooser.reset();
            if (result == File())
                return;
            presetPath = result;
            if (!savePreset(processor, presetPath))
                AlertWindow::showMessageBoxAsync(AlertWindow::WarningIcon, "UTAU VST Host GUI", "Failed to save preset.");
        });
    }

    AudioPluginInstance& processor;
    std::unique_ptr<AudioProcessorEditor> editor;
    TextButton saveButton;
    TextButton saveAsButton;
    Label infoLabel;
    File presetPath;
    std::unique_ptr<FileChooser> fileChooser;
};

class HostWindow : public DocumentWindow
{
public:
    HostWindow(const String& name, std::unique_ptr<Component> content)
        : DocumentWindow(name, Colours::darkgrey, DocumentWindow::allButtons)
    {
        setUsingNativeTitleBar(true);
        setContentOwned(content.release(), true);
        centreWithSize(getWidth(), getHeight());
        setVisible(true);
    }

    void closeButtonPressed() override
    {
        JUCEApplication::getInstance()->systemRequestedQuit();
    }
};

class VstHostGuiApp : public JUCEApplication
{
public:
    const String getApplicationName() override { return "UTAU VST Host GUI"; }
    const String getApplicationVersion() override { return "0.1.0"; }

    void initialise(const String& commandLine) override
    {
        StringArray args;
        args.addTokens(commandLine, true);
        args.removeEmptyStrings();

        GuiArgs parsed;
        if (!parseArgs(args, parsed))
            return fail("Invalid arguments.", 2, true);

        File pluginFile(parsed.pluginPath);
        if (!pluginFile.exists())
            return fail("Plugin file not found.", 3);

        AudioPluginFormatManager formats;
#if JUCE_PLUGINHOST_VST3
        formats.addFormat(new VST3PluginFormat());
#endif
#if JUCE_PLUGINHOST_VST
        formats.addFormat(new VSTPluginFormat());
#endif
#if JUCE_PLUGINHOST_LV2 && JUCE_LINUX
        formats.addFormat(new LV2PluginFormat());
#endif
#if JUCE_PLUGINHOST_LADSPA && JUCE_LINUX
        formats.addFormat(new LADSPAPluginFormat());
#endif
#if JUCE_PLUGINHOST_AU && JUCE_MAC
        formats.addFormat(new AudioUnitPluginFormat());
#endif

        String error;
        instance = loadPlugin(pluginFile, formats, 44100.0, parsed.blockSize, error);
        if (!instance)
            return fail(error.isNotEmpty() ? error : "Failed to load plugin.", 4);

        File presetFile(parsed.presetPath);
        if (presetFile.existsAsFile())
            applyPreset(*instance, presetFile);

        File saveFile(parsed.savePath);
        auto content = std::make_unique<HostComponent>(*instance, saveFile);
        window = std::make_unique<HostWindow>(pluginFile.getFileName(), std::move(content));
    }

    void shutdown() override
    {
        window.reset();
        if (instance)
        {
            instance->releaseResources();
            instance.reset();
        }
    }

private:
    void fail(const String& message, int code, bool showUsageHint = false)
    {
        setApplicationReturnValue(code);
        String fullMessage = message;
        if (showUsageHint)
            fullMessage << "\n\nUse --plugin <file> to open a plugin.";
        NativeMessageBox::showMessageBoxAsync(
            AlertWindow::WarningIcon,
            "UTAU VST Host GUI",
            fullMessage,
            nullptr,
            ModalCallbackFunction::create([this](int) { quit(); })
        );
    }

    std::unique_ptr<AudioPluginInstance> instance;
    std::unique_ptr<HostWindow> window;
};

START_JUCE_APPLICATION(VstHostGuiApp)
