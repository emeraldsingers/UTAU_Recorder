#include <JuceHeader.h>

using namespace juce;

struct ChainSlot
{
    String pluginPath;
    String presetPath;
    bool bypass = false;
};

struct Args
{
    String input;
    String output;
    String chain;
    int blockSize = 512;
};

static void printUsage()
{
    std::cout << "UTAU VST Host\n"
              << "Usage: utau_vst_host --input <file> --output <file> --chain <json> [--block <size>]\n";
}

static bool parseArgs(const StringArray& argv, Args& args)
{
    for (int i = 1; i < argv.size(); ++i)
    {
        String arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            printUsage();
            return false;
        }
        if (arg == "--input" && i + 1 < argv.size())
        {
            args.input = argv[++i];
            continue;
        }
        if (arg == "--output" && i + 1 < argv.size())
        {
            args.output = argv[++i];
            continue;
        }
        if (arg == "--chain" && i + 1 < argv.size())
        {
            args.chain = argv[++i];
            continue;
        }
        if (arg == "--block" && i + 1 < argv.size())
        {
            args.blockSize = jmax(64, String(argv[++i]).getIntValue());
            continue;
        }
    }

    return args.input.isNotEmpty() && args.output.isNotEmpty() && args.chain.isNotEmpty();
}

static String resolvePath(const File& baseDir, const String& path)
{
    String cleaned = path.trim();
    if (cleaned.length() >= 2 && (cleaned.startsWithChar('\"') && cleaned.endsWithChar('\"')))
        cleaned = cleaned.substring(1, cleaned.length() - 1).trim();
    if (cleaned.length() >= 2 && (cleaned.startsWithChar('\'') && cleaned.endsWithChar('\'')))
        cleaned = cleaned.substring(1, cleaned.length() - 1).trim();

    if (File::isAbsolutePath(cleaned))
        return File(cleaned).getFullPathName();
    return baseDir.getChildFile(cleaned).getFullPathName();
}

static bool loadChain(const File& chainFile, Array<ChainSlot>& slots, String& error)
{
    const auto jsonText = chainFile.loadFileAsString();
    var data = JSON::parse(jsonText);
    if (data.isVoid() || !data.isObject())
    {
        error = "Chain JSON is invalid.";
        return false;
    }

    var pluginsVar = data.getProperty("plugins", var());
    if (!pluginsVar.isArray())
    {
        error = "Chain JSON missing 'plugins' array.";
        return false;
    }

    auto* plugins = pluginsVar.getArray();
    if (plugins == nullptr || plugins->isEmpty())
    {
        error = "Chain JSON has no plugins.";
        return false;
    }

    const File baseDir = chainFile.getParentDirectory();
    for (const auto& entry : *plugins)
    {
        if (!entry.isObject())
            continue;
        auto* obj = entry.getDynamicObject();
        if (obj == nullptr)
            continue;

        ChainSlot slot;
        const auto rawPlugin = obj->getProperty("path").toString();
        const auto rawPreset = obj->getProperty("preset").toString();
        if (rawPlugin.isNotEmpty())
            slot.pluginPath = resolvePath(baseDir, rawPlugin);
        if (rawPreset.isNotEmpty())
            slot.presetPath = resolvePath(baseDir, rawPreset);
        slot.bypass = static_cast<bool>(obj->getProperty("bypass"));
        if (slot.pluginPath.isNotEmpty())
            slots.add(slot);
    }

    if (slots.isEmpty())
    {
        error = "Chain JSON has no valid plugin paths.";
        return false;
    }

    return true;
}

static std::unique_ptr<AudioPluginInstance> loadPlugin(const File& file,
                                                       AudioPluginFormatManager& formatManager,
                                                       double sampleRate,
                                                       int blockSize,
                                                       int numChannels,
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

    instance->setNonRealtime(true);
    auto layout = instance->getBusesLayout();
    if (layout.inputBuses.size() > 0)
        layout.inputBuses.getReference(0) = AudioChannelSet::canonicalChannelSet(numChannels);
    if (layout.outputBuses.size() > 0)
        layout.outputBuses.getReference(0) = AudioChannelSet::canonicalChannelSet(numChannels);
    instance->setBusesLayout(layout);
    instance->prepareToPlay(sampleRate, blockSize);
    instance->reset();

    return instance;
}

static bool applyPreset(AudioPluginInstance& instance, const String& presetPath)
{
    if (presetPath.isEmpty())
        return true;
    File presetFile(presetPath);
    if (!presetFile.existsAsFile())
        return false;

    MemoryBlock data;
    if (!presetFile.loadFileAsData(data))
        return false;

    instance.setStateInformation(data.getData(), static_cast<int>(data.getSize()));
    return true;
}

#if JUCE_WINDOWS
int wmain(int argc, wchar_t* argv[])
#else
int main(int argc, char* argv[])
#endif
{
    ScopedJuceInitialiser_GUI juceInit;

    StringArray argvStrings;
#if JUCE_WINDOWS
    for (int i = 0; i < argc; ++i)
        argvStrings.add(String(CharPointer_UTF16(argv[i])));
#else
    for (int i = 0; i < argc; ++i)
        argvStrings.add(String(argv[i]));
#endif

    Args args;
    if (!parseArgs(argvStrings, args))
    {
        printUsage();
        return 1;
    }

    File inputFile(args.input);
    File outputFile(args.output);
    File chainFile(args.chain);

    if (!inputFile.existsAsFile())
    {
        std::cerr << "Input file not found: " << inputFile.getFullPathName() << "\n";
        return 2;
    }

    if (!chainFile.existsAsFile())
    {
        std::cerr << "Chain file not found: " << chainFile.getFullPathName() << "\n";
        return 2;
    }

    AudioFormatManager formatManager;
    formatManager.registerBasicFormats();

    std::unique_ptr<AudioFormatReader> reader(formatManager.createReaderFor(inputFile));
    if (!reader)
    {
        std::cerr << "Failed to read input file.\n";
        return 3;
    }

    Array<ChainSlot> slots;
    String error;
    if (!loadChain(chainFile, slots, error))
    {
        std::cerr << error << "\n";
        return 4;
    }

    AudioPluginFormatManager pluginFormats;
#if JUCE_PLUGINHOST_VST3
    pluginFormats.addFormat(new VST3PluginFormat());
#endif
#if JUCE_PLUGINHOST_VST
    pluginFormats.addFormat(new VSTPluginFormat());
#endif
#if JUCE_PLUGINHOST_LV2 && JUCE_LINUX
    pluginFormats.addFormat(new LV2PluginFormat());
#endif
#if JUCE_PLUGINHOST_LADSPA && JUCE_LINUX
    pluginFormats.addFormat(new LADSPAPluginFormat());
#endif
#if JUCE_PLUGINHOST_AU && JUCE_MAC
    pluginFormats.addFormat(new AudioUnitPluginFormat());
#endif

    OwnedArray<AudioPluginInstance> instances;
    for (const auto& slot : slots)
    {
        if (slot.bypass)
            continue;
        File pluginFile(slot.pluginPath);
        String pluginError;
        auto instance = loadPlugin(
            pluginFile,
            pluginFormats,
            reader->sampleRate,
            args.blockSize,
            static_cast<int>(reader->numChannels),
            pluginError
        );
        if (!instance)
        {
            std::cerr << "Failed to load plugin: " << pluginFile.getFullPathName() << "\n";
            if (pluginError.isNotEmpty())
                std::cerr << pluginError << "\n";
            return 5;
        }
        if (slot.presetPath.isNotEmpty())
        {
            if (!applyPreset(*instance, slot.presetPath))
                std::cerr << "Warning: failed to apply preset " << slot.presetPath << "\n";
        }
        instances.add(instance.release());
    }

    const int numChannels = static_cast<int>(reader->numChannels);
    const int64 totalSamples = reader->lengthInSamples;

    outputFile.getParentDirectory().createDirectory();
    std::unique_ptr<FileOutputStream> outStream(outputFile.createOutputStream());
    if (!outStream)
    {
        std::cerr << "Failed to create output file.\n";
        return 6;
    }

    auto ext = outputFile.getFileExtension().toLowerCase();
    if (ext.startsWithChar('.'))
        ext = ext.substring(1);
    AudioFormat* outputFormat = formatManager.findFormatForFileExtension(ext);
    if (outputFormat == nullptr)
        outputFormat = formatManager.findFormatForFileExtension("wav");

    if (outputFormat == nullptr)
    {
        std::cerr << "No audio writer available for output file.\n";
        return 6;
    }

    std::unique_ptr<AudioFormatWriter> writer(outputFormat->createWriterFor(
        outStream.get(), reader->sampleRate, static_cast<unsigned int>(numChannels), 16, {}, 0));

    if (!writer)
    {
        std::cerr << "Failed to create output writer.\n";
        return 6;
    }

    outStream.release();

    AudioBuffer<float> buffer(numChannels, args.blockSize);
    MidiBuffer midi;
    int64 position = 0;

    while (position < totalSamples)
    {
        const int block = static_cast<int>(jmin<int64>(args.blockSize, totalSamples - position));
        buffer.clear();
        reader->read(&buffer, 0, block, position, true, true);
        midi.clear();
        for (auto* instance : instances)
            instance->processBlock(buffer, midi);
        writer->writeFromAudioSampleBuffer(buffer, 0, block);
        position += block;
    }

    return 0;
}
