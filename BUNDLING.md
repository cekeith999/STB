# Bundling Speech To Blender Add-on

## Included Dependencies

The add-on includes a bundled Python runtime (`stb_runtime/python/`) with the following packages pre-installed:

- **openai** (>=2.7.0) - For GPT fallback in voice commands
- **faster-whisper** - For speech-to-text transcription
- **numpy**, **onnxruntime** - Required by faster-whisper
- Other dependencies for voice processing

## Distribution

When distributing the add-on, **include the entire `stb_runtime/` folder**. This ensures:

1. ✅ All dependencies are available without user installation
2. ✅ Works on systems without Python installed
3. ✅ Consistent behavior across different systems
4. ✅ Auto-install fallback if package is missing

## Auto-Install Fallback

If the `openai` package is missing from the bundled Python, the add-on will automatically attempt to install it on first run (during `register()`). This handles edge cases where:

- The bundled Python exists but `openai` wasn't included
- The package was accidentally removed
- A fresh install needs dependencies

## Manual Installation (if needed)

If auto-install fails, users can manually install:

```powershell
# Windows PowerShell
& "path\to\addon\stb_runtime\python\python.exe" -m pip install openai
```

Or use the requirements.txt:

```powershell
& "path\to\addon\stb_runtime\python\python.exe" -m pip install -r requirements.txt
```

## File Structure

```
SpeechToBlender/
├── __init__.py              # Main add-on entry point
├── voice_to_blender.py      # Voice command script
├── requirements.txt         # Python dependencies list
├── stb_runtime/             # Bundled Python runtime
│   └── python/
│       └── Lib/
│           └── site-packages/
│               ├── openai/  # ✅ Pre-installed
│               └── ...      # Other dependencies
└── ...
```

