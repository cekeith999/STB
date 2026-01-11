# Speech To Blender v0.5.0 - MVP

Voice-controlled 3D modeling for Blender. Control Blender with natural language voice commands powered by AI.

## 🚀 Quick Start

### Installation

1. **Download** `SpeechToBlender_v0.5.0_MVP.zip`
2. **Open Blender** → Edit → Preferences → Add-ons
3. **Click "Install..."** → Select the ZIP file
4. **Enable** the add-on (check the box next to "Speech To Blender")

### First-Time Setup

#### 1. Get API Keys

You'll need at least one AI API key:

**Option A: OpenAI (GPT-4o) - Recommended**
- Get key from: https://platform.openai.com/api-keys
- Create a new key or use an existing one
- Paste into Blender preferences (split into Part 1 and Part 2 if the key is long)

**Option B: Google Gemini**
- Get key from: https://aistudio.google.com/app/apikey
- Create a new API key
- Paste into Blender preferences

#### 2. Configure Add-on

1. Open **Blender Preferences** → **Add-ons** → **Speech To Blender**
2. **Enter your API key(s)** in the "Voice & AI Settings" section
3. **Select your preferred AI model** (GPT-4o or Gemini)
4. **(Optional)** Set a "Modeling Target" for better context (e.g., "Coffee Mug", "Smartphone")

#### 3. Start Voice Mode

1. In Blender, open the **STB** panel (right sidebar, "STB" tab)
2. Click **"Start RPC"** - a console window will open
3. Wait for **"Voice: Running"** status (green checkmark)
4. Press **Alt+F** to toggle voice listening ON/OFF
   - 🟢 **ON**: Listening for voice commands
   - 🟡 **OFF**: Paused (won't process audio)

## 🎤 Usage

### Basic Commands

1. **Enable listening**: Press **Alt+F** (or use the toggle in UI)
2. **Speak your command**: 
   - "Add a cube"
   - "Extrude the selected face"
   - "Create a sphere with radius 2"
   - "Delete the selected object"
3. **Wait for execution**: The console shows transcription and commands
4. **Check results**: Your command should execute in Blender

### Advanced Features

**ReAct Reasoning Mode** (for complex multi-step commands):
- Enable in Preferences → "Use ReAct Reasoning"
- Allows the AI to break down complex tasks into steps
- Uses more API credits but handles complex requests better

**Modeling Target**:
- Set in Preferences → "Modeling Target"
- Gives AI context about what you're building
- Examples: "Echo Dot", "Coffee Mug", "Character Head"

## 📦 What's Included

- ✅ **Bundled Python runtime** (no Python installation needed)
- ✅ **Whisper speech-to-text** (local, no cloud, privacy-preserving)
- ✅ **OpenAI GPT-4o support** (high-quality command understanding)
- ✅ **Google Gemini support** (alternative AI model)
- ✅ **ReAct reasoning mode** (for complex commands)
- ✅ **Mesh analysis** (AI understands your 3D scene)
- ✅ **Screenshot context** (AI can see your viewport)

## 🔧 Troubleshooting

### Voice script won't start
- **Check the console window** for error messages
- **Verify API keys** are set correctly in preferences
- **Ensure Blender is not blocking** the script execution
- **Check Windows Firewall** isn't blocking localhost connections

### Commands not executing
- **Check RPC status** in STB panel (should show "Running")
- **Verify voice listening is ON** (Alt+F, should show 🟢)
- **Check console** for error messages
- **Try a simple command first** ("add a cube") to test

### API errors
- **Verify your API key** is correct and not expired
- **Check your API account** has credits/quota available
- **Try switching** to a different model provider (GPT ↔ Gemini)
- **Check internet connection** (API calls require internet)

### Transcription issues
- **Speak clearly** and wait for silence detection
- **Check microphone** is working in Windows settings
- **Reduce background noise** for better accuracy
- **Wait for silence** after speaking (system detects end of speech)

## 📋 System Requirements

- **Blender**: 3.6.0 or later (tested on 4.3)
- **OS**: Windows 10/11 (64-bit)
- **RAM**: 4GB minimum (8GB recommended)
- **Internet**: Required for AI model API calls
- **Microphone**: Any working microphone

## 🎯 Supported Commands

### Basic Operations
- Add primitives (cube, sphere, cylinder, etc.)
- Transform objects (move, rotate, scale)
- Delete objects
- Select objects

### Mesh Operations
- Extrude faces/edges/vertices
- Bevel edges
- Subdivide
- Loop cuts
- Merge vertices

### Scene Operations
- Import models (GLB, FBX, OBJ)
- Set camera
- Create materials (basic)
- Render (basic)

### Complex Operations (with ReAct)
- Multi-step modeling workflows
- Context-aware operations
- Geometry analysis and modification

## 🔒 Privacy & Security

- **Local speech-to-text**: Audio never leaves your computer
- **API keys stored locally**: Only in Blender preferences (encrypted by Blender)
- **No telemetry**: No data collection or tracking
- **Localhost only**: RPC server only accessible on your machine
- **Safety gate**: Dangerous operations are blocked

## 📚 Technical Details

- **Speech-to-Text**: faster-whisper (local transformer model)
- **Command Understanding**: GPT-4o or Gemini (cloud API)
- **Communication**: XML-RPC over localhost (port 8765)
- **Threading**: Safe multi-threaded execution

## 🐛 Known Limitations

- **Windows only** (currently)
- **English language** (speech-to-text)
- **Requires internet** for AI model API calls
- **API costs**: Uses OpenAI/Gemini API credits
- **Complex operations** may require ReAct mode

## 📞 Support

For issues, questions, or feedback:
- Check the console window for detailed error messages
- Review this README for common solutions
- Contact: [Your contact info here]

## 📄 License

[Your license info here]

## 🙏 Credits

- **Whisper**: OpenAI's speech recognition model
- **GPT-4o**: OpenAI's language model
- **Gemini**: Google's language model
- **Blender**: 3D creation suite

---

**Version**: 0.5.0  
**Last Updated**: January 2025

