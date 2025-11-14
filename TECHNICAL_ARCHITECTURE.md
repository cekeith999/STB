# Speech To Blender: Technical Architecture Documentation

## Overview

Speech To Blender is a voice-controlled Blender add-on that enables natural language interaction with Blender's 3D modeling capabilities. The system uses a multi-tier architecture with local processing, cloud AI services, and an RPC bridge for communication.

**Last Updated**: November 7, 2025

---

## System Architecture

### High-Level Flow

```
User Voice Input
    ↓
Microphone (sounddevice)
    ↓
Audio Recording (until silence detected)
    ↓
Speech-to-Text (faster-whisper via whisper-cli)
    ↓
Text Transcript
    ↓
Command Parsing (3-tier system)
    ├─→ Tier 1: I/O Rules (import/export)
    ├─→ Tier 2: Local Rules (regex patterns)
    └─→ Tier 3: GPT-4o (natural language understanding)
    ↓
Blender Operation JSON
    ↓
XML-RPC Bridge (localhost:8765)
    ↓
Safety Gate Validation
    ↓
Task Queue (_TASKQ)
    ↓
Blender Main Thread Execution (bpy.app.timers)
    ↓
Blender Scene Modification
```

---

## Core Components

### 1. Voice Script (`voice_to_blender.py`)

**Location**: `SpeechToBlender/voice_to_blender.py`

**Purpose**: Main voice processing script that runs as a separate Python process outside Blender.

**Responsibilities**:
- Audio capture from microphone
- Speech-to-text transcription
- Command parsing and intent recognition
- GPT-4o API integration for complex commands
- Communication with Blender via XML-RPC

**Key Functions**:
- `record_until_silence()`: Captures audio until silence is detected
- `transcribe(wav_path)`: Converts audio to text using whisper-cli
- `try_io_rules(text)`: Handles import/export commands
- `try_local_rules(text)`: Regex-based pattern matching for common commands
- `gpt_to_json(transcript)`: GPT-4o fallback for complex natural language
- `send_to_blender(cmd)`: Sends commands to Blender via RPC

**Dependencies**:
- `numpy`: Audio processing (RMS calculation)
- `sounddevice`: Microphone input/output
- `xmlrpc.client`: RPC communication with Blender
- `openai`: GPT-4o API client
- `subprocess`: Executes whisper-cli for transcription

**Configuration**:
- `WHISPER_MODEL`: "small" (faster-whisper model name)
- `WHISPER_LANG`: "en" (English)
- `ENABLE_GPT_FALLBACK`: True (enables GPT-4o fallback)
- `RPC_URL`: "http://127.0.0.1:8765/RPC2"

---

### 2. Blender Add-on (`__init__.py`)

**Location**: `SpeechToBlender/__init__.py`

**Purpose**: Blender add-on that hosts the RPC server and executes commands safely.

**Responsibilities**:
- XML-RPC server hosting (localhost:8765)
- Safety gate for operator validation
- Task queue management for thread-safe execution
- Voice script process management
- UI panels and preferences
- Undo grouping for voice commands

**Key Components**:

#### RPC Server
- **Host**: 127.0.0.1
- **Port**: 8765
- **Protocol**: XML-RPC
- **Thread**: Background thread (`_SERVER_THREAD`)
- **Methods Exposed**:
  - `enqueue_op_safe(op, kwargs)`: Queue a Blender operation
  - `get_openai_api_key()`: Retrieve API key from preferences
  - `start_voice_command()`: Mark beginning of voice command (undo grouping)
  - `get_super_mode_state()`: Get Super Mode status and target object

#### Safety Gate (`_is_safe_op`, `_safe_call_operator`)
- **Blacklist**: Blocks dangerous operators (file.quit, addon_disable, etc.)
- **Whitelist**: Allows safe operator prefixes (mesh.*, object.*, transform.*, etc.)
- **Auto-enable**: Automatically enables required import add-ons (io_mesh_stl, etc.)

#### Task Queue (`_TASKQ`)
- **Type**: `queue.Queue` (thread-safe)
- **Purpose**: Queue operations from RPC thread to Blender's main thread
- **Drain**: `bpy.app.timers` calls `_drain_task_queue()` periodically

#### Voice Process Management
- **Script**: `voice_to_blender.py`
- **Wrapper**: `voice_to_blender.bat` (Windows batch file to keep console open)
- **State**: `_VOICE_POPEN`, `_VOICE_RUNNING`

**Dependencies**:
- `bpy`: Blender Python API
- `xmlrpc.server`: RPC server implementation
- `threading`: Background server thread
- `queue`: Thread-safe task queue
- `subprocess`: Voice script process management

---

### 3. STB Core (`stb_core/`)

**Location**: `SpeechToBlender/stb_core/`

**Purpose**: Core functionality for providers, commands, and pipeline management.

#### Providers (`stb_core/providers/`)

**Meshy Provider** (`meshy.py`):
- **Purpose**: Integration with Meshy.ai API for text-to-3D generation
- **Functionality**:
  - Submit text-to-3D generation jobs
  - Poll job status
  - Download generated models
  - Import into Blender automatically
- **API**: Meshy.ai REST API
- **Formats**: GLB, FBX, OBJ

**Base Provider** (`base.py`):
- Abstract base class for all providers
- Defines provider interface

**Mock Provider** (`mock.py`):
- Testing provider for development
- Returns mock data without API calls

**Registry** (`registry.py`):
- Provider registration system
- Allows dynamic provider loading

#### Commands (`stb_core/commands/`)

**Safety** (`safety.py`):
- Command safety validation
- Operator whitelist/blacklist

**Schema** (`schema.py`):
- Command schema definitions
- Type validation

#### Pipeline (`pipeline.py`):
- Orchestrates provider workflows
- Handles async operations

#### Config (`config.py`):
- Configuration management
- Provider settings

---

## AI Models & Services

### 1. faster-whisper (Speech-to-Text)

**Type**: Local AI Model (Transformer-based)

**Implementation**: `whisper-cli.exe` (C++ implementation via whisper.cpp)

**Model**: "small" (default)
- **Size**: ~500MB
- **Accuracy**: ~95% for clear English speech
- **Speed**: ~200-500ms transcription time
- **Language**: English (configurable)

**Location**: `stb_runtime/whisper/whisper-cli.exe`

**Usage**:
```python
subprocess.run([WHISPER_CLI, "--model", "small", "--language", "en", "--input", wav_path])
```

**Output Format**: JSON
```json
{"text": "transcribed text here"}
```

**Technical Details**:
- Based on OpenAI's Whisper architecture
- Uses C++ implementation (whisper.cpp) for performance
- Runs locally (no cloud dependency)
- No API costs
- Privacy-preserving (audio never leaves local machine)

**Transformer Architecture**:
- Encoder-decoder transformer
- Trained on Common Voice, LibriSpeech, and other public datasets
- Converts audio spectrograms to text tokens

---

### 2. GPT-4o (Natural Language Understanding)

**Type**: Cloud AI Service (OpenAI API)

**Model**: `gpt-4o`

**Purpose**: 
- Natural language command understanding
- Complex operation selection
- Multi-step command parsing
- Context-aware operation mapping

**API Client**: `openai` Python package (>=2.7.0)

**Usage**:
```python
from openai import OpenAI
client = OpenAI(api_key=api_key)
resp = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0  # Deterministic output
)
```

**System Prompt**:
```
You are a Blender automation agent.
Output ONLY raw JSON (no prose, no code fences).
Each command must be of the form: {"op":"<module.op>","kwargs":{}}.
If multiple steps are implied, output a JSON array of such dicts.
Prefer creative operators (object/mesh/curve/transform/material/node/render).
Never use file/quit/addon/script/image.save operators.
```

**Output Format**: JSON
```json
{"op": "mesh.primitive_cube_add", "kwargs": {"size": 2}}
```

**API Key Management**:
- Stored in Blender preferences (split into two parts due to StringProperty limit)
- Cached in voice script to avoid repeated RPC calls
- Fallback to environment variable `OPENAI_API_KEY`

**Cost**: Pay-per-use (OpenAI API pricing)

**Latency**: ~1-3 seconds per request

---

## Databases & Data Storage

### Current Status: **No Databases Used**

The system does not currently use any databases. All data is stored in:

1. **Blender Preferences** (`bpy.context.preferences.addons["SpeechToBlender"]`):
   - OpenAI API key (split into two parts)
   - Meshy API key
   - Super Mode target object
   - Other user preferences

2. **In-Memory State**:
   - Task queue (`_TASKQ`)
   - RPC server state
   - Voice process state
   - API key cache

3. **Temporary Files**:
   - Audio recordings (WAV files) - deleted after transcription
   - Imported 3D models - stored in Blender scene

### Future Database Considerations

For future phases (Agent Mode, Partnership Mode), databases may be needed for:
- **Command History**: Log of all voice commands and operations
- **Scene Context**: Mesh analysis results, operation history
- **Learning Data**: User corrections, successful operation patterns
- **Multi-Agent Coordination**: Task assignments, conflict resolution

**Potential Database Solutions**:
- **SQLite**: Lightweight, file-based, no server required
- **JSON Files**: Simple key-value storage (already used for progress.json)
- **In-Memory Cache**: Redis (for multi-agent coordination)

---

## Transformers & Deep Learning

### Direct Transformer Usage: **None**

The system does not directly use the `transformers` Python library. However:

### Indirect Transformer Usage

1. **faster-whisper**:
   - Uses transformer architecture internally (encoder-decoder)
   - Implemented in C++ (whisper.cpp) for performance
   - No Python transformer library required

2. **GPT-4o**:
   - Transformer-based model (hosted by OpenAI)
   - Accessed via API, not local model
   - No local transformer inference

### Future Transformer Considerations

For future phases, local transformers may be needed for:
- **Mesh Analysis**: Understanding 3D geometry patterns
- **Operation Selection**: Local model for simple commands (reduce API costs)
- **Context Understanding**: Analyzing scene state and history

**Potential Local Models**:
- **Llama 3 / Mistral**: For simple command parsing (offline)
- **Specialized 3D Models**: If available for mesh analysis
- **Fine-tuned Models**: Trained on Blender operation patterns

---

## Dependencies & Technologies

### Python Packages

**Core Dependencies** (`requirements.txt`):
- `openai>=2.7.0`: GPT-4o API client

**Runtime Dependencies** (bundled in `stb_runtime/`):
- `numpy`: Audio processing, array operations
- `sounddevice`: Microphone input/output
- `faster-whisper`: Speech-to-text (via whisper-cli.exe, not Python package)

**Standard Library**:
- `xmlrpc.server`: RPC server
- `xmlrpc.client`: RPC client
- `subprocess`: Process execution
- `threading`: Background threads
- `queue`: Thread-safe queues
- `wave`: Audio file I/O
- `tempfile`: Temporary file management
- `json`: JSON parsing
- `re`: Regular expressions
- `os`, `sys`: System operations

### External Tools

1. **whisper-cli.exe**:
   - C++ implementation of Whisper
   - Location: `stb_runtime/whisper/whisper-cli.exe`
   - Model files: `ggml-tiny.en.bin`, etc.

2. **Bundled Python Runtime**:
   - Location: `stb_runtime/python/`
   - Self-contained Python 3.11 installation
   - Includes all dependencies pre-installed

### APIs & Services

1. **OpenAI API**:
   - Endpoint: `https://api.openai.com/v1/chat/completions`
   - Model: `gpt-4o`
   - Authentication: API key

2. **Meshy.ai API** (optional):
   - Endpoint: `https://api.meshy.ai` (configurable)
   - Purpose: Text-to-3D generation
   - Authentication: API key

---

## Data Flow Diagrams

### Voice Command Flow

```
┌─────────────────┐
│  User Speaks    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Microphone     │ (sounddevice)
│  Audio Capture  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Silence        │
│  Detection      │ (RMS threshold)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  WAV File       │ (temporary)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  whisper-cli    │ (faster-whisper)
│  Transcription  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Text Transcript│
└────────┬────────┘
         │
         ├───→ I/O Rules ────→ Import/Export Command
         │
         ├───→ Local Rules ───→ Regex Pattern Match
         │
         └───→ GPT-4o ────────→ Natural Language Understanding
                 │
                 ▼
         ┌─────────────────┐
         │  JSON Command   │ {"op": "...", "kwargs": {...}}
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │  XML-RPC Call   │ (localhost:8765)
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │  Safety Gate    │ (validation)
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │  Task Queue     │ (_TASKQ)
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │  Blender Main   │ (bpy.app.timers)
         │  Thread         │
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │  Blender Scene  │ (modified)
         └─────────────────┘
```

### RPC Communication Flow

```
┌─────────────────────┐         ┌─────────────────────┐
│  voice_to_blender  │         │  Blender Add-on     │
│  .py (Process)     │         │  (__init__.py)      │
└──────────┬──────────┘         └──────────┬──────────┘
           │                                │
           │  XML-RPC Request               │
           │  enqueue_op_safe(op, kwargs)   │
           ├───────────────────────────────>│
           │                                │
           │                                │  Safety Gate Check
           │                                │  _is_safe_op()
           │                                │
           │                                │  Queue Operation
           │                                │  _TASKQ.put()
           │                                │
           │  XML-RPC Response              │
           │  {"ok": true, "message": ...}  │
           │<───────────────────────────────┤
           │                                │
           │                                │  Timer Drains Queue
           │                                │  _drain_task_queue()
           │                                │
           │                                │  Execute Operation
           │                                │  bpy.ops.*
           │                                │
           │                                │  Undo Push
           │                                │  bpy.ops.ed.undo_push()
```

---

## File Structure

```
SpeechToBlender/
├── __init__.py                 # Blender add-on entry point
├── voice_to_blender.py          # Voice processing script
├── voice_to_blender.bat         # Windows batch wrapper
├── requirements.txt             # Python dependencies
│
├── stb_core/                    # Core functionality
│   ├── __init__.py
│   ├── config.py                # Configuration management
│   ├── pipeline.py              # Pipeline orchestration
│   ├── commands/                # Command definitions
│   │   ├── safety.py            # Safety validation
│   │   └── schema.py            # Command schemas
│   └── providers/               # AI provider integrations
│       ├── base.py              # Base provider class
│       ├── meshy.py             # Meshy.ai provider
│       ├── mock.py              # Mock provider (testing)
│       └── registry.py         # Provider registry
│
├── stb_runtime/                 # Bundled runtime
│   ├── python/                  # Python 3.11 runtime
│   │   ├── python.exe
│   │   └── Lib/
│   │       └── site-packages/   # Pre-installed packages
│   │           └── openai/     # OpenAI client
│   └── whisper/                 # Speech-to-text
│       ├── whisper-cli.exe      # Whisper executable
│       └── ggml-*.bin           # Model files
│
├── config/                      # Configuration files
│   └── config.json             # Provider settings
│
└── logs/                        # Log files
    └── progress.json            # Progress tracking
```

---

## Communication Protocols

### XML-RPC

**Protocol**: XML-RPC (XML Remote Procedure Call)

**Transport**: HTTP over localhost

**Port**: 8765

**Host**: 127.0.0.1 (localhost only, not exposed to network)

**Methods**:
- `enqueue_op_safe(op, kwargs)`: Queue a Blender operation
- `get_openai_api_key()`: Get API key from preferences
- `start_voice_command()`: Mark voice command start (undo grouping)
- `get_super_mode_state()`: Get Super Mode status

**Why XML-RPC?**:
- Simple, stdlib-only (no external dependencies)
- Works inside Blender's Python environment
- Sufficient for localhost communication
- Thread-safe with proper queue management

---

## Threading Model

### Threads

1. **Main Thread** (Blender UI):
   - Runs Blender's UI and scene operations
   - Drains task queue via `bpy.app.timers`
   - Executes all `bpy.ops.*` calls

2. **RPC Server Thread** (`_SERVER_THREAD`):
   - Background thread running XML-RPC server
   - Receives commands from voice script
   - Validates and queues operations (does NOT execute bpy.ops directly)

3. **Voice Script Process** (`voice_to_blender.py`):
   - Separate Python process (not a thread)
   - Handles audio I/O and transcription
   - Communicates via XML-RPC

### Thread Safety

- **Task Queue**: `queue.Queue` (thread-safe)
- **RPC Server**: Handles requests in background thread
- **Blender Operations**: Always executed on main thread via queue
- **No Direct bpy.ops Calls**: From RPC thread (prevents crashes)

---

## Configuration & Environment

### Environment Variables

- `OPENAI_API_KEY`: Fallback API key (if not in preferences)
- `WHISPER_CLI`: Path to whisper-cli executable
- `WHISPER_MODEL`: Model name (default: "small")
- `WHISPER_LANG`: Language code (default: "en")
- `FW_DEVICE`: faster-whisper device (cpu/cuda/auto)

### Blender Preferences

Stored in `bpy.context.preferences.addons["SpeechToBlender"]`:
- `openai_api_key_part1`: First part of OpenAI API key
- `openai_api_key_part2`: Second part of OpenAI API key
- `meshy_api_key`: Meshy.ai API key
- `meshy_base_url`: Meshy API endpoint
- `meshy_model`: Meshy model version
- `meshy_mode`: Preview or standard
- `super_mode_target_object`: Target object for Super Mode

---

## Performance Characteristics

### Latency Breakdown

1. **Audio Capture**: ~0.2-2 seconds (depends on speech length)
2. **Transcription**: ~200-500ms (faster-whisper small model)
3. **Local Rules**: ~5-10ms (regex matching)
4. **GPT-4o Call**: ~1-3 seconds (API latency)
5. **RPC Communication**: ~1-5ms (localhost)
6. **Blender Execution**: ~10-100ms (depends on operation)

**Total Latency**:
- **Local Rules Match**: ~250-600ms
- **GPT-4o Fallback**: ~1.5-4 seconds

### Optimization Strategies

- **API Key Caching**: Avoids repeated RPC calls (~20-100ms saved)
- **Local Rules First**: 80%+ of commands handled instantly
- **Reduced Sleep Delays**: Optimized timing (150-250ms saved)
- **No Key Validation**: Removed unnecessary API test (~500ms-2s saved)

---

## Security Considerations

### Safety Gate

- **Blacklist**: Blocks dangerous operators
- **Whitelist**: Allows safe operator prefixes
- **File Path Validation**: Checks import paths exist
- **No Network Exposure**: RPC server only on localhost

### API Key Security

- **Storage**: Blender preferences (encrypted by Blender)
- **Transmission**: Only over localhost XML-RPC
- **Caching**: In-memory only, cleared on RPC stop
- **No Logging**: Keys not logged or printed

### Privacy

- **Local STT**: Audio never leaves local machine
- **No Telemetry**: No data collection or tracking
- **User Control**: All operations require explicit voice commands

---

## Future Architecture Considerations

### Phase 2-5 Enhancements

1. **Mesh Context Analysis**:
   - Extract vertex/edge/face data
   - Analyze topology patterns
   - Feed to GPT-4o for context-aware operations

2. **Multi-Agent Coordination**:
   - Task distribution system
   - Conflict resolution
   - Real-time collaboration protocols

3. **Local Model Fallback**:
   - Llama 3 / Mistral for simple commands
   - Reduce API costs
   - Offline operation capability

4. **Database Integration**:
   - Command history logging
   - Scene context storage
   - Learning data collection

5. **Viewport Screenshot Analysis**:
   - GPT-4o vision API
   - Visual feedback for operations
   - Scene understanding

---

## Summary

**Current Stack**:
- **STT**: faster-whisper (local, transformer-based)
- **NLU**: GPT-4o (cloud, transformer-based)
- **Communication**: XML-RPC (localhost)
- **Storage**: Blender preferences + in-memory
- **Databases**: None
- **Direct Transformers**: None (indirect via faster-whisper and GPT-4o)

**Architecture**: Multi-process, multi-threaded, queue-based execution

**Key Design Principles**:
- Safety first (safety gate, validation)
- Performance (local rules, caching)
- Privacy (local STT, no telemetry)
- Flexibility (GPT fallback, extensible providers)



