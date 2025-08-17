# -*- coding: utf-8 -*-
# Minimal CLI wrapper around faster-whisper.
# Prints exactly ONE line of JSON: {"text": "..."} to stdout.

import os, sys, json, warnings, logging

# Silence noisy libraries and Python warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("CT2_VERBOSE", "0")
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

logging.basicConfig(level=logging.ERROR)
for name in ("ctranslate2", "faster_whisper", "urllib3", "huggingface_hub", "tokenizers", "onnxruntime"):
    logging.getLogger(name).setLevel(logging.ERROR)

def parse(args):
    cfg = {"model": "small", "language": "en", "input": None}
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--model" and i+1 < len(args):
            i += 1; cfg["model"] = args[i]
        elif a == "--language" and i+1 < len(args):
            i += 1; cfg["language"] = args[i]
        elif a == "--input" and i+1 < len(args):
            i += 1; cfg["input"] = args[i]
        i += 1
    return cfg

def main():
    cfg = parse(sys.argv[1:])
    if not cfg["input"] or not os.path.exists(cfg["input"]):
        print(json.dumps({"error":"missing or invalid --input"}))
        sys.exit(2)
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        print(json.dumps({"error": f"faster-whisper not installed: {e}"}))
        sys.exit(3)

    # Device selection: CPU by default; set FW_DEVICE=auto/cpu/cuda to override.
    device = os.environ.get("FW_DEVICE","cpu")
    compute_type = "int8" if device == "cpu" else "float16"

    try:
        model = WhisperModel(cfg["model"], device=device, compute_type=compute_type)
        segments, info = model.transcribe(cfg["input"], language=cfg["language"], vad_filter=True)
        out_text = "".join(s.text for s in segments).strip()
        print(json.dumps({"text": out_text}, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(4)

if __name__ == "__main__":
    main()
