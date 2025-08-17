# mic_to_whisper_local.py
# Listens to your mic, saves a WAV when you finish speaking, runs whisper.cpp locally, prints transcript.

import sounddevice as sd
import numpy as np
import wave, os, time, tempfile, subprocess, sys
from datetime import datetime

# ========= CONFIG =========
WHISPER_EXE = r"C:\Users\Jordan\Downloads\whisper-bin-Win32\Release\whisper-cli.exe"
MODEL_PATH  = r"C:\Users\Jordan\Downloads\whisper-bin-Win32\Release\ggml-tiny.en.bin"
# ==========================

SAMPLE_RATE = 16000
BLOCK_SEC   = 0.2
SILENCE_RMS = 500
SILENCE_HOLD= 0.8
MIN_SPOKEN  = 0.7

def rms_int16(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(block.astype(np.int32) ** 2)))

def record_one_take():
    print("🎙️  Listening… speak now. (Ctrl+C to quit)")
    frames, spoken_sec, silence_sec = [], 0.0, 0.0
    block_samples = int(BLOCK_SEC * SAMPLE_RATE)

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16') as stream:
            while True:
                block, _ = stream.read(block_samples)
                block = block.reshape(-1)
                frames.append(block)
                level = rms_int16(block)
                if level < SILENCE_RMS:
                    silence_sec += BLOCK_SEC
                else:
                    spoken_sec += BLOCK_SEC
                    silence_sec = 0.0
                if silence_sec >= SILENCE_HOLD and spoken_sec >= MIN_SPOKEN:
                    break
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
        return None

    audio = np.concatenate(frames, axis=0).astype(np.int16)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmpdir = os.path.join(tempfile.gettempdir(), "voice_clips")
    os.makedirs(tmpdir, exist_ok=True)
    wav_path = os.path.join(tmpdir, f"clip_{ts}.wav")

    with wave.open(wav_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())

    print(f"💾 Saved: {wav_path}  (spoken≈{spoken_sec:.1f}s)")
    return wav_path

def transcribe_with_whisper(wav_path: str) -> str | None:
    if not os.path.isfile(WHISPER_EXE):
        print(f"❌ whisper-cli.exe not found: {WHISPER_EXE}")
        return None
    if not os.path.isfile(MODEL_PATH):
        print(f"❌ model file not found: {MODEL_PATH}")
        return None
    base_out = wav_path
    cmd = [WHISPER_EXE, "-m", MODEL_PATH, "-f", wav_path, "-otxt", "--no-timestamps", "-of", base_out]
    try:
        print("🧠 Transcribing with whisper.cpp…")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print("❌ whisper-cli failed:")
            print(res.stderr or res.stdout)
            return None
    except Exception as e:
        print(f"❌ Failed to run whisper-cli: {e}")
        return None

    txt_path = base_out + ".txt"
    if not os.path.exists(txt_path):
        print("❌ Expected transcript file not found:", txt_path)
        return None

    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    return text if text else None

def main():
    print("Voice → WAV → whisper.cpp (Ctrl+C to exit)\n")
    while True:
        wav = record_one_take()
        if not wav:
            break
        text = transcribe_with_whisper(wav)
        if text:
            print(f"📝 Transcript: {text}\n")
        else:
            print("⚠️ No transcription.\n")
        time.sleep(0.25)

if __name__ == "__main__":
    print(f"Using whisper: {WHISPER_EXE}")
    print(f"Using model  : {MODEL_PATH}")
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye.")
        sys.exit(0)
