"""
Gemini Live Listener (Phase 2): background thread that streams mic audio to Gemini Live
and pushes received text/code to the action queue. No top-level heavy imports.
Phase 5: scan text for PREDICT keywords and push instant primitives.
"""
import asyncio
import threading
import queue

# Phase 5: keywords that trigger instant PREDICT (must match addon _LIVE_PREDICT_MAP keys)
PREDICT_KEYWORDS = ("cube", "sphere", "cylinder", "plane", "cone", "torus", "camera", "light", "iphone")


class LiveListener(threading.Thread):
    """
    Listener thread: runs asyncio loop, connects to Gemini Live, streams mic,
    pushes incoming text to action_queue as {"type": "CODE"|"PREDICT"|"ERROR", "payload": str}.
    """

    def __init__(self, api_key: str, action_queue: queue.Queue, system_instruction: str = ""):
        super().__init__(daemon=True)
        self.api_key = api_key or ""
        self.action_queue = action_queue
        self.system_instruction = system_instruction or "You are a Blender assistant. Respond with only executable Blender Python code (bpy.ops... or bpy.context...). No explanations."
        self._running = False
        self._loop = None

    def run(self):
        self._running = True
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._run_session())
        except Exception as e:
            if self.action_queue:
                self.action_queue.put({"type": "ERROR", "payload": str(e)})
            import traceback
            traceback.print_exc()
        finally:
            self._running = False
            if self._loop:
                self._loop.close()

    def stop(self):
        self._running = False

    @property
    def running(self):
        return self._running

    async def _run_session(self):
        from google import genai
        import pyaudio

        client = genai.Client(api_key=self.api_key)
        model = "gemini-2.0-flash-exp"
        config = {
            "response_modalities": ["TEXT"],
            "system_instruction": self.system_instruction,
        }
        if self.system_instruction:
            config["system_instruction"] = self.system_instruction

        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        SEND_SAMPLE_RATE = 16000
        CHUNK_SIZE = 1024

        pya = pyaudio.PyAudio()
        audio_stream = None
        audio_queue_mic = asyncio.Queue(maxsize=5)

        async def listen_audio():
            nonlocal audio_stream
            try:
                mic_info = pya.get_default_input_device_info()
                audio_stream = await asyncio.to_thread(
                    pya.open,
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=SEND_SAMPLE_RATE,
                    input=True,
                    input_device_index=mic_info["index"],
                    frames_per_buffer=CHUNK_SIZE,
                )
                kwargs = {"exception_on_overflow": False} if not __debug__ else {}
                while self._running:
                    data = await asyncio.to_thread(audio_stream.read, CHUNK_SIZE, **kwargs)
                    await audio_queue_mic.put({"data": data, "mime_type": "audio/pcm"})
            except asyncio.CancelledError:
                pass
            except Exception as e:
                if self.action_queue:
                    self.action_queue.put({"type": "ERROR", "payload": f"Mic: {e}"})

        async def send_realtime(session):
            try:
                while self._running:
                    msg = await asyncio.wait_for(audio_queue_mic.get(), timeout=0.5)
                    await session.send_realtime_input(audio=msg)
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                pass
            except Exception as e:
                if self.action_queue:
                    self.action_queue.put({"type": "ERROR", "payload": f"Send: {e}"})

        async def receive_and_queue(session):
            try:
                turn = session.receive()
                async for response in turn:
                    if not self._running:
                        break
                    sc = getattr(response, "server_content", None)
                    if not sc:
                        continue
                    mt = getattr(sc, "model_turn", None)
                    if not mt:
                        continue
                    for part in getattr(mt, "parts", []) or []:
                        text_obj = getattr(part, "text", None)
                        text = getattr(text_obj, "text", None) if text_obj else (getattr(part, "text", None) or "")
                        if isinstance(text, str) and text.strip():
                            t = text.strip()
                            text_lower = t.lower()
                            for kw in PREDICT_KEYWORDS:
                                if kw in text_lower:
                                    self.action_queue.put({"type": "PREDICT", "payload": kw})
                                    break
                            self.action_queue.put({"type": "CODE", "payload": t})
            except asyncio.CancelledError:
                pass
            except Exception as e:
                if self.action_queue:
                    self.action_queue.put({"type": "ERROR", "payload": f"Receive: {e}"})

        try:
            async with client.aio.live.connect(model=model, config=config) as live_session:
                await asyncio.gather(
                    send_realtime(live_session),
                    listen_audio(),
                    receive_and_queue(live_session),
                )
        except asyncio.CancelledError:
            pass
        finally:
            if audio_stream:
                try:
                    audio_stream.stop_stream()
                    audio_stream.close()
                except Exception:
                    pass
            pya.terminate()
