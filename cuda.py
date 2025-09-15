import torch
import queue
import logging
import threading
import numpy as np
import gradio as gr
from scipy import signal
from threading import Lock
from nemo.collections.asr.models import ASRModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("asr_app")

# Constants
SR = 16_000
CHUNK_SECONDS = 4
CHUNK_SAMPLES = SR * CHUNK_SECONDS
MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

asr_lock = Lock()

class SharedModel:
    def __init__(self):
        logger.info("Downloading Parakeet-TDT once …")
        self.model = ASRModel.from_pretrained(model_name=MODEL_NAME)
        self.model.eval()
        
        # Move to appropriate device
        self.model = self.model.to(DEVICE)
        
        # Use mixed precision for better performance when using GPU
        if DEVICE == "cuda":
            self.model = self.model.to(torch.bfloat16)

        with torch.inference_mode():
            _ = self.model.transcribe([np.zeros(SR, dtype=np.float32)])
            logger.info("Model loaded fully on %s", DEVICE)

shared_model = SharedModel()

class ASRSession:
    def __init__(self):
        self.audio_q = queue.Queue(maxsize=8)
        self.txt_q = queue.Queue()
        self.transcripts = []
        self.active = True
        threading.Thread(target=self._worker, daemon=True).start()

    def close(self):
        self.active = False
        self.audio_q.put(None)

    def _worker(self):
        buf = np.array([], dtype=np.float32)
        while self.active:
            try:
                while len(buf) < CHUNK_SAMPLES and self.active:
                    audio_chunk = self.audio_q.get()
                    if audio_chunk is None:
                        self.active = False
                        break
                    buf = np.concatenate([buf, audio_chunk])
                if not self.active:
                    break
                while len(buf) >= CHUNK_SAMPLES and self.active:
                    chunk, buf = buf[:CHUNK_SAMPLES], buf[CHUNK_SAMPLES:]
                    with torch.inference_mode():
                        with asr_lock:
                            out = shared_model.model.transcribe([chunk])
                    self.txt_q.put(out[0].text)
            except Exception as e:
                logger.error(f"ASR error: {e}")
        while not self.txt_q.empty():
            self.txt_q.get()

    def preprocess(self, audio):
        sr, y = audio
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != SR:
            y = signal.resample_poly(y, SR, sr)
        y = y.astype(np.float32)
        y /= (np.abs(y).max() + 1e-9)
        return y

def stream_fn(audio, state: ASRSession):
    if state.active:
        state.audio_q.put(state.preprocess(audio))
    while not state.txt_q.empty():
        text = state.txt_q.get()
        state.transcripts.append(text)
    return (
        " ".join(state.transcripts) if state.transcripts else "…listening…",
        state,
    )

with gr.Blocks() as demo:
    mic = gr.Audio(sources=["microphone"], type="numpy", streaming=True)
    out = gr.Textbox(label="Transcription")
    session_state = gr.State(lambda: ASRSession())

    mic.stream(
        fn=stream_fn,
        inputs=[mic, session_state],
        outputs=[out, session_state],
        stream_every=0.5,
    )

if __name__ == "__main__":
    logger.info("Launching UI")
    demo.launch()