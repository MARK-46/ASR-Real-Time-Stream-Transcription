import os
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_models_cache"

import torch
import numpy as np

class WhisperModel:
    def __init__(self, model_name: str = "openai/whisper-large-v3"):
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._pipe = None
        self.load_model(model_name=model_name)

    def load_model(self, model_name: str):
        try:
            self.unload_model()
            from transformers import pipeline
            print(f"[I] Loading Whisper model '{model_name}'...")
            self._pipe = pipeline(
                task="automatic-speech-recognition",
                model=model_name,
                chunk_length_s=30,
                device=self._device,
            )
            print(f"[I] Whisper model loaded successfully on {self._device}")
        except Exception as e:
            print(f"[E] Failed to load Whisper model: {e}")
            self._pipe = None
            raise

    def unload_model(self):
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def transcribe(self, speech_array: np.ndarray) -> str:
        if self._pipe is None:
            print("[W] Model not loaded. Cannot transcribe.")
            return ""
        
        try:
            inputs = {
                "array": speech_array, 
                "sampling_rate": self._pipe.feature_extractor.sampling_rate
            }
            result = self._pipe(
                inputs, 
                batch_size=8, 
                generate_kwargs={"task": 'transcribe'}
            )
            return result["text"]
        except Exception as e:
            print(f"[E] Transcription failed: {e}")
            return ""
