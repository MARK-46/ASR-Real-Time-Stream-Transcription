import os
os.environ["NEMO_CACHE_DIR"] = "/tmp/nemo_models_cache"

import torch
import numpy as np

            
class CanaryModel:
    def __init__(self, model_name: str = "nvidia/canary-qwen-2.5b"):
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._model = None
        self.load_model(model_name=model_name)
        
    def load_model(self, model_name: str):
        try:
            self.unload_model()
            from nemo.collections.speechlm2.models import SALM
            print(f"[I] Loading Canary-Qwen model '{model_name}'...")
            self._model = SALM.from_pretrained(model_name).to(self._device)
            print(f"[I] Canary-Qwen model loaded successfully on {self._device}")
        except Exception as e:
            print(f"[E] Failed to load Canary-Qwen model: {e}")
            self._model = None
            raise

    def unload_model(self):
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def transcribe(self, speech_array: np.ndarray) -> str:
        if self._model is None:
            print("[W] Model not loaded. Cannot transcribe.")
            return ""
        
        try:
            if isinstance(speech_array, np.ndarray):
                speech_tensor = torch.from_numpy(speech_array).float()
            else:
                speech_tensor = speech_array
            
            audios = speech_tensor.unsqueeze(0).to(self._device)
            audio_lens = torch.tensor([speech_tensor.shape[0]], dtype=torch.int64).to(self._device)
            
            answer_ids = self._model.generate(
                prompts=[
                    [{"role": "user", "content": f"Transcribe the following: {self._model.audio_locator_tag}"}],
                ],
                audios=audios,
                audio_lens=audio_lens,
                max_new_tokens=128,
            )
            return self._model.tokenizer.ids_to_text(answer_ids[0].cpu())
        except Exception as e:
            print(f"[E] Transcription failed: {e}")
            return ""