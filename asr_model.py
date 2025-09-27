import torch
import numpy as np
from typing import Union, List
from nemo.collections.asr.models import ASRModel as _ASRModel


class ASRModel:
    def __init__(self, sample_rate: int, model_name: str):
        if sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {sample_rate}")
        
        self.sample_rate = sample_rate
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        
        try:
            self._load_model(model_name)
        except Exception as e:
            print(f"[E] Failed to load model: {e}")
            raise
    
    def _load_model(self, model_name: str):
        print(f"[I] Loading Parakeet-TDT model...")
        
        self.model = _ASRModel.from_pretrained(model_name=model_name)
        self.model.eval()
        
        if self.device == "cuda":
            self.model = self.model.to(device=self.device, dtype=torch.bfloat16)
        else:
            self.model = self.model.to(device=self.device)
        
        self._warm_up()
        print(f"[I] Model loaded successfully on {self.device}")
    
    def _warm_up(self):
        with torch.inference_mode():
            dummy_audio = np.zeros(self.sample_rate, dtype=np.float32)
            _ = self.model.transcribe([dummy_audio])
    
    def transcribe_chunks(self, audio_chunks: Union[np.ndarray, List[np.ndarray]]) -> str:
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        if isinstance(audio_chunks, np.ndarray):
            audio_chunks = [audio_chunks]
        
        try:
            with torch.inference_mode():
                results = self.model.transcribe(audio_chunks, )
                
            texts = []
            for i, result in enumerate(results):
                if hasattr(result, 'text') and result.text:
                    print(f"[I] Transcribe[{i}]: {result.text}")
                    texts.append(result.text.strip())
            
            return ' '.join(texts)
            
        except Exception as e:
            print(f"[E] Transcription failed: {e}")
            return ""
    
    def close(self):
        if self.model is not None:
            del self.model
            self.model = None
            
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            print(f"[I] Model resources released")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()