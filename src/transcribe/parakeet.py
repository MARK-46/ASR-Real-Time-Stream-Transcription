import torch
import numpy as np


class ParakeetModel:
    def __init__(self, model_name: str = "nvidia/parakeet-tdt-0.6b-v3"):
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._model = None
        # Supported models:
        # EncDecRNNTBPEModel
        #   stt_en_conformer_transducer_large
        #   nvidia/parakeet-rnnt-1.1b
        #   nvidia/parakeet-tdt-0.6b-v3
        #   nvidia/parakeet-tdt-1.1b
        # EncDecCTCModelBPE
        #   nvidia/parakeet-ctc-1.1b
        self.load_model(model_name=model_name)

    def load_model(self, model_name: str):
        try:
            self.unload_model()
            import nemo.collections.asr as nemo_asr
            print(f"[I] Loading Parakeet model '{model_name}'...")
            model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name).to(self._device)
            model.cfg.decoding.strategy = "greedy_batch"
            model.change_decoding_strategy(model.cfg.decoding)
            model.eval()
            print(f"[I] Parakeet model loaded successfully on {self._device}")
            self._model = model
        except Exception as e:
            print(f"[E] Failed to load Parakeet model: {e}")
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
            # Use autocast only on CUDA devices
            if self._device == 'cuda':
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    with torch.inference_mode():
                        return self._transcribe_internal(speech_array)
            else:
                with torch.inference_mode():
                    return self._transcribe_internal(speech_array)
        except Exception as e:
            print(f"[E] Transcription failed: {e}")
            return ""
    
    def _transcribe_internal(self, speech_array: np.ndarray) -> str:
        transcripts = self._model.transcribe(
            audio=[speech_array],
            batch_size=1,
            timestamps=False,
            verbose=False
        )
        transcript = transcripts[0]
        
        if isinstance(transcript, str):
            return transcript
        if hasattr(transcript, "text"):
            return transcript.text
        if isinstance(transcript, dict) and "text" in transcript:
            return transcript["text"]
        return str(transcript)
