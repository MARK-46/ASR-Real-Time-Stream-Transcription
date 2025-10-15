import torch
import traceback
import numpy as np
import session_writer as sw
import soxr_stream_resampler as soxr
import nemo.collections.asr as nemo_asr

class Transcriptor:
    """Transcribe (nemo - nvidia/parakeet-tdt-0.6b-v3)"""

    def __init__(self, sample_rate: int = 16_000, model_name: str = 'nvidia/parakeet-tdt-0.6b-v3'):
        if sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {sample_rate}")
        self._sample_rate = sample_rate
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._asr_model = self._load_model(model_name=model_name, device=self._device)
        self._resampler = soxr.SOXRStreamAudioResampler()
        self._session_writer = sw.SessionWriter(location='./sessions', sample_rate=sample_rate, channels=1)

    def _load_model(self, model_name: str, device: str):
        print(f"[I] Loading Parakeet-TDT model...")
        model: nemo_asr.models.EncDecHybridRNNTCTCBPEModel = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name=model_name, map_location=device)
        if device == "cuda":
            model = model.cuda()
        else:
            model = model.cpu()
        model.eval()
        print(f"[I] Model loaded successfully on {self._device}")
        return model
    
    def resample_audio(self, audio_chunk: np.ndarray, in_rate: int) -> np.ndarray:
        audio_chunk = self._resampler.resample(
            audio=audio_chunk,
            in_rate=in_rate,
            out_rate=self._sample_rate
        )
        return audio_chunk

    def transcribe_audio(self, audio_chunk: np.ndarray) -> str:
        try:
            with torch.inference_mode():
                transcripts = self._asr_model.transcribe(
                    audio=[audio_chunk],
                    batch_size=1,
                    timestamps=False,
                    verbose=True
                )
                self._session_writer.write(audio_chunk=audio_chunk)
                transcript = transcripts[0]
                if isinstance(transcript, str):
                    return transcript
                if hasattr(transcript, "text"):
                    return transcript.text
                if isinstance(transcript, dict) and "text" in transcript:
                    return transcript["text"]
                return str(transcript)
        except Exception as e:
            print(f"[E] process_audio error: {e}")
            traceback.print_stack()
            return ""

    def open(self):
        self._session_writer.open()

    def close(self):
        self._session_writer.close()
