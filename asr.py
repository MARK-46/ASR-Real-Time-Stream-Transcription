# python==3.11
# numpy==1.26.4
# torchaudio==2.8.0
# nemo_toolkit[asr]==2.4.0

import gc
import torch
import torchaudio
import numpy as np
import nemo.collections.asr as nemo_asr
from pydub import AudioSegment

class Transcriptor:
    """Transcribe (nemo - nvidia/parakeet-tdt-0.6b-v3)"""

    TARGET_NORMALIZATION_LEVEL = 0.9
    MIN_RMS_THRESHOLD = 1e-8

    def __init__(self, sample_rate: int, model_name: str):
        if sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {sample_rate}")
        self.sample_rate = sample_rate
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self._load_model(model_name)
    
    def _load_model(self, model_name: str):
        print(f"[I] Loading Parakeet-TDT model...")
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
        self.model.eval()
        self.model.to(self.device)
        self.model.to(torch.float32)
        self._warm_up()
        print(f"[I] Model loaded successfully on {self.device}")
    
    def _warm_up(self):
        with torch.inference_mode():
            # dummy_audio = np.random.normal(0, 1e-4, int(self.sample_rate * 0.1)).astype(np.float32)
            # _ = self.model.transcribe([dummy_audio])

            audio = AudioSegment.from_file("C:\\_Mark\\Projects\\ASR-Real-Time-Stream-Transcription\\scripts\\2025-10-12 12-08-06.mp3")
            audio = audio[:30_000]
            duration_sec = audio.duration_seconds
            target_sr = 16000
            if audio.frame_rate != target_sr:
                audio = audio.set_frame_rate(target_sr)
            if audio.channels == 2:
                audio = audio.set_channels(1)
            audio.export("C:\\_Mark\\Projects\\ASR-Real-Time-Stream-Transcription\\scripts\\2025-10-12 12-08-06.wav", format="wav")
            if duration_sec > 480:
                self.model.change_attention_model("rel_pos_local_attn", [256,256])
                self.model.change_subsampling_conv_chunking_factor(1)  # 1 = auto select
            self.model.to(torch.bfloat16)
            txt = self.model.transcribe(["C:\\_Mark\\Projects\\ASR-Real-Time-Stream-Transcription\\scripts\\2025-10-12 12-08-06.wav"], timestamps=True)[0].text
            print(f'> {txt}')
    
    def prepare_audio(self, audio_sr: int, audio_chunk: np.ndarray) -> np.ndarray:
        if not isinstance(audio_chunk, np.ndarray):
            raise TypeError("Audio data must be a numpy array")
        if audio_chunk.size == 0:
            raise ValueError("Empty audio chunk received")
        if audio_sr <= 0:
            raise ValueError(f"Invalid sample rate: {audio_sr}")
            
        audio = self.__stereo2mono(audio_chunk)
        audio = self.__cast2float32(audio)
        audio = self.__resample_audio(audio, audio_sr, self.sample_rate)
        audio = np.nan_to_num(audio, nan=0.0)
        audio = self.__normalize_audio(audio)
        return audio

    def transcribe_audio(self, audio_chunk: np.ndarray) -> str:
        try:
            with torch.inference_mode():
                output = self.model.transcribe([audio_chunk])[0]
                if isinstance(output, str):
                    return output
                if hasattr(output, "text"):
                    return output.text
                if isinstance(output, dict) and "text" in output:
                    return output["text"]
                return str(output)
        except Exception as e:
            print(f"[E] process_audio error: {e}")
            return ""
    
    def __stereo2mono(self, audio_chunk: np.ndarray) -> np.ndarray:
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.mean(axis=1)
        return audio_chunk

    def __cast2float32(self, audio_chunk: np.ndarray) -> np.ndarray:
        dtype = audio_chunk.dtype
        if dtype == np.int16:
            return audio_chunk.astype(np.float32) / 32768.0
        if dtype == np.int32:
            return audio_chunk.astype(np.float32) / 2147483648.0
        return audio_chunk.astype(np.float32)

    def __resample_audio(self, audio_chunk: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return audio_chunk
        tensor = torch.from_numpy(audio_chunk).unsqueeze(0)  # shape [1, T]
        res = torchaudio.functional.resample(tensor, orig_freq=orig_sr, new_freq=target_sr)
        return res.squeeze(0).numpy()
    
    def __normalize_audio(self, audio_chunk: np.ndarray, target_level: float = None) -> np.ndarray:
        if target_level is None:
            target_level = self.TARGET_NORMALIZATION_LEVEL
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        if rms < self.MIN_RMS_THRESHOLD:
            return audio_chunk
        gain = target_level / rms
        normalized = audio_chunk * gain
        peak = np.max(np.abs(normalized))
        if peak > 1.0:
            normalized = normalized / peak
        return normalized
    
    def close(self):
        if self.model is not None:
            del self.model
            self.model = None
            if self.device == "cuda":
                torch.cuda.empty_cache()
            print(f"[I] Model resources released")
        gc.collect()