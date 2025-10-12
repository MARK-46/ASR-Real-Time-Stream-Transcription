import torch
import torchaudio
import numpy as np

from asr_model import ASRModel
from session_writer import SessionWriter


class Transcription:
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.asr_model = ASRModel(model_name="nvidia/parakeet-tdt-0.6b-v3", sample_rate=sample_rate)
        self.session_writer = SessionWriter(location='./sessions', sample_rate=sample_rate, channels=1)

    def start(self):
        self.session_writer.open()

    def stop(self):
        self.session_writer.close()

    def close(self):
        self.asr_model.close()
        self.session_writer.close()

    # поток аудио чанка по 5 секунд
    def process_audio(self, audio_sr: int, audio_chunk: np.ndarray) -> str:
        try:
            # Convert stereo to mono
            if audio_chunk.ndim > 1:
                audio_chunk = audio_chunk.mean(axis=1)

            # Cast to float32
            if audio_chunk.dtype == np.int16:
                audio_chunk = audio_chunk.astype(np.float32) / 32768.0
            elif audio_chunk.dtype == np.int32:
                audio_chunk = audio_chunk.astype(np.float32) / 2147483648.0
            else:
                audio_chunk = audio_chunk.astype(np.float32)

            # Resample to self.sample_rate
            audio_chunk = self.__resample_audio(audio_chunk, audio_sr, self.sample_rate)

            # Normalize
            audio_chunk = self.__normalize_audio(audio_chunk)

            # Write in to file
            self.session_writer.write(audio_chunk)

            # Transcribe (nemo - nvidia/parakeet-tdt-0.6b-v3)
            text = self.asr_model.transcribe_chunks([audio_chunk]) # без поддержки инкрементальной обработки.

            return text
        except Exception as e:
            print(f"[E] process_audio error: {e}")
            return ""
    
    def __resample_audio(self, audio_chunk: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return audio_chunk
        
        tensor = torch.from_numpy(audio_chunk).unsqueeze(0)  # [1, T]
        resampled = torchaudio.functional.resample(tensor, orig_freq=orig_sr, new_freq=target_sr)
        return resampled.squeeze(0).numpy()
    
    def __normalize_audio(self, audio_chunk: np.ndarray, target_level: float = 0.9) -> np.ndarray:
        # RMS нормализация с ограничением пиков
        rms = np.sqrt(np.mean(audio_chunk**2))
        if rms < 1e-6:
            return audio_chunk
    
        # коэффициент усиления
        gain = target_level / rms
        normalized = audio_chunk * gain
    
        # предотвращение клиппинга
        peak = np.max(np.abs(normalized))
        if peak > 1.0:
            normalized = normalized / peak
    
        return normalized
    