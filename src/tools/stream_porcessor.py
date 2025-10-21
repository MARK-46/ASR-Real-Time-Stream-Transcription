import torch
import numpy as np
from queue import Queue
from typing import Tuple, Optional

from src.tools.translator import Translator
from src.tools.audio_writer import AudioWriter
from src.tools.voice_detector import VoiceDetector
from src.transcribe.whisper import WhisperModel
from src.transcribe.parakeet import ParakeetModel
from src.transcribe.canary import CanaryModel


class StreamProcessor:
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._transcribes_queue = Queue()
        
        self._audio_writer = AudioWriter(
            output_dir='./sessions',
            sample_rate=self.sample_rate,
            prefix='chunk'
        )

        self._detector = VoiceDetector(
            sample_rate=self.sample_rate,
            on_speech_end=self._on_speech_end
        )
        self._detector.set_options(
            frame_duration = 0.03,
            energy_threshold = 0.01,
            min_silence_duration = 1.5,
        )

        # self._asr = WhisperModel(model_name="openai/whisper-large-v3")
        # self._asr = ParakeetModel(model_name="nvidia/parakeet-tdt-0.6b-v3")
        self._asr = CanaryModel(model_name="nvidia/canary-qwen-2.5b")

        self._translator = Translator(source='en', target='ru')
    
    def process_chunk(self, chunk: np.ndarray):
        self._detector.process_chunk(chunk)
    
    def start(self):
        print(f"[I] StreamProcessor started")
    
    def stop(self):
        self._detector.reset()
        print("[I] StreamProcessor stopped")
    
    def _on_speech_end(self, audio_data: np.ndarray, duration: float):
        if duration < 0.5:
            return # skip short audio
        
        chunk_path = self._audio_writer.write_chunk(audio_data)
        text_en, text_ru = self.transcribe_segment(speech_array=audio_data)
        self._transcribes_queue.put((chunk_path, duration, text_en, text_ru))

    def get_transcribe(self) -> Optional[Tuple[str, float, str, str]]:
        return None if self._transcribes_queue.empty() else self._transcribes_queue.get_nowait()
    
    def transcribe_segment(self, speech_array: np.ndarray) -> Tuple[str, str]:
        text_en = self._asr.transcribe(speech_array=speech_array)
        text_ru = self._translator.translate(text=text_en)
        return (text_en, text_ru)