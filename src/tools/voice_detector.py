import numpy as np
from typing import Optional, Callable

class VoiceDetector:
    def __init__(
        self, 
        sample_rate: int, 
        frame_duration: float = 0.03,       # seconds
        energy_threshold: float = 0.01,     # RMS threshold for speech detection
        min_silence_duration: float = 2.5,  # seconds to detect end of speech
        on_speech_end: Optional[Callable[[np.ndarray, float], None]] = None
    ):
        self.sample_rate = sample_rate
        self.on_speech_end_fn = on_speech_end
        self.set_options(
            frame_duration=frame_duration,
            energy_threshold=energy_threshold,
            min_silence_duration=min_silence_duration,
        )

    def set_options(
        self, 
        frame_duration: float,
        energy_threshold: float,
        min_silence_duration: float,
    ):
        self.frame_duration = frame_duration
        self.frame_samples = int(self.sample_rate * self.frame_duration)
        self.energy_threshold = energy_threshold
        self.min_silence_duration = min_silence_duration
        self.min_silence_samples = int(self.sample_rate * self.min_silence_duration)
        self.overlap_samples = self.min_silence_samples // 2
        self.reset()

    def process_chunk(self, chunk: np.ndarray):
        """
        chunk:
        - type:        numpy.ndarray
        - dtype:       float32
        - shape:       (N',)             # одномерный массив (моно)
        - channels:    1                 # аудио сведено в моно
        - sample_rate: 16000             # Гц после ресемплинга
        - value_range: [-1.0, 1.0]       # нормализовано по амплитуде
        - format:      raw PCM (не WAV, просто массив сэмплов)
        """
        chunk = np.concatenate((self.pending_buffer, chunk))
        self.pending_buffer = np.array([], dtype=np.float32)

        while len(chunk) >= self.frame_samples:
            frame = chunk[:self.frame_samples]
            chunk = chunk[self.frame_samples:]
            energy = np.sqrt(np.mean(frame**2))
            is_speech = energy > self.energy_threshold

            if self.current_state == 'silence':
                self.silence_buffer = np.concatenate((self.silence_buffer, frame))
                if len(self.silence_buffer) > self.overlap_samples:
                    self.silence_buffer = self.silence_buffer[-self.overlap_samples:]
                
                if is_speech:
                    self.current_state = 'speech'
                    self.utterance_buffer = np.concatenate((self.silence_buffer, frame))
                    self.silence_buffer = np.array([], dtype=np.float32)
                    self.consecutive_silence_samples = 0
            else:  # speech
                self.utterance_buffer = np.concatenate((self.utterance_buffer, frame))
                
                if is_speech:
                    self.consecutive_silence_samples = 0
                else:
                    self.consecutive_silence_samples += self.frame_samples
                    
                    if self.consecutive_silence_samples >= self.min_silence_samples:
                        trim_samples = self.consecutive_silence_samples - self.overlap_samples
                        
                        if trim_samples > 0:
                            utterance = self.utterance_buffer[:-trim_samples]
                        else:
                            utterance = self.utterance_buffer.copy()
                        
                        utt_duration = len(utterance) / self.sample_rate
                        
                        if self.on_speech_end_fn:
                            self.on_speech_end_fn(utterance, utt_duration)
                        
                        remaining_silence = self.utterance_buffer[-trim_samples:] if trim_samples > 0 else np.array([], dtype=np.float32)
                        self.silence_buffer = remaining_silence.copy()
                        
                        self.current_state = 'silence'
                        self.consecutive_silence_samples = 0
                        self.utterance_buffer = np.array([], dtype=np.float32)

        self.pending_buffer = chunk

    def reset(self):
        self.utterance_buffer = np.array([], dtype=np.float32)
        self.pending_buffer = np.array([], dtype=np.float32)
        self.silence_buffer = np.array([], dtype=np.float32)
        self.current_state = 'silence'
        self.consecutive_silence_samples = 0
