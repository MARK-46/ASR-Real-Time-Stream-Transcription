import os
import shutil
import numpy as np
import soundfile as sf


class AudioWriter:
    def __init__(
        self,
        output_dir: str = './sessions',
        sample_rate: int = 16000,
        prefix: str = 'chunk'
    ):
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self.prefix = prefix
        self._chunk_id = 0
        
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def write_chunk(self, audio_chunk: np.ndarray) -> str:
        filepath = self._get_filepath()
        
        with sf.SoundFile(
            filepath,
            mode='w',
            samplerate=self.sample_rate,
            channels=1,
            format='WAV'
        ) as w:
            w.write(audio_chunk)
        
        self._chunk_id += 1
        return filepath
    
    def _get_filepath(self) -> str:
        filename = f"{self.prefix}-{self._chunk_id}.wav"
        return os.path.join(self.output_dir, filename)
    
    def get_chunk_count(self) -> int:
        return self._chunk_id
    
    def reset_counter(self):
        self._chunk_id = 0
        print("[W] AudioWriter counter reset")