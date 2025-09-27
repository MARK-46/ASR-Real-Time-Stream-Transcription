import os
import time
import soundfile as sf

class SessionWriter:
    def __init__(self, location: str, sample_rate: int, channels: int):
        self.location = location
        self.sample_rate = sample_rate
        self.channels = channels
        self.writer = None
        self.path = None
        os.makedirs(self.location, exist_ok=True)
        print(f"[I] Directory ensured: {self.location}")

    def open(self):
        self.close()
        try:
            self.path = os.path.join(self.location, f"session-{int(time.time())}.wav")
            self.writer = sf.SoundFile(
                self.path, 
                mode="w", 
                samplerate=self.sample_rate, 
                channels=self.channels,
            )
            print(f"[I] SessionWriter opened file: {self.path} "
                  f"(Sample Rate: {self.sample_rate}, Channels: {self.channels})")
        except Exception as e:
            print(f"[E] Failed to open file: {self.path}, Exception: {e}")

    def write(self, audio_chunk):
        if self.writer is None:
            print("[W] No open session to write audio.")
            return
        try:
            self.writer.write(audio_chunk)
            print(f"[I] Written chunk of size: {len(audio_chunk)} samples")
        except Exception as e:
            print(f"[E] Failed to write audio chunk: {e}")

    def close(self):
        if self.writer is not None:
            try:
                self.writer.close()
                print(f"[I] SessionWriter saved file: {self.path}")
            except Exception as e:
                print(f"[E] Failed to close file: {self.path}, Exception: {e}")
            finally:
                self.writer = None
                self.path = None
