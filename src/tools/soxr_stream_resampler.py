import time
import numpy as np
import soxr

CLEAR_STREAM_AFTER_SECS = 0.2

class SOXRStreamAudioResampler:
    def __init__(self):
        self._in_rate: float | None = None
        self._out_rate: float | None = None
        self._last_resample_time: float = 0
        self._soxr_stream: soxr.ResampleStream | None = None

    def _initialize(self, in_rate: float, out_rate: float):
        self._in_rate = in_rate
        self._out_rate = out_rate
        self._last_resample_time = time.time()
        self._soxr_stream = soxr.ResampleStream(
            in_rate=in_rate, 
            out_rate=out_rate, 
            num_channels=1,
            quality="VHQ",  # Very High Quality
            dtype="int16"
        )

    def _maybe_clear_internal_state(self):
        current_time = time.time()
        time_since_last_resample = current_time - self._last_resample_time
        if time_since_last_resample > CLEAR_STREAM_AFTER_SECS:
            if self._soxr_stream:
                self._soxr_stream.clear()
        self._last_resample_time = current_time

    def _maybe_initialize_sox_stream(self, in_rate: int, out_rate: int):
        if self._soxr_stream is None:
            self._initialize(in_rate, out_rate)
        else:
            self._maybe_clear_internal_state()

        if self._in_rate != in_rate or self._out_rate != out_rate:
            raise ValueError(
                f"SOXRStreamAudioResampler cannot be reused with different sample rates: "
                f"expected {self._in_rate}->{self._out_rate}, got {in_rate}->{out_rate}"
            )

    def resample(self, audio: np.ndarray, in_rate: int, out_rate: int, is_last: bool = False) -> np.ndarray:
        if in_rate == out_rate:
            return audio

        self._maybe_initialize_sox_stream(in_rate, out_rate)
        return self._soxr_stream.resample_chunk(audio, last=is_last)
    
    def reset(self):
        if self._soxr_stream:
            self._soxr_stream.clear()
        self._last_resample_time = time.time()
