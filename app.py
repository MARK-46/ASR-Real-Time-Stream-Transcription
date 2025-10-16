# python 3.11, Gradio 5.46.0, nemo 2.4.0
# pip install silero-vad torch torchaudio

import asr
import gradio as gr
import numpy as np
import torch
import translator
from silero_vad import load_silero_vad
from collections import deque

class StreamingVAD:
    def __init__(self, sample_rate=16000, threshold=0.5):
        self.model = load_silero_vad()
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.is_speaking = False
        
    def process_chunk(self, audio_chunk):
        if isinstance(audio_chunk, np.ndarray):
            if len(audio_chunk.shape) > 1:
                audio_chunk = audio_chunk[:, 0] if audio_chunk.shape[1] > 0 else audio_chunk[0, :]
            audio_tensor = torch.from_numpy(audio_chunk.copy())
        else:
            audio_tensor = audio_chunk
        if audio_tensor.dtype == torch.int16 or (audio_tensor.abs().max() > 1.0 and audio_tensor.abs().max() < 50000):
            audio_tensor = audio_tensor.float() / 32768.0
        elif audio_tensor.dtype != torch.float32:
            audio_tensor = audio_tensor.float()
        max_amplitude = audio_tensor.abs().max().item()
        if max_amplitude < 0.01:
            return {'probability': 0.0, 'is_speech': False, 
                   'speech_started': False, 'speech_ended': False}
        window_size = 512
        speech_probs = []
        for i in range(0, len(audio_tensor), window_size):
            window = audio_tensor[i:i+window_size]
            if len(window) < window_size:
                window = torch.nn.functional.pad(window, (0, window_size - len(window)))
            try:
                prob = self.model(window, self.sample_rate).item()
                speech_probs.append(prob)
            except Exception as e:
                print(f"[VAD ERROR] Window processing failed: {e}")
                continue
        if not speech_probs:
            speech_prob = 0.0
        else:
            speech_prob = max(speech_probs)
        result = {
            'probability': speech_prob,
            'is_speech': speech_prob > self.threshold,
            'speech_started': False,
            'speech_ended': False
        }
        if not self.is_speaking and speech_prob > self.threshold:
            result['speech_started'] = True
            self.is_speaking = True
        elif self.is_speaking and speech_prob <= self.threshold:
            result['speech_ended'] = True
            self.is_speaking = False
        return result


class ImprovedTranscriptorWithVAD:
    def __init__(self, model_name, sample_rate, overlap_duration=0.5, use_vad=True, 
                 vad_threshold=0.5, silence_chunks=1, min_speech_chunks=2):
        self.transcriptor = asr.Transcriptor(model_name=model_name, sample_rate=sample_rate)
        self.sample_rate = sample_rate
        self.overlap_samples = int(overlap_duration * sample_rate)
        self.use_vad = use_vad
        self.max_silence_chunks = silence_chunks
        self.min_speech_chunks = min_speech_chunks
        if use_vad:
            self.vad = StreamingVAD(sample_rate=sample_rate, threshold=vad_threshold)
        self.previous_chunk = None
        self.text_buffer = deque(maxlen=3)
        self.speech_buffer = []
        self.silence_counter = 0
        
    def open(self):
        self.previous_chunk = None
        self.text_buffer.clear()
        self.speech_buffer = []
        self.silence_counter = 0
        self.transcriptor.open()
    
    def close(self):
        result = ""
        if len(self.speech_buffer) > 0:
            full_audio = np.concatenate(self.speech_buffer)
            result = self._transcribe_chunk(full_audio, self.sample_rate)
        self.previous_chunk = None
        self.text_buffer.clear()
        self.speech_buffer = []
        self.silence_counter = 0
        self.transcriptor.close()
        return result
    
    def _merge_chunks(self, current_chunk):
        if self.previous_chunk is None or len(self.previous_chunk) < self.overlap_samples:
            return current_chunk
        original_dtype = current_chunk.dtype
        overlap = self.previous_chunk[-self.overlap_samples:].astype(np.float32)
        if len(current_chunk) >= self.overlap_samples:
            fade_out = np.linspace(1, 0, self.overlap_samples)
            fade_in = np.linspace(0, 1, self.overlap_samples)
            current_overlap = current_chunk[:self.overlap_samples].astype(np.float32)
            overlap_mixed = (overlap * fade_out + current_overlap * fade_in)
            overlap_mixed = overlap_mixed.astype(original_dtype)
            merged = np.concatenate([overlap_mixed, current_chunk[self.overlap_samples:]])
        else:
            merged = np.concatenate([overlap.astype(original_dtype), current_chunk])
        return merged.astype(original_dtype)
    
    def _clean_text(self, text, prev_text):
        if not text or not prev_text:
            return text
        text_words = text.strip().split()
        prev_words = prev_text.strip().split()
        if not text_words or not prev_words:
            return text
        max_overlap = min(5, len(prev_words), len(text_words))
        for i in range(max_overlap, 0, -1):
            if prev_words[-i:] == text_words[:i]:
                return ' '.join(text_words[i:])
        return text
    
    def _transcribe_chunk(self, audio_chunk, in_rate):
        merged_chunk = self._merge_chunks(audio_chunk)
        self.previous_chunk = audio_chunk.copy()
        text = self.transcriptor.transcribe_audio(
            audio_chunk=merged_chunk
        )
        if not text or not text.strip():
            return ""
        prev_text = self.text_buffer[-1] if self.text_buffer else ""
        cleaned_text = self._clean_text(text, prev_text)
        self.text_buffer.append(cleaned_text)
        return cleaned_text
    
    def transcribe_audio(self, audio_chunk, in_rate):
        if in_rate != self.sample_rate:
            audio_chunk = self.transcriptor.resample_audio(audio_chunk, in_rate)
            in_rate = self.sample_rate
        if not self.use_vad:
            return self._transcribe_chunk(audio_chunk, in_rate)
        vad_result = self.vad.process_chunk(audio_chunk)
        if vad_result['is_speech']:
            self.speech_buffer.append(audio_chunk)
            self.silence_counter = 0
            return ""
        elif len(self.speech_buffer) > 0:
            self.silence_counter += 1
            enough_speech = len(self.speech_buffer) >= self.min_speech_chunks
            enough_silence = self.silence_counter >= self.max_silence_chunks
            if enough_speech and enough_silence:
                full_audio = np.concatenate(self.speech_buffer)
                text = self._transcribe_chunk(full_audio, in_rate)
                self.speech_buffer = []
                self.silence_counter = 0
                return text
            elif not enough_speech:
                print(f"[VAD] Слишком короткая речь: {len(self.speech_buffer)}/{self.min_speech_chunks}")
        return ""

_transcriptor = ImprovedTranscriptorWithVAD(
    model_name="nvidia/parakeet-tdt_ctc-1.1b", 
    sample_rate=16_000,
    overlap_duration=0.3,        # Меньше overlap = быстрее
    use_vad=True,
    vad_threshold=0.4,           # Ниже порог = чувствительнее
    silence_chunks=1,            # Ждать всего 1 тихий чанк (5 сек)
    min_speech_chunks=1          # Минимум 1 чанк речи (5 сек)
)
_translator = translator.Translator()

async def stream_fn(audio):
    global _transcriptor
    global _translator
    if audio is None:
        return ""
    audio_sr, audio_chunk = audio
    text = _transcriptor.transcribe_audio(audio_chunk=audio_chunk, in_rate=audio_sr)
    print(f'>> EN: {text}')
    return _translator.translate(text)

with gr.Blocks(
    title="Live Transcription with VAD",
    theme=gr.themes.Soft(),
    css="""
        #transcription_box {
            padding: 8px;
            background: #ffffff;
            height: 300px;
            overflow-y: auto;
            border: 1px solid #ccc;
            scroll-behavior: smooth;
        }
    """,
    js="""
        function refresh() {
            const url = new URL(window.location);
            if (url.searchParams.get('__theme') !== 'light') {
                url.searchParams.set('__theme', 'light');
                window.location.href = url.href;
            }
            const box = document.getElementById('transcription_box');
            if (box) {
                const observer = new MutationObserver(() => {
                    box.scrollTop = box.scrollHeight;
                });
                observer.observe(box, { childList: true, subtree: true });
            }
        }
    """
) as web:
    with gr.Row():
        with gr.Column(scale=1):
            input = gr.Audio(
                label="🎤 Входное аудио (микрофон)",
                sources=["microphone"],
                type="numpy",
                streaming=True,
            )
        with gr.Column(scale=1):
            output = gr.Markdown(
                label="📝 Результат распознавания",
                elem_id="transcription_box",
            )
    
    input.stream(
        fn=stream_fn,
        inputs=[input],
        outputs=[output],
        stream_every=1.5,
    )

    input.start_recording(fn=_transcriptor.open, inputs=None, outputs=None)
    input.stop_recording(fn=_transcriptor.close, inputs=None, outputs=None)

    print("[I] Запуск интерфейса с VAD и улучшенной обработкой")
    web.launch(debug=False, server_name="0.0.0.0", server_port=8080)