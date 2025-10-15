# python 3.11, Gradio 5.46.0, nemo 2.4.0
# pip install silero-vad torch torchaudio

import asr
import gradio as gr
import numpy as np
import torch
from silero_vad import load_silero_vad
from collections import deque

class StreamingVAD:
    def __init__(self, sample_rate=16000, threshold=0.5):
        self.model = load_silero_vad()
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.is_speaking = False
        
    def process_chunk(self, audio_chunk):
        """Определяет наличие речи в аудио чанке"""
        # Диагностика входных данных
        print(f"[VAD DEBUG] Input type: {type(audio_chunk)}, shape: {audio_chunk.shape if hasattr(audio_chunk, 'shape') else 'N/A'}, dtype: {audio_chunk.dtype if hasattr(audio_chunk, 'dtype') else 'N/A'}")
        
        # Конвертируем numpy в torch tensor
        if isinstance(audio_chunk, np.ndarray):
            # Если многоканальный - берем первый канал
            if len(audio_chunk.shape) > 1:
                audio_chunk = audio_chunk[:, 0] if audio_chunk.shape[1] > 0 else audio_chunk[0, :]
            
            audio_tensor = torch.from_numpy(audio_chunk.copy())
        else:
            audio_tensor = audio_chunk
        
        # Нормализация в диапазон [-1, 1]
        if audio_tensor.dtype == torch.int16 or (audio_tensor.abs().max() > 1.0 and audio_tensor.abs().max() < 50000):
            audio_tensor = audio_tensor.float() / 32768.0
        elif audio_tensor.dtype != torch.float32:
            audio_tensor = audio_tensor.float()
        
        # Проверка амплитуды
        max_amplitude = audio_tensor.abs().max().item()
        print(f"[VAD DEBUG] Tensor shape: {audio_tensor.shape}, max amplitude: {max_amplitude:.4f}")
        
        # Если аудио слишком тихое, сразу возвращаем 0
        if max_amplitude < 0.01:
            print("[VAD DEBUG] Audio too quiet, skipping VAD")
            return {'probability': 0.0, 'is_speech': False, 
                   'speech_started': False, 'speech_ended': False}
        
        # VAD требует чанки по 512 сэмплов для 16kHz
        window_size = 512
        speech_probs = []
        
        # Разбиваем на окна и обрабатываем каждое
        for i in range(0, len(audio_tensor), window_size):
            window = audio_tensor[i:i+window_size]
            
            # Паддинг если окно меньше нужного размера
            if len(window) < window_size:
                window = torch.nn.functional.pad(window, (0, window_size - len(window)))
            
            try:
                prob = self.model(window, self.sample_rate).item()
                speech_probs.append(prob)
            except Exception as e:
                print(f"[VAD ERROR] Window processing failed: {e}")
                continue
        
        # Берем максимальную/среднюю вероятность по всем окнам
        if not speech_probs:
            speech_prob = 0.0
        else:
            speech_prob = max(speech_probs)  # Или можно использовать mean(speech_probs)
        
        print(f"[VAD DEBUG] Processed {len(speech_probs)} windows, max speech probability: {speech_prob:.4f}")
        
        # Получаем вероятность речи
        try:
            print(f"[VAD DEBUG] Speech probability: {speech_prob:.4f}")
        except Exception as e:
            print(f"[VAD ERROR] {e}")
            return {'probability': 0.0, 'is_speech': False, 
                   'speech_started': False, 'speech_ended': False}
        
        # Определяем состояние
        result = {
            'probability': speech_prob,
            'is_speech': speech_prob > self.threshold,
            'speech_started': False,
            'speech_ended': False
        }
        
        # Детектируем начало/конец речи
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
        """
        Args:
            silence_chunks: Сколько тихих чанков ждать перед обработкой (меньше = быстрее)
            min_speech_chunks: Минимум речевых чанков перед распознаванием
            vad_threshold: Порог определения речи (0-1)
        """
        self.transcriptor = asr.Transcriptor(model_name=model_name, sample_rate=sample_rate)
        self.sample_rate = sample_rate
        self.overlap_samples = int(overlap_duration * sample_rate)
        self.use_vad = use_vad
        
        # Настройки VAD
        self.max_silence_chunks = silence_chunks  # Уменьшите для быстрой реакции
        self.min_speech_chunks = min_speech_chunks  # Минимум речи перед распознаванием
        
        # VAD для определения речи
        if use_vad:
            self.vad = StreamingVAD(sample_rate=sample_rate, threshold=vad_threshold)
        
        # Буферы
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
        # Обрабатываем оставшееся в буфере
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
        """Создает чанк с перекрытием из предыдущего"""
        if self.previous_chunk is None or len(self.previous_chunk) < self.overlap_samples:
            return current_chunk
        
        # Сохраняем оригинальный dtype
        original_dtype = current_chunk.dtype
        
        # Берем последнюю часть предыдущего чанка
        overlap = self.previous_chunk[-self.overlap_samples:].astype(np.float32)
        
        # Применяем плавное затухание/нарастание (crossfade)
        if len(current_chunk) >= self.overlap_samples:
            fade_out = np.linspace(1, 0, self.overlap_samples)
            fade_in = np.linspace(0, 1, self.overlap_samples)
            
            current_overlap = current_chunk[:self.overlap_samples].astype(np.float32)
            overlap_mixed = (overlap * fade_out + current_overlap * fade_in)
            
            # Конвертируем обратно в оригинальный dtype
            overlap_mixed = overlap_mixed.astype(original_dtype)
            
            merged = np.concatenate([overlap_mixed, current_chunk[self.overlap_samples:]])
        else:
            merged = np.concatenate([overlap.astype(original_dtype), current_chunk])
        
        return merged.astype(original_dtype)
    
    def _clean_text(self, text, prev_text):
        """Удаляет дублирующиеся слова на границах"""
        if not text or not prev_text:
            return text
        
        text_words = text.strip().split()
        prev_words = prev_text.strip().split()
        
        if not text_words or not prev_words:
            return text
        
        # Ищем совпадение последних слов предыдущего текста с началом текущего
        max_overlap = min(5, len(prev_words), len(text_words))
        
        for i in range(max_overlap, 0, -1):
            if prev_words[-i:] == text_words[:i]:
                return ' '.join(text_words[i:])
        
        return text
    
    def _transcribe_chunk(self, audio_chunk, in_rate):
        """Внутренний метод распознавания"""
        # Создаем чанк с перекрытием
        merged_chunk = self._merge_chunks(audio_chunk)
        
        # Сохраняем текущий чанк для следующей итерации
        self.previous_chunk = audio_chunk.copy()
        
        # Распознаем речь
        text = self.transcriptor.transcribe_audio(
            audio_chunk=merged_chunk
        )
        
        if not text or not text.strip():
            return ""
        
        # Очищаем от дубликатов
        prev_text = self.text_buffer[-1] if self.text_buffer else ""
        cleaned_text = self._clean_text(text, prev_text)
        
        # Сохраняем в буфер
        self.text_buffer.append(cleaned_text)
        
        return cleaned_text
    
    def transcribe_audio(self, audio_chunk, in_rate):
        """Распознает аудио с VAD и улучшенной склейкой"""
        # ВАЖНО: Ресемплируем в целевой sample rate используя встроенный метод
        if in_rate != self.sample_rate:
            print(f"[RESAMPLE] {in_rate}Hz -> {self.sample_rate}Hz")
            audio_chunk = self.transcriptor.resample_audio(audio_chunk, in_rate)
            in_rate = self.sample_rate
        
        if not self.use_vad:
            # Без VAD - обычное распознавание
            return self._transcribe_chunk(audio_chunk, in_rate)
        
        # С VAD - умное накопление
        vad_result = self.vad.process_chunk(audio_chunk)
        
        print(f"[VAD] prob={vad_result['probability']:.2f}, "
              f"speech={vad_result['is_speech']}, "
              f"buffer={len(self.speech_buffer)}, silence={self.silence_counter}")
        
        # Если есть речь - накапливаем
        if vad_result['is_speech']:
            self.speech_buffer.append(audio_chunk)
            self.silence_counter = 0
            return ""  # Пока не распознаем
        
        # Если тишина после речи
        elif len(self.speech_buffer) > 0:
            self.silence_counter += 1
            
            # Проверяем минимальную длину речи и достаточную паузу
            enough_speech = len(self.speech_buffer) >= self.min_speech_chunks
            enough_silence = self.silence_counter >= self.max_silence_chunks
            
            if enough_speech and enough_silence:
                # Склеиваем все накопленные чанки
                full_audio = np.concatenate(self.speech_buffer)
                
                duration_sec = len(full_audio) / self.sample_rate
                print(f"[VAD] Обработка: {len(full_audio)} сэмплов ({duration_sec:.2f}s)")
                
                # Распознаем
                text = self._transcribe_chunk(full_audio, in_rate)
                
                # Очищаем буфер
                self.speech_buffer = []
                self.silence_counter = 0
                
                return text
            elif not enough_speech:
                print(f"[VAD] Слишком короткая речь: {len(self.speech_buffer)}/{self.min_speech_chunks}")
        
        return ""


# БЫСТРЫЙ РЕЖИМ (минимальная задержка)
transcriptor = ImprovedTranscriptorWithVAD(
    model_name="nvidia/parakeet-tdt_ctc-1.1b", 
    sample_rate=16_000,
    overlap_duration=0.3,        # Меньше overlap = быстрее
    use_vad=True,
    vad_threshold=0.4,           # Ниже порог = чувствительнее
    silence_chunks=1,            # Ждать всего 1 тихий чанк (5 сек)
    min_speech_chunks=1          # Минимум 1 чанк речи (5 сек)
)

def stream_fn(audio):
    global transcriptor
    if audio is None:
        return ""
    
    audio_sr, audio_chunk = audio
    
    # Диагностика
    print(f"\n[STREAM] Sample rate: {audio_sr}, chunk shape: {audio_chunk.shape}, dtype: {audio_chunk.dtype}")
    print(f"[STREAM] Chunk stats: min={audio_chunk.min()}, max={audio_chunk.max()}, mean={audio_chunk.mean():.2f}")
    
    text = transcriptor.transcribe_audio(audio_chunk=audio_chunk, in_rate=audio_sr)
    if text:
        print(f'> {text}')
    return text

with gr.Blocks(
    title="Live Transcription with VAD",
    theme=gr.themes.Soft()
) as web:
    gr.Markdown("""
    # 🎙️ Потоковое распознавание речи с VAD
    
    **Технологии:**
    - ✅ Silero VAD для определения речи
    - ✅ Перекрытие чанков (overlap) для плавности
    - ✅ Crossfade между чанками
    - ✅ Удаление дублирующихся слов
    - ✅ Накопление речи по фразам (не режет слова)
    
    **Как работает:**
    - Микрофон постоянно слушает
    - VAD определяет где есть речь
    - Накапливает полные фразы
    - Распознает только законченные фразы (после паузы)
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input = gr.Audio(
                label="🎤 Входное аудио (микрофон)",
                sources=["microphone"],
                type="numpy",
                streaming=True,
            )
        with gr.Column(scale=1):
            output = gr.TextArea(
                label="📝 Результат распознавания",
                elem_id="transcription_box",
                autoscroll=True,
                lines=15,
                placeholder="Здесь появится распознанный текст..."
            )
    
    # Подключаем стриминг
    input.stream(
        fn=stream_fn,
        inputs=[input],
        outputs=[output],
        stream_every=1.5,  # Уменьшите для более быстрой реакции (минимум ~1.0)
    )

    # События старта/стопа
    input.start_recording(fn=transcriptor.open, inputs=None, outputs=None)
    input.stop_recording(fn=transcriptor.close, inputs=None, outputs=None)

    print("[I] Запуск интерфейса с VAD и улучшенной обработкой")
    web.launch(debug=False, server_name="0.0.0.0", server_port=8080)