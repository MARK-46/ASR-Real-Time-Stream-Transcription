import re
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

class Translator:
    def __init__(self, source: str = 'en', target: str = 'ru'):
        self.translated_texts = []
        self.buffered_texts = []
        self.model_name = 'utrobinmv/t5_translate_en_ru_zh_large_1024'
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.prefix = f'translate to {target}: '

    def _translate(self, text: str):
        src_text = self.prefix + text
        input_ids = self.tokenizer(src_text, return_tensors="pt")
        generated_tokens = self.model.generate(**input_ids.to(self.device))
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    def _translate_transcription(self, text):
        rus_text = ''
        if not any('а' <= c <= 'я' or c == 'ё' for c in text.lower()):
            try:
                rus_text = f'<div style="font-weight: bold;">{self._translate(text)}</div>'
            except Exception as e:
                print(f"Translation failed: {e}")
                rus_text = '<div style="font-weight: bold; color: red;">Translation failed</div>'
        return f"""
            <div style="font-family: system-ui, sans-serif;">
                <div style="color: gray; font-size: 14px;">{text}</div>
                {rus_text}
            </div>
        """

    def translate(self, text: str):
        if text:
            self.buffered_texts.append(text)
            
            text = " ".join(self.buffered_texts).strip()
            sentences = re.findall(r'[^.!?]+[.!?]?', text)
            if sentences:
                self.translated_texts.extend([self._translate_transcription(s.strip()) for s in sentences])
                tail = re.sub(r'^(?:[^.!?]+[.!?]?\s*)+', '', text).strip()
                self.buffered_texts = [tail] if tail else []
        return "".join(self.translated_texts)

    def clean(self):
        self.translated_texts.clear()
        self.buffered_texts.clear()
