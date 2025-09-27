import re
from deep_translator import GoogleTranslator

class Translator:
    def __init__(self, source: str = 'en', target: str = 'ru'):
        self.translator = GoogleTranslator(source=source, target=target)
        self.translated_texts = []
        self.buffered_texts = []

    def _translate_transcription(self, text):
        rus_text = ''
        if not any('а' <= c <= 'я' or c == 'ё' for c in text.lower()):
            try:
                rus_text = f'<div style="font-weight: bold;">{self.translator.translate(text)}</div>'
            except Exception as e:
                print(f"Translation failed: {e}")
                rus_text = '<div style="font-weight: bold; color: red;">Translation failed</div>'
        return f"""
            <div style="font-family: system-ui, sans-serif;">
                <div style="color: gray; font-size: 14px;">{text}</div>
                {rus_text}
            </div>
        """

    def append_text(self, text: str):
        self.buffered_texts.append(text)

    def translate(self):
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
