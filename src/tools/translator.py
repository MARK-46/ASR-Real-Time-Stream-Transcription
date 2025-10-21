import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

class Translator:
    def __init__(self, source: str = 'en', target: str = 'ru'):
        self.model_name = 'utrobinmv/t5_translate_en_ru_zh_large_1024'
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.prefix = f'translate to {target}: '

    def translate(self, text: str):
        src_text = self.prefix + text
        input_ids = self.tokenizer(src_text, return_tensors="pt")
        generated_tokens = self.model.generate(**input_ids.to(self.device))
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
