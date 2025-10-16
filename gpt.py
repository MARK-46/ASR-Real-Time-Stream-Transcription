from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import io
import os

MODEL_NAME = "openai/gpt-oss-20b"

# Патчинг open() для корректного чтения UTF-8 на Windows
import builtins
open_builtin = open
def open_utf8(path, *args, **kwargs):
    if path.endswith("chat_template.jinja") and "encoding" not in kwargs:
        kwargs["encoding"] = "utf-8"
    return open_builtin(path, *args, **kwargs)
builtins.open = open_utf8

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype="auto"
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

def correct_transcript(raw_text: str) -> str:
    prompt = f"""
Ты — профессиональный корректор речи.
Исправь ошибки распознавания, грамматики и пунктуации, не меняя смысла.
Текст:
{raw_text}

Исправленный вариант:
"""
    result = generator(
        prompt,
        max_new_tokens=256,
        temperature=0.2,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )[0]["generated_text"]

    if "Исправленный вариант:" in result:
        result = result.split("Исправленный вариант:")[-1].strip()
    return result


if __name__ == "__main__":
    draft = "здравствуйте это эээ компания бета хотели уточнить по заказ тридцать два"
    print(correct_transcript(draft))
