import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cpu' #or 'cpu' for translate on cpu
torch.set_default_device(device)

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

def translate(text: str):
    prefix = 'Correct errors in the text, clarify and improve its meaning, then translate from English to Russian, preserving accuracy and natural language: '
    src_text = prefix + text
    inputs = tokenizer(src_text, return_tensors="pt", return_attention_mask=False)
    outputs = model.generate(**inputs, max_length=200)
    text = tokenizer.batch_decode(outputs)[0]
    print(text)

translate("shell we start or need to wait?")
