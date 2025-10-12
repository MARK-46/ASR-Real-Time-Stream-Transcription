# app.py – Urdu ASR Studio with Faster-Whisper + optional LLM Polishing

import os
import json
from typing import List, Optional

import gradio as gr
import torch
import faster_whisper

# ────────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────────

os.environ.setdefault("HF_HOME", "/home/user/app/.cache")

MODEL_ID_CT2 = "openai/whisper-large-v3"
GROQ_MODEL = "openai/gpt-oss-120b"

DEFAULT_SYSTEM_PROMPT_UR = (
    "Вы опытный редактор на языке русского. "
    "При написании текста используйте орфографию, пунктуацию, интервалы и естественную беглость. "
    "Сохраняйте стиль и значение слов говорящего, не преувеличивайте, а распространенные английские термины оставьте такими, какие они есть."
)

# ────────────────────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────────────────────

def format_timestamp(seconds: float, format_type: str = "srt") -> str:
    total_ms = int(round((seconds or 0.0) * 1000))
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    sec, ms = divmod(rem_ms, 1000)
    sep = "," if format_type == "srt" else "."
    return f"{hours:02d}:{minutes:02d}:{sec:02d}{sep}{ms:03d}"

def basic_urdu_cleanup(text: str) -> str:
    if not text:
        return text
    t = " ".join(text.split())
    replacements = {
        " ,": ",", " .": ".", " ?": "?", " !": "!",
        " ،": "،", " ۔": "۔",
        ",": "،", ";": "؛",
        ". . .": "…", "...": "…",
    }
    for a, b in replacements.items():
        t = t.replace(a, b)
    t = t.replace(" ،", "،").replace(" ۔", "۔").replace(" ؛", "؛").replace(" ؟", "؟")
    for p in ["،", "؛", ",", ";"]:
        t = t.replace(p, p + " ")
    return " ".join(t.split()).strip()

# ───── Groq (OpenAI-compatible) client helpers ─────

def get_groq_client(api_key: Optional[str] = None):
    key = (api_key or os.getenv("GROQ_API_KEY", "")).strip()
    if not key:
        return None, "No GROQ_API_KEY provided."
    try:
        from groq import Groq  # type: ignore
        return Groq(api_key=key), None
    except Exception as e:
        return None, f"Groq client init failed: {e}"

def enhance_text_with_llm(text: str, api_key: Optional[str], temperature: float = 0.2,
                          system_prompt: str = DEFAULT_SYSTEM_PROMPT_UR) -> str:
    client, err = get_groq_client(api_key)
    if not client:
        if err:
            print(f"[LLM] {err} (falling back to basic cleanup)")
        return basic_urdu_cleanup(text)
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=float(temperature),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Пожалуйста, верните следующий текст на русском языке (без дополнительных комментариев):\n\n" + text},
            ],
        )
        return (resp.choices[0].message.content or "").strip() or basic_urdu_cleanup(text)
    except Exception as e:
        print(f"[LLM] Full-text enhance failed: {e}")
        return basic_urdu_cleanup(text)

def enhance_lines_with_llm(lines: List[str], api_key: Optional[str], temperature: float = 0.2,
                           system_prompt: str = DEFAULT_SYSTEM_PROMPT_UR) -> List[str]:
    if not lines:
        return lines
    client, err = get_groq_client(api_key)
    if not client:
        return [basic_urdu_cleanup(x) for x in lines]

    numbered = "\n".join(f"{i+1}. {ln}" for i, ln in enumerate(lines))
    user_msg = "Улучшите произношение этих предложений на русском. Верните то же количество строк в той же последовательности и с тем же счетом:\n\n" + numbered
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=float(temperature),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        improved_map = {}
        for line in raw.splitlines():
            s = line.strip()
            if not s or "." not in s:
                continue
            num, rest = s.split(".", 1)
            if num.strip().isdigit():
                improved_map[int(num) - 1] = rest.strip()
        return [improved_map.get(i, basic_urdu_cleanup(lines[i])) for i in range(len(lines))]
    except Exception as e:
        print(f"[LLM] Line enhance failed: {e}")
        return [basic_urdu_cleanup(x) for x in lines]

def test_groq(api_key: Optional[str], temperature: float, system_prompt: str) -> str:
    client, err = get_groq_client(api_key)
    if not client:
        return f"❌ LLM not ready: {err}"
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=float(temperature),
            messages=[
                {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT_UR},
                {"role": "user", "content": "Уточните короткую фразу и верните: \"это тест\"."},
            ],
        )
        txt = (resp.choices[0].message.content or "").strip()
        return f"✅ LLM OK · Sample: {txt}" if txt else "⚠️ LLM responded but empty content."
    except Exception as e:
        return f"❌ LLM call failed: {e}"

# ────────────────────────────────────────────────────────────────────────────────
# Whisper (CT2) Model
# ────────────────────────────────────────────────────────────────────────────────

print(f"CUDA available: {torch.cuda.is_available()}")
print("Loading model... this may take a minute the first time.")
model = faster_whisper.WhisperModel(
    MODEL_ID_CT2,
    device="cuda" if torch.cuda.is_available() else "cpu",
    compute_type="auto",
)
print("✅ Model loaded successfully!")

# ────────────────────────────────────────────────────────────────────────────────
# Core Transcription
# ────────────────────────────────────────────────────────────────────────────────

def transcribe_audio(
    audio_path: Optional[str],
    output_format: str,
    beam_size: int,
    llm_enhance: bool,
    llm_api_key: Optional[str],
    llm_temperature: float,
    llm_system_prompt: str,
):
    if not audio_path:
        raise gr.Error("Please upload or record an audio clip.")

    seg_iter, info = model.transcribe(
        audio_path, language="en", beam_size=int(beam_size),
        word_timestamps=False, vad_filter=False
    )

    segments, raw_lines = [], []
    for seg in seg_iter:
        text = (seg.text or "").strip()
        segments.append({"start": seg.start, "end": seg.end, "text": text})
        raw_lines.append(text)

    if llm_enhance:
        if output_format == "text":
            cleaned_lines = [enhance_text_with_llm(" ".join(raw_lines), llm_api_key, llm_temperature, llm_system_prompt)]
        else:
            cleaned_lines = enhance_lines_with_llm(raw_lines, llm_api_key, llm_temperature, llm_system_prompt)
    else:
        cleaned_lines = (
            [basic_urdu_cleanup(" ".join(raw_lines))] if output_format == "text"
            else [basic_urdu_cleanup(x) for x in raw_lines]
        )

    if output_format == "text":
        return cleaned_lines[0]
    if output_format == "srt":
        lines = []
        for i, s in enumerate(segments, 1):
            txt = cleaned_lines[i-1] if len(cleaned_lines) == len(segments) else s["text"]
            lines += [str(i), f"{format_timestamp(s['start'],'srt')} --> {format_timestamp(s['end'],'srt')}", txt, ""]
        return "\n".join(lines)
    if output_format == "vtt":
        lines = ["WEBVTT", ""]
        for i, s in enumerate(segments, 1):
            txt = cleaned_lines[i-1] if len(cleaned_lines) == len(segments) else s["text"]
            lines += [f"{format_timestamp(s['start'],'vtt')} --> {format_timestamp(s['end'],'vtt')}", txt, ""]
        return "\n".join(lines)
    if output_format == "json":
        segs_out = []
        for i, s in enumerate(segments):
            txt = cleaned_lines[i] if len(cleaned_lines) == len(segments) else s["text"]
            segs_out.append({"start": s["start"], "end": s["end"], "text": txt})
        return json.dumps({"text": " ".join(cleaned_lines), "segments": segs_out}, ensure_ascii=False, indent=2)

    raise gr.Error(f"Unsupported format: {output_format}")

# ────────────────────────────────────────────────────────────────────────────────
# UI
# ────────────────────────────────────────────────────────────────────────────────

theme = gr.themes.Soft(primary_hue="rose", secondary_hue="violet", neutral_hue="slate")

with gr.Blocks(title="ASR Studio — Faster-Whisper + LLM Polishing", theme=theme) as iface:
    # Custom CSS to fix spacing + output height
    gr.HTML("""
    <style>
      .gradio-container { padding-bottom: 16px !important; }
      #result_box textarea {
        min-height: 260px !important;
        max-height: 360px !important;
        overflow-y: auto !important;
      }
    </style>
    """)

    gr.Markdown(
        "## **Urdu STT with GPT-OSS 120B**  \n"
        "High-quality Urdu transcription with Faster-Whisper (CT2) and optional Groq LLM polishing."
    )

    with gr.Row():
        with gr.Column(scale=5):
            audio = gr.Audio(
                sources=["upload","microphone"], type="filepath",
                label="Upload or Record Audio",
                waveform_options={"show_controls": False},
                autoplay=False, streaming=False,
            )
            with gr.Accordion("Transcription Settings", open=False):
                with gr.Row():
                    fmt = gr.Radio(choices=["text","srt","vtt","json"], value="text", label="Output Format")
                    beam = gr.Slider(1,10,5,step=1,label="Beam Size")
            with gr.Accordion("LLM Polishing (Optional)", open=False):
                llm_toggle = gr.Checkbox(value=False,label="Polish Urdu text with LLM (Groq · openai/gpt-oss-120b)")
                with gr.Row():
                    llm_temp = gr.Slider(0.0,1.0,0.2,step=0.05,label="LLM Temperature")
                    llm_key = gr.Textbox(label="GROQ_API_KEY (optional if set in environment)", type="password", value="")
                llm_sys = gr.Textbox(label="LLM System Prompt (Urdu)", value=DEFAULT_SYSTEM_PROMPT_UR, lines=3)
                with gr.Row():
                    test_btn = gr.Button("Test LLM", variant="secondary")
                    test_status = gr.Markdown("")
            with gr.Row():
                btn = gr.Button("Transcribe", variant="primary")

        with gr.Column(scale=7):
            out = gr.Textbox(label="Result", lines=14, max_lines=30, show_copy_button=True, elem_id="result_box")

    btn.click(fn=transcribe_audio, inputs=[audio, fmt, beam, llm_toggle, llm_key, llm_temp, llm_sys], outputs=out)
    test_btn.click(fn=test_groq, inputs=[llm_key,llm_temp,llm_sys], outputs=[test_status])

if __name__ == "__main__":
    iface.launch()