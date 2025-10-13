# python 3.11, Gradio 5.46.0, nemo 2.4.0

import numpy as np
import gradio as gr

from asr import Transcriptor
from translator import Translator
from session_writer import SessionWriter

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8080
SAMPLE_RATE = 16_000

translator = Translator(source='en', target='ru')
transcriptor = Transcriptor(model_name="nvidia/parakeet-tdt-0.6b-v3", sample_rate=SAMPLE_RATE)
session_writer = SessionWriter(location='./sessions', sample_rate=SAMPLE_RATE, channels=1)

def stream_fn(audio: np.ndarray):
    try:
        if audio is None:
            raise ValueError("Audio input cannot be None")
        
        audio_sr, audio_chunk = audio
        audio_chunk = transcriptor.prepare_audio(audio_sr=audio_sr, audio_chunk=audio_chunk)
        session_writer.write(audio_chunk)
        
        text_en = transcriptor.transcribe_audio(audio_chunk=audio_chunk)
        text_ru = translator.translate(text_en)

        return (text_ru)
    except Exception as e:
        print(f"[E] Stream error: {e}")
        return (f"[E] Stream error: {e}")

def clear_output():
    translator.clean()
    return ("")

css = """
footer {
    display:none !important;
} 

.audio-container > .top-panel {
    display:none!important;
} 

.mic-select {
    max-width: none;
}

#transcription_box {
    padding: 8px;
    background: #ffffff;
    height: 300px;
    overflow-y: auto;
    border: 1px solid #ccc;
    scroll-behavior: smooth;
}

.status-indicator {
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: bold;
}

.status-active {
    background-color: #d4edda;
    color: #155724;
}

.status-inactive {
    background-color: #f8d7da;
    color: #721c24;
}
"""

js_func = """
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

with gr.Blocks(
    title="Live Transcription with Improved Processing",
    theme=gr.themes.Soft(primary_hue=gr.themes.colors.red, secondary_hue=gr.themes.colors.purple),
    css=css,
    js=js_func
) as web:
    gr.Markdown("# Live Transcription with Enhanced Audio Processing (ENG -> RUS)")
    gr.Markdown("🎤 Real-time speech recognition with improved buffering and quality")

    with gr.Row():
        with gr.Column():
            input = gr.Audio(
                label="Input (Enhanced Processing)",
                sources=["microphone"],
                type="numpy",
                streaming=True,
            )
        with gr.Column():
            output = gr.Markdown(label="Transcription", elem_id="transcription_box")
            clear_btn = gr.Button("Clear Transcription", variant='primary')

    clear_btn.click(
        fn=clear_output,
        inputs=None,
        outputs=[output],
    )
    
    input.stream(
        fn=stream_fn,
        inputs=[input],
        outputs=[output],
        stream_every=15.0,
    )

    input.start_recording(fn=session_writer.open, inputs=None, outputs=None)
    input.stop_recording(fn=session_writer.close, inputs=None, outputs=None)

    print("[I] Launching UI with enhanced audio processing")
    web.launch(debug=False, server_name=SERVER_HOST, server_port=SERVER_PORT)
