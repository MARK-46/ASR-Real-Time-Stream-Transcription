# python 3.11, Gradio 5.46.0, nemo 2.4.0

import numpy as np
import gradio as gr

from translator import Translator
from transcriptor import Transcription

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8080
SAMPLE_RATE = 16_000

transcriptor = Transcription(sample_rate=SAMPLE_RATE)
translator = Translator(source='en', target='ru')

def stream_fn(audio: np.ndarray):
    try:
        if audio is None:
            raise ValueError("Audio input cannot be None")
        
        audio_sr, audio_chunk = audio

        if not isinstance(audio_chunk, np.ndarray):
            raise TypeError("Audio data must be a numpy array")

        if audio_chunk.size == 0:
            raise TypeError("Empty audio chunk received")
        
        text_en = transcriptor.process_audio(audio_sr=audio_sr, audio_chunk=audio_chunk)

        # add en text to translate
        translator.append_text(text_en)

        # get all translated textes
        text_ru = translator.translate()

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
        stream_every=10.0,
    )

    input.start_recording(fn=transcriptor.start, inputs=None, outputs=None)
    input.stop_recording(fn=transcriptor.stop, inputs=None, outputs=None)

    print("[I] Launching UI with enhanced audio processing")
    web.launch(debug=False, server_name=SERVER_HOST, server_port=SERVER_PORT)