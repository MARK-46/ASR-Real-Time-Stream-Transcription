import torch
import torchaudio
import numpy as np
import gradio as gr
from typing import Tuple
from pathlib import Path
from pydub import AudioSegment
from src.tools import stream_porcessor as sp

_processor = sp.StreamProcessor(sample_rate=16_000)

def _convert_to_float32(y):
    if y.dtype == np.int16:
        return y.astype(np.float32) / 32768.0
    elif y.dtype == np.int32:
        return y.astype(np.float32) / 2147483648.0
    elif y.dtype == np.float64:
        return y.astype(np.float32)
    else:
        return y.astype(np.float32)


def stream_fn(audio: Tuple[int, np.ndarray], segments_state: list):
    try:
        audio_sr, audio_chunk = audio
        
        # Convert to mono if stereo
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.mean(axis=1)
        
        # Properly convert to float32 with correct scaling
        audio_chunk = _convert_to_float32(audio_chunk)
        
        # Clip to [-1, 1] range (safety measure)
        audio_chunk = np.clip(audio_chunk, -1.0, 1.0)

        # Resample to 16kHz
        y_tensor = torch.tensor(audio_chunk)
        y_resampled = torchaudio.functional.resample(y_tensor, orig_freq=audio_sr, new_freq=_processor.sample_rate).numpy()
        
        # Process through pipeline
        _processor.process_chunk(y_resampled)

        transcription = _processor.get_transcribe()
        if transcription:
            chunk_path, duration, text_en, text_ru = transcription
            segments_state.append([chunk_path, f"{duration:.3f}s", text_en, text_ru])

        return segments_state, segments_state
    except Exception as e:
        print(f"Error in stream_fn: {e}")
        return segments_state, segments_state


def play_segment(evt: gr.SelectData, raw_ts_list) -> Tuple[gr.Audio, str, str]:
    if not isinstance(raw_ts_list, list):
        print(f"Warning: raw_ts_list is not a list ({type(raw_ts_list)}). Cannot play segment.")
        return gr.Audio(value=None, visible=False)

    selected_index = evt.index[0]

    if selected_index < 0 or selected_index >= len(raw_ts_list):
        print(f"Invalid index {selected_index} selected for list of length {len(raw_ts_list)}.")
        return gr.Audio(value=None, visible=False)

    current_audio_path = raw_ts_list[selected_index][0]
    
    if not current_audio_path or not Path(current_audio_path).exists():
        print(f"Warning: Audio path '{current_audio_path}' not found or invalid for clipping.")
        return gr.Audio(value=None, visible=False)

    audio = AudioSegment.from_file(current_audio_path)
    samples = np.array(audio.get_array_of_samples())
    frame_rate = audio.frame_rate

    if samples.size == 0:
        print(f"Warning: Audio resulted in empty samples array.")
        return gr.Audio(value=None, visible=False)

    text_en, text_ru = _processor.transcribe_segment(speech_array=samples)

    return (
        gr.Audio(value=(frame_rate, samples), autoplay=True, 
                label=f"Segment: {current_audio_path}", 
                interactive=False, 
                visible=True),
        text_en, 
        text_ru
    )


def handle_deselect():
    return gr.Audio(value=None, visible=False), None, None


# Gradio interface
with gr.Blocks(title="Live Transcription (ENG -> RUS)", theme=gr.themes.Soft(), css_paths='static/styles.css') as web:
    with gr.Row():
        input = gr.Audio(
            label="🎤 Входное аудио (микрофон)",
            sources=["microphone"],
            type="numpy",
            streaming=True,
            elem_classes="audio-container",
        )

    with gr.Row():
        vis_timestamps_df = gr.DataFrame(
            headers=["Segment Path", "Duration", "English", "Русский"],
            datatype=["str", "number", "str", "str"],
            wrap=True,
            label="📝 Результат распознавания",
            column_widths=["15%", "5%", "40%", "40%"],
        )
    with gr.Row():
        with gr.Column():
            selected_segment_player = gr.Audio(label="Selected Segment", interactive=False)
            clear_btn = gr.Button(value="Close Segment", variant='primary')
        with gr.Column():
            output_en = gr.TextArea(
                label="📝 Результат распознавания (ENG)",
                elem_id="transcription_box_en",
                lines=3,
                autoscroll=True,
                interactive=False,
                placeholder="English transcription will appear here...",
            )
            output_ru = gr.TextArea(
                label="📝 Результат распознавания (RUS)",
                elem_id="transcription_box_ru",
                lines=3,
                autoscroll=True,
                interactive=False,
                placeholder="Русская транскрипция появится здесь...",
            )

    segments_state = gr.State([])

    input.stream(
        fn=stream_fn,
        inputs=[input, segments_state],
        outputs=[vis_timestamps_df, segments_state],
        stream_every=0.5, # new_chunk duration 0.5 sec
    )

    input.start_recording(fn=_processor.start, inputs=None, outputs=None)
    input.stop_recording(fn=_processor.stop, inputs=None, outputs=None)

    vis_timestamps_df.select(
        fn=play_segment,
        inputs=[segments_state],
        outputs=[selected_segment_player, output_en, output_ru],
    )

    clear_btn.click(
        fn=handle_deselect,
        inputs=[],
        outputs=[selected_segment_player, output_en, output_ru],
    )

if __name__ == "__main__":
    web.launch(debug=False, server_name="0.0.0.0", server_port=8080)