"""Web app (Gradio). Runs locally and on Hugging Face Spaces / Docker.

    python app.py   ->  open http://localhost:7860
"""
import os
import tempfile
import uuid
from pathlib import Path

import gradio as gr

from subtitler import add_subtitles
from subtitler.transcribe import LANGUAGES, UZ_MODEL

OUT_DIR = Path(tempfile.gettempdir()) / "video_subtitle_outputs"
OUT_DIR.mkdir(exist_ok=True)
DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "small")


def process(video_path, language_name, model_size, max_words, styled):
    if not video_path:
        raise gr.Error("Please upload a video first.")
    language = LANGUAGES[language_name]
    if language == "uz" and not UZ_MODEL:
        gr.Warning("No fine-tuned Uzbek model configured (UZ_WHISPER_MODEL); "
                   "using stock Whisper, which is weak for Uzbek.")
    out = OUT_DIR / f"{uuid.uuid4().hex[:8]}_subtitled.mp4"
    try:
        res = add_subtitles(video_path, out, model_size=model_size, language=language,
                            max_words=int(max_words), styled=styled)
    except ValueError as e:
        raise gr.Error(str(e))
    return str(res.video), str(res.srt), res.text


with gr.Blocks(title="Video Subtitler") as demo:
    gr.Markdown("# Video Subtitler\nUpload a short video (up to 3 minutes) and get it back with "
                "English or Uzbek subtitles.")
    with gr.Row():
        with gr.Column():
            video_in = gr.Video(label="Input video", sources=["upload"])
            lang = gr.Radio(list(LANGUAGES), value="English", label="Language spoken in the video")
            model = gr.Dropdown(["tiny", "base", "small", "medium"], value=DEFAULT_MODEL,
                                label="Whisper model for English (Uzbek uses the fine-tuned model)")
            words = gr.Slider(1, 8, value=4, step=1, label="Max words per caption")
            styled = gr.Checkbox(value=True, label="Reels style (big bold centred text)")
            btn = gr.Button("Add subtitles", variant="primary")
        with gr.Column():
            video_out = gr.Video(label="Subtitled video")
            srt_out = gr.File(label="Subtitle file (.srt)")
            text_out = gr.Textbox(label="Transcript", lines=4)
    btn.click(process, [video_in, lang, model, words, styled], [video_out, srt_out, text_out])

if __name__ == "__main__":
    demo.queue(max_size=10).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
    )