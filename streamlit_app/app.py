"""Streamlit version of the subtitle app, for free hosting on Streamlit Community Cloud.

Local test:   streamlit run streamlit_app/app.py
Cloud:        share.streamlit.io -> New app -> main file: streamlit_app/app.py
"""
import os
import sys
import tempfile
import threading
import uuid
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Must be set BEFORE importing subtitler (the Uzbek model is read at import time).
# Override in Streamlit Cloud -> app settings -> Secrets:  UZ_WHISPER_MODEL = "..."
try:
    if "UZ_WHISPER_MODEL" in st.secrets:
        os.environ["UZ_WHISPER_MODEL"] = st.secrets["UZ_WHISPER_MODEL"]
except Exception:  # no secrets file locally
    pass
os.environ.setdefault("UZ_WHISPER_MODEL", "Mavlyudabegim/rubaistt-medium-ct2")

from subtitler import add_subtitles  # noqa: E402
from subtitler.transcribe import LANGUAGES, load_model, resolve_model  # noqa: E402

OUT_DIR = Path(tempfile.gettempdir()) / "subtitle_outputs"
OUT_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Video Subtitler", page_icon="🎬")
st.title("Video Subtitler")
st.write("Upload a short video (up to 3 minutes) and get it back with **Uzbek** or **English** subtitles.")
st.caption("Uzbek model: [islomov/rubaistt_v2_medium](https://huggingface.co/islomov/rubaistt_v2_medium) "
           "(Apache-2.0). Runs on a free CPU, so a 1-minute reel takes a few minutes.")


@st.cache_resource(show_spinner="Loading speech model (first time only)...")
def get_model(name: str):
    return load_model(name)

@st.cache_resource
def job_lock() -> threading.Lock:
    # ONE lock shared by all visitors: the free server has little RAM, so only one
    # video is processed at a time (two at once = out of memory = app crash).
    return threading.Lock()


video = st.file_uploader("Video", type=["mp4", "mov", "m4v", "webm", "mkv"])
col1, col2 = st.columns(2)
language_name = col1.radio("Language spoken in the video", list(LANGUAGES), index=1)
max_words = col2.slider("Max words per caption", 1, 8, 4)
styled = st.checkbox("Reels style (big bold centred text)", value=True)

if video and st.button("Add subtitles", type="primary"):
    lock = job_lock()
    if not lock.acquire(blocking=False):
        st.warning("Another visitor's video is being processed right now. "
                   "Please try again in a minute or two.")
        st.stop()
    try:
        language = LANGUAGES[language_name]
        get_model(resolve_model("small", language))  # load once, shared by all visitors

        job = uuid.uuid4().hex[:8]
        src = OUT_DIR / f"{job}_input{Path(video.name).suffix}"
        src.write_bytes(video.getbuffer())
        out = OUT_DIR / f"{job}_subtitled.mp4"

        with st.spinner("Transcribing and rendering... this can take a few minutes."):
            try:
                res = add_subtitles(src, out, model_size="small", language=language,
                                    max_words=max_words, styled=styled)
            except ValueError as e:
                st.error(str(e))
                st.stop()
            finally:
                src.unlink(missing_ok=True)

        # keep results in memory for this visitor, delete files so the disk stays clean
        video_bytes, srt_bytes = res.video.read_bytes(), res.srt.read_bytes()
        res.video.unlink(missing_ok=True)
        res.srt.unlink(missing_ok=True)
    finally:
        lock.release()

    st.success("Done!")
    st.video(video_bytes)
    c1, c2 = st.columns(2)
    c1.download_button("Download video", video_bytes, file_name="subtitled.mp4", mime="video/mp4")
    c2.download_button("Download .srt", srt_bytes, file_name="subtitles.srt", mime="text/plain")
    st.text_area("Transcript", res.text, height=120)