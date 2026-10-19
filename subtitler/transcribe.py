"""Step 2: speech-to-text with word-level timestamps (faster-whisper, free and local)."""
from __future__ import annotations

import os
from functools import lru_cache

from .captions import Word

# Your fine-tuned Uzbek model in CTranslate2 format: a Hub id ("username/whisper-small-uz-ct2")
# or a local folder. Set it as an environment variable (or a Space "Variable" on Hugging Face).
UZ_MODEL = os.getenv("UZ_WHISPER_MODEL", "")

LANGUAGES = {"English": "en", "Uzbek": "uz"}

import wave
import numpy as np

def load_wav(path: str) -> np.ndarray:
    """Read the 16 kHz 16-bit WAV that media.extract_audio() writes, as float32 in [-1, 1].

    We pass this array to faster-whisper instead of a file path, so faster-whisper does
    not decode audio with PyAV (some PyAV versions break it: "unexpected keyword
    argument 'metadata_errors'")."""
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("expected 16-bit PCM WAV")
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if w.getnchannels() > 1:
            data = data.reshape(-1, w.getnchannels()).mean(axis=1)
    return data.astype(np.float32) / 32768.0


def resolve_model(model_size: str, language: str | None) -> str:
    """Pick the fine-tuned model for Uzbek, the stock Whisper model otherwise."""
    if language == "uz" and UZ_MODEL:
        return UZ_MODEL
    return model_size


@lru_cache(maxsize=3)
def load_model(model: str = "small", device: str = "auto", compute_type: str = "int8"):
    """Load once and reuse. `model` = size name (tiny/base/small/medium/large-v3),
    a Hub id of a CTranslate2 model, or a local folder.

    int8 keeps memory low and runs well on CPU (free hosting tiers have no GPU).
    """
    from faster_whisper import WhisperModel

    return WhisperModel(model, device=device, compute_type=compute_type)


def transcribe(audio_path: str, model_size: str = "small", language: str | None = "en") -> list[Word]:
    model = load_model(resolve_model(model_size, language))
    segments, _info = model.transcribe(
        load_wav(audio_path),
        language=language,        # "en", "uz", or None = auto-detect
        word_timestamps=True,     # needed for short, well-timed captions
        vad_filter=True,          # skip silence / music-only parts
    )
    words: list[Word] = []
    for seg in segments:  # generator: transcription actually runs here
        for w in seg.words or []:
            text = w.word.strip()
            if text:
                words.append(Word(start=w.start, end=w.end, text=text))
    return words