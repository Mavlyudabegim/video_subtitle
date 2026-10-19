"""The full pipeline: video -> read -> transcribe -> captions -> burned-in video."""
from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .captions import group_words, write_ass, write_srt
from .media import burn_subtitles, extract_audio, probe
from .transcribe import transcribe

MAX_DURATION_SEC = 180  # "short videos" for now; raise when you need longer ones


@dataclass
class Result:
    video: Path   # video with burned-in subtitles
    srt: Path     # plain subtitle file (useful for uploading separately)
    text: str     # full transcript


def add_subtitles(
    input_video: str | Path,
    output_video: str | Path,
    model_size: str = "small",
    max_words: int = 4,
    styled: bool = True,
    language: str | None = "en",
) -> Result:
    input_video, output_video = Path(input_video), Path(output_video)

    # 1. Read the video
    info = probe(input_video)
    if not info["has_audio"]:
        raise ValueError("This video has no audio track, so there is nothing to subtitle.")
    if info["duration"] > MAX_DURATION_SEC:
        raise ValueError(f"Video is {info['duration']:.0f}s; the limit is {MAX_DURATION_SEC}s.")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        wav = extract_audio(input_video, tmp / "audio.wav")

        # 2. Speech -> timed words
        words = transcribe(str(wav), model_size=model_size, language=language)
        if not words:
            raise ValueError("No speech was detected in this video.")

        # 3. Words -> captions -> subtitle file
        captions = group_words(words, max_words=max_words)
        srt_tmp = write_srt(captions, tmp / "subs.srt")
        if styled:
            subs = write_ass(captions, tmp / "subs.ass", info["width"], info["height"])
        else:
            subs = srt_tmp

        # 4. Burn subtitles and save
        burn_subtitles(input_video, subs, output_video)
        srt_out = output_video.with_suffix(".srt")
        shutil.copy(srt_tmp, srt_out)

    return Result(video=output_video, srt=srt_out, text=" ".join(w.text for w in words))