"""read the video (probe + extract audio) and burn subtitles into it with ffmpeg."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


class FFmpegError(RuntimeError):
    pass


def _run(cmd: list[str], cwd: str | Path | None = None) -> str:
    if shutil.which(cmd[0]) is None:
        raise FFmpegError(f"'{cmd[0]}' not found. Install ffmpeg and make sure it is on PATH.")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise FFmpegError(f"Command failed: {' '.join(cmd)}\n{result.stderr[-2000:]}")
    return result.stdout


def probe(video: str | Path) -> dict:
    """Return width, height (as displayed, rotation applied), duration and whether there is audio."""
    out = _run([
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_streams", "-show_format", str(video),
    ])
    data = json.loads(out)
    vstream = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    if vstream is None:
        raise FFmpegError("No video stream found in the file.")

    width, height = int(vstream["width"]), int(vstream["height"])
    # Phone videos are often stored landscape + a rotation flag. ffmpeg auto-rotates,
    # so the subtitle canvas must match the *displayed* size.
    rotation = int(vstream.get("tags", {}).get("rotate", 0) or 0)
    for sd in vstream.get("side_data_list", []):
        if "rotation" in sd:
            rotation = int(sd["rotation"])
    if abs(rotation) % 180 == 90:
        width, height = height, width

    return {
        "width": width,
        "height": height,
        "duration": float(data["format"].get("duration", 0.0)),
        "has_audio": any(s["codec_type"] == "audio" for s in data["streams"]),
    }


def extract_audio(video: str | Path, wav_out: str | Path) -> Path:
    """Extract mono 16 kHz WAV, the format Whisper expects."""
    _run(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(wav_out)])
    return Path(wav_out)


def burn_subtitles(video: str | Path, subs: str | Path, output: str | Path) -> Path:
    """Render subtitles (.ass or .srt) onto the video frames and save an MP4."""
    video, subs, output = Path(video).resolve(), Path(subs).resolve(), Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    flt = "ass" if subs.suffix.lower() == ".ass" else "subtitles"

    # Optional limits for small servers (set as environment variables):
    #   SUBTITLE_MAX_HEIGHT=1280 -> 4K/1080p phone videos are scaled down before encoding
    #   FFMPEG_THREADS=2         -> fewer encoder threads = much less RAM
    # Subtitle size/position stay correct: libass scales the .ass canvas to the frame.
    filters = []
    max_h = int(os.getenv("SUBTITLE_MAX_HEIGHT", "0") or 0)
    if max_h:
        filters.append(f"scale=-2:'min({max_h},ih)'")
    filters.append(f"{flt}={subs.name}")
    threads = os.getenv("FFMPEG_THREADS")

    # Run inside the subtitle's folder and pass only its file name: avoids ffmpeg
    # filter-path escaping problems (spaces, colons, Windows drive letters).
    _run([
        "ffmpeg", "-y", "-i", str(video),
        "-vf", ",".join(filters),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        *(["-threads", threads] if threads else []),
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(output),
    ], cwd=subs.parent)
    return output