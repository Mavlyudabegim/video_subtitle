"""turn timed words into short caption chunks and write SRT / ASS files."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Word:
    start: float  # seconds
    end: float
    text: str


@dataclass
class Caption:
    start: float
    end: float
    text: str


def group_words(
    words: list[Word],
    max_words: int = 4,
    max_chars: int = 24,
    max_gap: float = 0.6,
) -> list[Caption]:
    """Group words into short, reels-style captions.

    A new caption starts when the current one is full (words/chars), when there is
    a pause longer than `max_gap` seconds, or after sentence-ending punctuation.
    """
    captions: list[Caption] = []
    current: list[Word] = []

    def flush() -> None:
        if current:
            text = " ".join(w.text for w in current)
            captions.append(Caption(current[0].start, current[-1].end, text))
            current.clear()

    for w in words:
        if current:
            candidate = " ".join(x.text for x in current) + " " + w.text
            if (
                len(current) >= max_words
                or len(candidate) > max_chars
                or w.start - current[-1].end > max_gap
                or current[-1].text.endswith((".", "?", "!"))
            ):
                flush()
        current.append(w)
    flush()
    return captions


# ---------- SRT ----------

def _srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(captions: list[Caption], path: str | Path) -> Path:
    lines = []
    for i, c in enumerate(captions, 1):
        lines += [str(i), f"{_srt_time(c.start)} --> {_srt_time(c.end)}", c.text, ""]
    path = Path(path)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------- ASS (styled, reels-like) ----------

def _ass_time(t: float) -> str:
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360_000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", "\\N")


def write_ass(
    captions: list[Caption],
    path: str | Path,
    width: int,
    height: int,
    font: str = "Arial",
    uppercase: bool = True,
) -> Path:
    """Write a styled ASS file: big bold white text, black outline, lower-middle of frame."""
    font_size = max(24, int(height * 0.055))
    outline = max(2, font_size // 12)
    margin_v = int(height * 0.22)  # keep captions above the reels/TikTok UI buttons
    margin_h = int(width * 0.08)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Reels,{font},{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,{outline},0,2,{margin_h},{margin_h},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    for c in captions:
        text = c.text.upper() if uppercase else c.text
        events.append(
            f"Dialogue: 0,{_ass_time(c.start)},{_ass_time(c.end)},Reels,,0,0,0,,{_ass_escape(text)}"
        )
    path = Path(path)
    path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return path