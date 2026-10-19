"""Build your own Uzbek dataset from real videos: dialects + Russian words.

Three steps:

1. segment  - cut videos into short speech clips and write draft transcripts with a model
       python finetune/label_tool.py segment videos/ --out data/my_reels

2. review   - correct the drafts in a small web app (listen, fix text, pick dialect)
       python finetune/label_tool.py review --out data/my_reels      # http://localhost:7861

3. export   - make train/test manifests (split BY VIDEO, so test videos are never trained on)
       python finetune/label_tool.py export --out data/my_reels --test-share 0.3

Then:
   python finetune/evaluate.py --data data/my_reels/manifests/test.csv --models ...   # WER per dialect
   python finetune/train.py --data data/uz/manifests --extra-train data/my_reels/manifests/train.csv ...

Spelling rules for corrections (keep them consistent!):
   * Uzbek Latin only. Russian words in Latin as pronounced: prosto, voobshe, koroche, ladno
     (if you type Cyrillic, it is transliterated automatically on export)
   * Write dialect words AS SPOKEN, do not "fix" them to literary Uzbek
   * No need for capitals or punctuation; they are removed anyway
   * Unclear speech / music only / another language -> tick "skip"
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from uz_text import normalize_uz  # noqa: E402

SR = 16_000
MEDIA_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4a", ".mp3", ".wav", ".ogg", ".aac"}
DIALECTS = ["standard", "tashkent", "samarkand", "fergana", "khorezm", "other"]
FIELDS = ["id", "video", "start", "end", "duration", "path", "draft", "text",
          "dialect", "russian", "skip", "status"]


# ---------------------------------------------------------------- CSV helpers

def labels_path(out: Path) -> Path:
    return out / "labels.csv"


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written file


# ---------------------------------------------------------------- 1. segment

def group_speech(chunks: list[dict], max_sec: float, max_gap: float) -> list[tuple[float, float]]:
    """Merge VAD speech chunks (in samples) into clips of at most `max_sec` seconds."""
    clips: list[list[float]] = []
    for c in chunks:
        start, end = c["start"] / SR, c["end"] / SR
        if clips and start - clips[-1][1] <= max_gap and end - clips[-1][0] <= max_sec:
            clips[-1][1] = end
        else:
            clips.append([start, end])
    return [(s, e) for s, e in clips if e - s >= 1.0]


def cmd_segment(args) -> None:
    from faster_whisper import WhisperModel
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    from subtitler.media import extract_audio

    out = Path(args.out)
    clip_dir = out / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(labels_path(out))
    done_videos = {r["video"] for r in rows}

    inputs = [Path(p) for p in args.inputs]
    videos = sorted(f for p in inputs for f in ([p] if p.is_file() else p.rglob("*"))
                    if f.suffix.lower() in MEDIA_EXT)
    if not videos:
        sys.exit("No video/audio files found.")

    model_name = args.model or os.getenv("UZ_WHISPER_MODEL") or "small"
    print(f"Draft model: {model_name}")
    model = WhisperModel(model_name, device="auto", compute_type="int8")
    vad = VadOptions(max_speech_duration_s=args.max_sec, min_silence_duration_ms=400, speech_pad_ms=200)

    for video in videos:
        vid = video.stem
        if vid in done_videos:
            print(f"skip {video.name} (already segmented)")
            continue
        wav = out / "audio" / f"{vid}.wav"
        wav.parent.mkdir(exist_ok=True)
        extract_audio(video, wav)  # ffmpeg -> mono 16 kHz
        audio = sf.read(str(wav), dtype="float32")[0]
        clips = group_speech(get_speech_timestamps(audio, vad), args.max_sec, max_gap=0.8)
        print(f"{video.name}: {len(clips)} clips")
        for i, (start, end) in enumerate(clips):
            piece = audio[int(start * SR):int(end * SR)]
            path = clip_dir / f"{vid}_{i:03d}.wav"
            sf.write(path, piece, SR)
            segments, _ = model.transcribe(piece, language="uz", vad_filter=False, beam_size=5)
            draft = " ".join(s.text.strip() for s in segments).strip()
            rows.append({
                "id": f"{vid}_{i:03d}", "video": vid, "start": f"{start:.2f}", "end": f"{end:.2f}",
                "duration": f"{end - start:.2f}", "path": str(path.resolve()), "draft": draft,
                "text": "", "dialect": "", "russian": "", "skip": "", "status": "todo",
            })
        write_rows(labels_path(out), rows)  # save after every video
    print(f"Saved {len(rows)} clips to {labels_path(out)}. Next: review")


# ---------------------------------------------------------------- 2. review (web app)

def cmd_review(args) -> None:
    import gradio as gr

    path = labels_path(Path(args.out))
    rows = read_rows(path)
    if not rows:
        sys.exit(f"No clips in {path}. Run 'segment' first.")

    def first_todo() -> int:
        return next((i for i, r in enumerate(rows) if r["status"] != "done"), 0)

    def show(i: int):
        i = max(0, min(i, len(rows) - 1))
        r = rows[i]
        done = sum(r_["status"] == "done" for r_ in rows)
        info = (f"**Clip {i + 1} / {len(rows)}** · `{r['id']}` · {float(r['duration']):.1f} s · "
                f"labeled: {done}/{len(rows)}")
        return (i, info, r["path"], r["draft"], r["text"] or r["draft"],
                r["dialect"] or None, r["russian"] == "1", r["skip"] == "1")

    def save(i, text, dialect, russian, skip):
        r = rows[i]
        if not skip and (not text.strip() or not dialect):
            gr.Warning("Fill in the text and choose a dialect (or tick skip).")
            return show(i)
        r.update(text=text.strip(), dialect=dialect or "", russian="1" if russian else "0",
                 skip="1" if skip else "0", status="done")
        write_rows(path, rows)
        return show(i + 1)

    with gr.Blocks(title="Uzbek labeling") as app:
        gr.Markdown("## Uzbek speech labeling\nListen, correct the text **as spoken** (Latin; "
                    "Russian words in Latin too: *prosto, koroche*), choose the dialect, save.")
        idx = gr.State(first_todo())
        info = gr.Markdown()
        audio = gr.Audio(type="filepath", label="Clip", interactive=False)
        draft = gr.Textbox(label="Model draft (read only)", interactive=False)
        text = gr.Textbox(label="Correct transcript", lines=3)
        with gr.Row():
            dialect = gr.Radio(DIALECTS, label="Dialect")
            russian = gr.Checkbox(label="Contains Russian words")
            skip = gr.Checkbox(label="Skip (unclear / music / not Uzbek)")
        with gr.Row():
            prev_b = gr.Button("← Previous")
            save_b = gr.Button("Save & next →", variant="primary")
            next_b = gr.Button("Next (no save) →")
            todo_b = gr.Button("Jump to first unlabeled")

        outs = [idx, info, audio, draft, text, dialect, russian, skip]
        app.load(lambda i: show(i), idx, outs)
        save_b.click(save, [idx, text, dialect, russian, skip], outs)
        prev_b.click(lambda i: show(i - 1), idx, outs)
        next_b.click(lambda i: show(i + 1), idx, outs)
        todo_b.click(lambda: show(first_todo()), None, outs)

    clip_dir = str((Path(args.out) / "clips").resolve())
    app.launch(server_port=args.port, allowed_paths=[clip_dir])


# ---------------------------------------------------------------- 3. export

def cmd_export(args) -> None:
    out = Path(args.out)
    rows = [r for r in read_rows(labels_path(out)) if r["status"] == "done" and r["skip"] != "1"]
    if not rows:
        sys.exit("No labeled clips yet. Run 'review' first.")

    # Split by video: clips from one video are very similar, so mixing them
    # between train and test would make the test score look better than it is.
    videos = sorted({r["video"] for r in rows})
    random.Random(args.seed).shuffle(videos)
    n_test = max(1, round(len(videos) * args.test_share)) if len(videos) > 1 else 0
    test_videos = set(videos[:n_test])

    man_dir = out / "manifests"
    man_dir.mkdir(exist_ok=True)
    splits = {"train": [], "test": []}
    for r in rows:
        text = normalize_uz(r["text"])
        if not text:
            continue
        # "source" = group used by evaluate.py for per-group WER, e.g. "samarkand+ru"
        source = r["dialect"] + ("+ru" if r["russian"] == "1" else "")
        split = "test" if r["video"] in test_videos else "train"
        splits[split].append({"path": r["path"], "text": text, "source": source,
                              "duration": r["duration"]})

    for name, part in splits.items():
        with open(man_dir / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["path", "text", "source", "duration"])
            w.writeheader()
            w.writerows(part)
        hours = sum(float(p["duration"]) for p in part) / 3600
        groups = {}
        for p in part:
            groups[p["source"]] = groups.get(p["source"], 0) + 1
        print(f"{name:<5} {len(part):>5} clips {hours * 60:6.1f} min  {groups}")
    print(f"Test videos: {sorted(test_videos)}")
    print(f"Manifests in {man_dir}")


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("segment", help="cut videos into clips + draft transcripts")
    s.add_argument("inputs", nargs="+", help="video/audio files or folders")
    s.add_argument("--out", default="data/my_reels")
    s.add_argument("--model", default=None,
                   help="draft model (default: $UZ_WHISPER_MODEL, else 'small')")
    s.add_argument("--max-sec", type=float, default=20.0, help="max clip length")

    r = sub.add_parser("review", help="web app to correct drafts")
    r.add_argument("--out", default="data/my_reels")
    r.add_argument("--port", type=int, default=7861)

    e = sub.add_parser("export", help="labeled clips -> train/test manifests")
    e.add_argument("--out", default="data/my_reels")
    e.add_argument("--test-share", type=float, default=0.3, help="share of VIDEOS kept for testing")
    e.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    {"segment": cmd_segment, "review": cmd_review, "export": cmd_export}[args.cmd](args)


if __name__ == "__main__":
    main()