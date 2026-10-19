"""Download Uzbek speech data and write train / validation / test manifests.

Sources:
  * USC  - ISSAI Uzbek Speech Corpus, ~105 h, 958 speakers, MIT license
           https://huggingface.co/datasets/issai/Uzbek_Speech_Corpus
  * FLEURS uz_uz - Google, ~10 h of read Wikipedia sentences
           https://huggingface.co/datasets/google/fleurs

Output: <out>/manifests/{train,validation,test}.csv  with columns  path,text,source,duration

    python finetune/prepare_data.py --out /kaggle/temp/uzdata
"""
from __future__ import annotations

import argparse
import csv
import sys
import tarfile
from collections import Counter
from pathlib import Path

import soundfile as sf

from uz_text import has_cyrillic, normalize_uz  # noqa: E402

SPLIT_ALIASES = {"train": "train", "dev": "validation", "valid": "validation",
                 "validation": "validation", "test": "test"}


def _keep(text: str, duration: float, max_sec: float) -> bool:
    # Whisper sees 30 s windows; skip empty, too long, or Cyrillic-script transcripts
    return bool(text) and 0.3 < duration <= max_sec and not has_cyrillic(text)


def prepare_usc(out: Path, max_sec: float) -> list[dict]:
    from huggingface_hub import hf_hub_download

    usc_dir = out / "usc"
    marker = usc_dir / ".extracted"
    if not marker.exists():
        print("Downloading USC (~9.5 GB)...")
        tar_path = hf_hub_download("issai/Uzbek_Speech_Corpus", "ISSAI_USC.tar.gz",
                                   repo_type="dataset", local_dir=out / "downloads")
        print("Extracting...")
        usc_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path) as tar:
            kwargs = {"filter": "data"} if sys.version_info >= (3, 12) else {}
            tar.extractall(usc_dir, **kwargs)
        marker.touch()
        Path(tar_path).unlink(missing_ok=True)  # free disk space

    rows, skipped = [], Counter()
    for wav in usc_dir.rglob("*.wav"):
        split = next((SPLIT_ALIASES[p.lower()] for p in wav.parts if p.lower() in SPLIT_ALIASES), None)
        txt = wav.with_suffix(".txt")
        if split is None or not txt.exists():
            skipped["no split or transcript"] += 1
            continue
        text = normalize_uz(txt.read_text(encoding="utf-8"))
        duration = sf.info(str(wav)).duration
        if not _keep(text, duration, max_sec):
            skipped["filtered"] += 1
            continue
        rows.append({"path": str(wav), "text": text, "source": "usc",
                     "split": split, "duration": round(duration, 2)})
    print(f"USC: kept {len(rows)} utterances, skipped {dict(skipped)}")
    return rows


def prepare_fleurs(out: Path, max_sec: float) -> list[dict]:
    from datasets import Audio, load_dataset

    rows = []
    for split in ["train", "validation", "test"]:
        ds = load_dataset("google/fleurs", "uz_uz", split=split)
        ds = ds.cast_column("audio", Audio(decode=False))  # raw bytes; we write wav files ourselves
        split_dir = out / "fleurs" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for i, ex in enumerate(ds):
            path = split_dir / f"{ex['id']}_{i}.wav"
            if not path.exists():
                audio = ex["audio"]
                if audio.get("bytes"):
                    path.write_bytes(audio["bytes"])
                else:
                    path.write_bytes(Path(audio["path"]).read_bytes())
            text = normalize_uz(ex["transcription"])
            duration = sf.info(str(path)).duration
            if _keep(text, duration, max_sec):
                rows.append({"path": str(path), "text": text, "source": "fleurs",
                             "split": split, "duration": round(duration, 2)})
    print(f"FLEURS: kept {len(rows)} utterances")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/uz", help="where audio + manifests go (needs ~25 GB)")
    ap.add_argument("--max-sec", type=float, default=30.0)
    ap.add_argument("--skip-usc", action="store_true", help="FLEURS only (small, quick test)")
    ap.add_argument("--skip-fleurs", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    rows = []
    if not args.skip_usc:
        rows += prepare_usc(out, args.max_sec)
    if not args.skip_fleurs:
        rows += prepare_fleurs(out, args.max_sec)

    man_dir = out / "manifests"
    man_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "validation", "test"]:
        part = [r for r in rows if r["split"] == split]
        with open(man_dir / f"{split}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["path", "text", "source", "duration"])
            w.writeheader()
            for r in part:
                w.writerow({k: r[k] for k in w.fieldnames})
        hours = sum(r["duration"] for r in part) / 3600
        print(f"{split:<10} {len(part):>7} utterances  {hours:6.1f} h")
    print(f"Manifests written to {man_dir}")


if __name__ == "__main__":
    main()