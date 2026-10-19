"""Compare models on the test sets: WER / CER, split by data source.

    # baseline (original Whisper) vs fine-tuned model
    python finetune/evaluate.py --data /kaggle/temp/uzdata/manifests/test.csv \
        --models openai/whisper-small /kaggle/working/whisper-small-uz

    # short videos, reels: a CSV with columns path,text (path = .wav file)
    python finetune/evaluate.py --data my_reels.csv --models ...
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import jiwer
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.insert(0, str(Path(__file__).parent))
from data import SAMPLE_RATE, load_audio, read_manifest  # noqa: E402
from uz_text import normalize_uz  # noqa: E402


@torch.no_grad()
def transcribe_all(model_id: str, rows: list[dict], batch_size: int) -> tuple[list[str], float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()

    preds, start = [], time.time()
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        feats = processor.feature_extractor([load_audio(r["path"]) for r in batch],
                                            sampling_rate=SAMPLE_RATE, return_tensors="pt")
        out = model.generate(feats.input_features.to(device, dtype),
                             language="uzbek", task="transcribe", max_new_tokens=225)
        preds += processor.batch_decode(out, skip_special_tokens=True)
        print(f"  {model_id}: {min(i + batch_size, len(rows))}/{len(rows)}", end="\r")
    print()
    return preds, time.time() - start


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV with path,text[,source]")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--save", default="eval_results.csv", help="per-utterance predictions")
    args = ap.parse_args()

    rows = read_manifest(args.data, args.max_samples)
    refs = [normalize_uz(r["text"]) for r in rows]
    sources = [r.get("source") or "all" for r in rows]

    all_preds = {}
    print(f"{'model':<45} {'source':<8} {'WER %':>7} {'CER %':>7} {'sec':>7}")
    for model_id in args.models:
        raw, secs = transcribe_all(model_id, rows, args.batch_size)
        preds = [normalize_uz(p) for p in raw]
        all_preds[model_id] = preds
        groups = defaultdict(list)
        for ref, pred, src in zip(refs, preds, sources):
            groups[src].append((ref, pred))
            groups["ALL"].append((ref, pred))
        for src, pairs in sorted(groups.items()):
            r, p = [x for x, _ in pairs], [y for _, y in pairs]
            print(f"{model_id[-45:]:<45} {src:<8} {100 * jiwer.wer(r, p):7.2f} "
                  f"{100 * jiwer.cer(r, p):7.2f} {secs if src == 'ALL' else 0:7.0f}")

    with open(args.save, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "reference", *args.models])
        for i, r in enumerate(rows):
            w.writerow([r["path"], refs[i], *[all_preds[m][i] for m in args.models]])
    print(f"Per-utterance predictions saved to {args.save}")


if __name__ == "__main__":
    main()