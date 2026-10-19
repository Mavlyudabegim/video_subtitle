"""Fine-tune Whisper for Uzbek speech recognition.

    python finetune/train.py --data /kaggle/temp/uzdata/manifests --out /kaggle/working/whisper-small-uz
    # resume after a session timeout:
    python finetune/train.py ... --resume
    # push the result to your Hugging Face account:
    python finetune/train.py ... --push-to-hub <your-username>/whisper-small-uz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jiwer
import torch
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          WhisperForConditionalGeneration, WhisperProcessor)

from data import ManifestDataset, WhisperCollator, read_manifest  # noqa: E402
from uz_text import normalize_uz  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with train/validation.csv")
    ap.add_argument("--out", default="whisper-small-uz")
    ap.add_argument("--base-model", default="openai/whisper-small")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--eval-samples", type=int, default=500, help="validation subset (generation is slow)")
    ap.add_argument("--max-train", type=int, default=None, help="limit train rows (quick tests)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--extra-train", nargs="*", default=[],
                    help="more train CSVs, e.g. your labeled reels (data/my_reels/manifests/train.csv)")
    ap.add_argument("--extra-repeat", type=int, default=3,
                    help="repeat extra data N times so a few hours are not drowned by USC/FLEURS")
    ap.add_argument("--push-to-hub", default=None, metavar="REPO_ID")
    args = ap.parse_args()

    processor = WhisperProcessor.from_pretrained(args.base_model, language="uzbek", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    model.generation_config.language = "uzbek"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.use_cache = False  # needed with gradient checkpointing

    data = Path(args.data)
    train_rows = read_manifest(data / "train.csv", args.max_train)
    for extra in args.extra_train:
        extra_rows = read_manifest(extra)
        print(f"extra: {extra} -> {len(extra_rows)} clips x{args.extra_repeat}")
        train_rows += extra_rows * args.extra_repeat
    train_ds = ManifestDataset(train_rows)
    eval_ds = ManifestDataset(read_manifest(data / "validation.csv", args.eval_samples))
    print(f"train: {len(train_ds)}  eval: {len(eval_ds)}")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        preds = [normalize_uz(t) for t in processor.batch_decode(pred_ids, skip_special_tokens=True)]
        refs = [normalize_uz(t) for t in processor.batch_decode(label_ids, skip_special_tokens=True)]
        pairs = [(r, p) for r, p in zip(refs, preds) if r]
        return {
            "wer": 100 * jiwer.wer([r for r, _ in pairs], [p for _, p in pairs]),
            "cer": 100 * jiwer.cer([r for r, _ in pairs], [p for _, p in pairs]),
        }

    use_gpu = torch.cuda.is_available()
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=min(500, args.steps // 10),
        max_steps=args.steps,
        gradient_checkpointing=True,
        fp16=use_gpu,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,
        logging_steps=25,
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        dataloader_num_workers=2,
        remove_unused_columns=False,  # our dataset returns raw audio, not model inputs
        report_to="none",
        push_to_hub=bool(args.push_to_hub),
        hub_model_id=args.push_to_hub,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=WhisperCollator(processor, augment=True),
        compute_metrics=compute_metrics,
        processing_class=processor,
    )
    # Note: the eval collator also gets augment=True here; that's fine for tracking progress.
    # Use evaluate.py for the clean, final numbers.

    has_ckpt = any(Path(args.out).glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=bool(args.resume and has_ckpt))

    trainer.save_model(args.out)
    processor.save_pretrained(args.out)
    if args.push_to_hub:
        trainer.push_to_hub()
    print(f"Saved fine-tuned model to {args.out}")


if __name__ == "__main__":
    main()