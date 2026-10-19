"""Dataset + collator. Audio is loaded and turned into log-mel features on the fly,
so we never store ~100 GB of precomputed features on disk."""
from __future__ import annotations

import csv
import random

import librosa
import numpy as np
import soundfile as sf
import torch

SAMPLE_RATE = 16_000


def read_manifest(path: str, max_rows: int | None = None, seed: int = 0) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if max_rows and len(rows) > max_rows:
        rows = random.Random(seed).sample(rows, max_rows)
    return rows


def load_audio(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio


class ManifestDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        return {"audio": load_audio(r["path"]), "text": r["text"]}


class WhisperCollator:
    def __init__(self, processor, augment: bool = False):
        self.processor = processor
        self.augment = augment

    def _augment(self, audio: np.ndarray) -> np.ndarray:
        # Cheap robustness for real-world reels: random gain + light background noise
        audio = audio * random.uniform(0.6, 1.2)
        if random.random() < 0.5:
            noise_level = random.uniform(0.001, 0.01)
            audio = audio + np.random.randn(len(audio)).astype(np.float32) * noise_level
        return np.clip(audio, -1.0, 1.0)

    def __call__(self, batch: list[dict]) -> dict:
        audios = [self._augment(b["audio"]) if self.augment else b["audio"] for b in batch]
        feats = self.processor.feature_extractor(
            audios, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features

        # Labels = <|startoftranscript|><|uz|><|transcribe|><|notimestamps|> text <|endoftext|>
        labels = self.processor.tokenizer([b["text"] for b in batch], padding=True,
                                          return_tensors="pt")
        ids = labels.input_ids.masked_fill(labels.attention_mask.ne(1), -100)
        # The model adds <|startoftranscript|> itself when shifting labels right
        if (ids[:, 0] == self.processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")).all():
            ids = ids[:, 1:]
        return {"input_features": feats, "labels": ids}