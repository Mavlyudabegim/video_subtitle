# 🎬 Uzbek & English Video Subtitler

Automatic subtitles for short videos (Reels, Shorts, TikToks) in **Uzbek** and **English**.
Upload a video → speech is transcribed with Whisper-based models → styled, word-timed captions
are burned into the video → download the result (plus an `.srt` file).

The project also explores a harder question: **how well does speech recognition work for
Uzbek, a low-resource language, on real social-media videos with regional dialects and
Uzbek–Russian code-switching?** It includes fine-tuning experiments, a model benchmark,
a labeling tool for building a dialect test set, and Uzbek-specific text normalization.


Demo video:

https://github.com/user-attachments/assets/7d3f9e90-1028-4406-bd4f-6901c129e76a


[Live demo (Streamlit)](https://videosubtitle.streamlit.app/)

---

## Contents

- [Features](#features)
- [How it works](#how-it-works)
- [Results](#results)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Uzbek speech recognition: experiments](#uzbek-speech-recognition-experiments)
- [Building a dialect / code-switching dataset](#building-a-dialect--code-switching-dataset)
- [Deployment](#deployment)
- [Configuration](#configuration)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Acknowledgements](#acknowledgements)

---

## Features

- **Two languages:** English (OpenAI Whisper) and Uzbek (fine-tuned Whisper models).
- **Reels-style captions:** short chunks of 1–8 words, big bold text with outline, placed
  above the app UI area; size scales with the video resolution.
- **Word-level timing:** captions follow the speech closely (word timestamps + voice
  activity detection to skip silence and music).
- **Uzbek-aware text handling:**
  - all apostrophe variants (`ʻ ʼ ‘ ’ \``) unified, so `o'`, `g'` and the glottal stop are consistent;
  - words split by the tokenizer at apostrophes (`to 'yliyat`) are merged back (`to'yliyat`);
  - Cyrillic (Uzbek Cyrillic or Russian words) is transliterated to Uzbek Latin (`просто → prosto`).
- **Outputs:** video with burned-in subtitles (`.mp4`) and a subtitle file (`.srt`).
- **Three interfaces:** command line, Gradio web app, Streamlit web app.
- **Runs on CPU:** models are converted to CTranslate2 (faster-whisper) with int8 inference.
- **Tested:** unit tests for caption grouping, subtitle formats, Uzbek normalization and audio loading.

---

## How it works

```
 video ──► 1. Read        ffprobe: size, rotation, duration, audio track
       ──► 2. Audio       ffmpeg → mono 16 kHz WAV
       ──► 3. Transcribe  faster-whisper (word timestamps, VAD)
                          English: Whisper  ·  Uzbek: fine-tuned Whisper (rubaistt)
       ──► 4. Captions    words → short caption chunks (max words / chars / pauses)
                          → .srt  and styled .ass (font, outline, position)
       ──► 5. Render      ffmpeg + libass burns subtitles into the frames → .mp4
```

| Step | Module | Key details |
|---|---|---|
| Read & audio | `subtitler/media.py` | handles phone-video rotation; optional downscaling for small servers |
| Transcribe | `subtitler/transcribe.py` | model chosen per language; audio passed as a NumPy array (avoids PyAV decode issues) |
| Captions | `subtitler/captions.py` | new caption on: word limit, character limit, pause > 0.6 s, or sentence end |
| Pipeline | `subtitler/pipeline.py` | `add_subtitles(input, output, language=..., ...)` runs everything |

---

## Results

### 1. Fine-tuning Whisper-base for Uzbek (on a laptop)

Whisper-base fine-tuned on **FLEURS Uzbek** (~10 h of read speech, 1,000 steps, ~36 min on an
Apple-Silicon MacBook GPU):

| Model | Test set | WER ↓ | CER ↓ |
|---|---|---|---|
| openai/whisper-base (original) | FLEURS uz test (200 clips) | 145.2 % | 97.6 % |
| **whisper-base + my fine-tune** | FLEURS uz test (200 clips) | **62.9 %** | **20.1 %** |
| openai/whisper-base (original) | my reels (11 clips)* | 175.7 % | 130.2 % |
| **whisper-base + my fine-tune** | my reels (11 clips)* | **42.2 %** | **12.0 %** |

\* Preliminary: very small test set, and the reference transcripts were corrected from model
drafts, which biases results towards the fine-tuned model. See [Limitations](#limitations).

WER above 100 % means the original model produced more wrong words than the reference has —
it mostly hallucinated instead of transcribing Uzbek.

### 2. Comparing open Uzbek models on real videos

| Model | Size | Training data (as reported by authors) | On my videos |
|---|---|---|---|
| openai/whisper-base | 74 M | multilingual, very little Uzbek | unusable |
| whisper-base, my fine-tune | 74 M | FLEURS uz (~10 h) | readable, many errors |
| BlueRaccoon/whisper-small-uz | 244 M | Common Voice 11 + FLEURS | weak |
| navai-uz/whisper-base-uzbek | 74 M | ~750 h (Common Voice + FLEURS) | better, still errors |
| Abduqayum/whisper-uzbek-medium-callcenter | 769 M | ~780 h (audiobooks, podcasts) | good |
| **islomov/rubaistt_v2_medium** ← used in the app | 769 M | ~475 h incl. Tashkent-dialect podcasts | **best** |

The app uses **rubaistt_v2_medium** because it gave the best subtitles on real Uzbek videos.
Even the best model still struggles with **Samarkand-dialect speech** and **Russian words
inside Uzbek sentences**, which motivated the dataset work below.

---

## Project structure

```
video_subtitle/
├── subtitler/                 # the core library
│   ├── media.py               # ffprobe / ffmpeg: read video, extract audio, burn subtitles
│   ├── transcribe.py          # faster-whisper, model selection, Uzbek word fixes
│   ├── captions.py            # words → caption chunks → .srt / styled .ass
│   └── pipeline.py            # add_subtitles(): the full pipeline
├── app.py                     # Gradio web app (local use / public share link)
├── cli.py                     # command-line tool
├── streamlit_app/
│   ├── app.py                 # Streamlit web app (free hosting)
│   └── requirements.txt
├── finetune/                  # Uzbek ASR experiments
│   ├── prepare_data.py        # download USC + FLEURS → train/validation/test manifests
│   ├── train.py               # fine-tune Whisper (Hugging Face Seq2SeqTrainer)
│   ├── evaluate.py            # WER / CER per data source or dialect group
│   ├── export_ct2.py          # convert to CTranslate2 (float16 / int8) + upload to the Hub
│   ├── label_tool.py          # segment videos → review drafts in a web app → export dataset
│   ├── hf_to_manifest.py      # sample any Hugging Face audio dataset (replay data)
│   ├── uz_text.py             # Uzbek normalization (apostrophes, Cyrillic → Latin)
│   ├── data.py                # dataset + collator (features computed on the fly)
│   └── requirements.txt
├── tests/                     # pytest unit tests
├── requirements.txt           # app dependencies
├── packages.txt               # system packages for cloud hosting (ffmpeg, fonts)
├── Dockerfile                 # container build (models baked into the image)
└── conftest.py
```

---

## Installation

**Requirements:** Python 3.10+ (3.11 recommended) and **ffmpeg built with libass**
(needed to draw subtitles).

```bash
git clone https://github.com/Mavlyudabegim/video_subtitle.git
cd video_subtitle

python3.11 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt pytest
pytest                               # all tests should pass
```

**ffmpeg:**

| OS | Command |
|---|---|
| macOS | `brew install ffmpeg-full` then `export PATH="$(brew --prefix ffmpeg-full)/bin:$PATH"` (the standard Homebrew `ffmpeg` has no libass) |
| Ubuntu / Debian | `sudo apt install ffmpeg` |
| Windows | `winget install Gyan.FFmpeg` |

Check: `ffmpeg -hide_banner -filters | grep " ass "` should print one line.

**Uzbek model** (CTranslate2 version of rubaistt, ~1.5 GB, downloaded automatically on first use):

```bash
export UZ_WHISPER_MODEL=Mavlyudabegim/rubaistt-medium-ct2
```

---

## Usage

### Command line

```bash
# English
python cli.py my_video.mp4 --model small

# Uzbek
python cli.py my_video.mp4 --lang uz -o my_video_uz.mp4

# options
python cli.py my_video.mp4 --lang uz --words 3 --plain   # 3 words per caption, plain SRT style
```

Output: `my_video_subtitled.mp4` and `my_video_subtitled.srt` next to the input.

### Python

```python
from subtitler import add_subtitles

res = add_subtitles("reel.mp4", "reel_subtitled.mp4", language="uz", max_words=4)
print(res.text)      # full transcript
print(res.srt)       # path to the .srt file
```

### Web apps

```bash
python app.py                          # Gradio → http://localhost:7860
GRADIO_SHARE=1 python app.py           # + temporary public link (while your computer runs it)
streamlit run streamlit_app/app.py     # Streamlit → http://localhost:8501
```

---

## Uzbek speech recognition: experiments

All scripts are in `finetune/` (install with `pip install -r finetune/requirements.txt`).

### Data

| Dataset | Size | Notes |
|---|---|---|
| [ISSAI Uzbek Speech Corpus (USC)](https://huggingface.co/datasets/issai/Uzbek_Speech_Corpus) | ~105 h, 958 speakers | MIT license, read speech |
| [Google FLEURS `uz_uz`](https://huggingface.co/datasets/google/fleurs) | ~10 h | read Wikipedia sentences |
| Own labeled reels | small, growing | spontaneous speech, dialects, Russian words |

### Fine-tuning a Whisper model

```bash
# 1. data (FLEURS only for a quick run; drop --skip-usc for USC + FLEURS)
python finetune/prepare_data.py --out data/uz --skip-usc

# 2. baseline
python finetune/evaluate.py --data data/uz/manifests/test.csv \
    --models openai/whisper-base --max-samples 200

# 3. train (Apple-Silicon GPU, CUDA GPU, or CPU)
python finetune/train.py --data data/uz/manifests --out runs/whisper-base-uz \
    --base-model openai/whisper-base --steps 1000 --batch-size 8

# 4. compare
python finetune/evaluate.py --data data/uz/manifests/test.csv \
    --models openai/whisper-base runs/whisper-base-uz --max-samples 200

# 5. use in the app
python finetune/export_ct2.py --model runs/whisper-base-uz --out models/whisper-base-uz-ct2
export UZ_WHISPER_MODEL=models/whisper-base-uz-ct2
```

Training details: full fine-tuning with `Seq2SeqTrainer`, log-mel features computed on the fly
(no huge feature cache), light augmentation (random gain + noise), gradient checkpointing,
validation WER/CER during training. Options for larger models on a free T4 GPU:
`--freeze-encoder` (train only the decoder) and `--optim adafactor` (less memory).

### Adapting an already fine-tuned model (continued fine-tuning)

`train.py` can start from any Whisper checkpoint, e.g. rubaistt, and mix in extra data:

```bash
python finetune/train.py --data data/uz/manifests \
    --base-model islomov/rubaistt_v2_medium --out runs/rubaistt-dialect \
    --extra-train data/replay/replay_podcasts.csv data/my_reels/manifests/train.csv \
    --steps 600 --batch-size 4 --grad-accum 4 --lr 3e-6 \
    --freeze-encoder --optim adafactor
```

- **Replay data** (`hf_to_manifest.py`) mixes in a sample of the original model's training
  domains so it does not forget standard Uzbek (catastrophic forgetting).
- A **small learning rate** and few steps adjust the model instead of overwriting it.
- `--keep-style` keeps capitals and punctuation in the labels, matching rubaistt's output style.

---

## Building a dialect / code-switching dataset

Public Uzbek datasets are mostly read, standard speech. To measure (and later improve)
performance on real reels, `finetune/label_tool.py` turns videos into a labeled dataset:

```bash
# 1. cut videos into speech clips (voice activity detection) + draft transcripts
python finetune/label_tool.py segment videos/ --out data/my_reels

# 2. correct drafts in a small web app (http://localhost:7861)
python finetune/label_tool.py review --out data/my_reels

# 3. export train/test manifests, split BY VIDEO (no leakage between train and test)
python finetune/label_tool.py export --out data/my_reels

# 4. WER per group: standard, tashkent, samarkand, ... and "+ru" (Russian words present)
python finetune/evaluate.py --data data/my_reels/manifests/test.csv --models <model> ...
```

**Labeling conventions**

- Uzbek Latin only; Russian words in Latin as pronounced (`prosto`, `koroche`) — Cyrillic is
  converted automatically.
- Dialect words are written **as spoken**, not "corrected" to literary Uzbek.
- Each clip is tagged with a dialect (standard / tashkent / samarkand / fergana / khorezm / other)
  and whether it contains Russian words.
- Test videos should be transcribed **from scratch** (without model drafts) to avoid bias.

---

## Deployment

### Streamlit Community Cloud (free)

1. Push the repository to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io): **Create app** → repository, branch `main`,
   main file `streamlit_app/app.py`, **Python 3.11** (Advanced settings).
3. Optional: set `UZ_WHISPER_MODEL` under **Settings → Secrets**.

The free tier has little RAM, so the Streamlit app is configured to be light:

- int8 version of the Uzbek model, 2 CPU threads, greedy decoding;
- one video processed at a time (a lock shared by all visitors);
- uploads limited to 30 MB; videos downscaled to max 1280 px height before encoding;
- temporary files deleted after each job — **uploaded videos are not stored**.

### Docker

```bash
docker build -t video-subtitler .
docker run -p 7860:7860 video-subtitler
```

The image includes ffmpeg (with libass) and bakes the models in, so containers start without
downloading ~2 GB.

### Converting models for deployment

```bash
# float16 (best for local use)
python finetune/export_ct2.py --model models/rubaistt-medium --out models/rubaistt-medium-ct2

# int8 (half the size, much less RAM on CPU servers) + upload to the Hugging Face Hub
python finetune/export_ct2.py --model models/rubaistt-medium --out models/rubaistt-medium-int8-ct2 \
    --quantization int8 --upload <hf-username>/rubaistt-medium-int8-ct2
```

---

## Configuration

All settings are environment variables (or Streamlit secrets); defaults work locally.

| Variable | Default | Purpose |
|---|---|---|
| `UZ_WHISPER_MODEL` | – | Uzbek model: Hub id or local folder (CTranslate2 format) |
| `WHISPER_MODEL` | `small` | default English model in the Gradio app |
| `WHISPER_THREADS` | all cores | CPU threads for speech recognition |
| `WHISPER_BEAM` | `5` | beam size (`1` = greedy: faster, less memory) |
| `SUBTITLE_MAX_HEIGHT` | off | downscale videos taller than this before encoding |
| `FFMPEG_THREADS` | auto | encoder threads (fewer = less memory) |
| `GRADIO_SHARE` | off | `1` = create a temporary public Gradio link |
| `PORT` | `7860` | port for the Gradio app |

Maximum video length is 3 minutes (`MAX_DURATION_SEC` in `subtitler/pipeline.py`).

---

## Limitations

- **Small, preliminary evaluation.** The real-video test set has only 11 clips and its labels
  were corrected from model drafts; the per-dialect numbers are not yet reliable.
- **Dialects and code-switching.** All tested models still make many errors on Samarkand-dialect
  speech and on Russian words inside Uzbek sentences.
- **Subtitle style for Uzbek fine-tune.** Models trained on normalized text output lowercase
  text without punctuation (the reels caption style uses capitals anyway).
- **Free hosting.** The online demo runs on a small shared CPU: slow (minutes per reel), one user
  at a time, and it may restart under memory pressure. For best quality, run locally.
- **Read-speech training data.** USC and FLEURS are mostly clean read speech, unlike noisy,
  spontaneous social-media audio.

---

## Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper) and
  [faster-whisper](https://github.com/SYSTRAN/faster-whisper) / CTranslate2
- [islomov/rubaistt_v2_medium](https://huggingface.co/islomov/rubaistt_v2_medium) (Apache-2.0) —
  Uzbek model used in the app; training pipeline:
  [Islomov49/rubaistt_v2-open-sourced](https://github.com/Islomov49/rubaistt_v2-open-sourced)
- Models compared: [navai-uz/whisper-base-uzbek](https://huggingface.co/navai-uz/whisper-base-uzbek),
  [Abduqayum/whisper-uzbek-medium-callcenter](https://huggingface.co/Abduqayum/whisper-uzbek-medium-callcenter),
  [BlueRaccoon/whisper-small-uz](https://huggingface.co/BlueRaccoon/whisper-small-uz)
- Datasets: [ISSAI Uzbek Speech Corpus](https://huggingface.co/datasets/issai/Uzbek_Speech_Corpus),
  [Google FLEURS](https://huggingface.co/datasets/google/fleurs)
- ffmpeg and libass for video processing and subtitle rendering

Model weights and datasets keep their own licenses.

---

## License

The code in this repository is released under the MIT License — see [`LICENSE`](LICENSE).
