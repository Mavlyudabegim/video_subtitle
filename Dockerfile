FROM python:3.11-slim

# ffmpeg (Debian build includes libass, needed to burn subtitles) + a font for the captions
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the models into the image, so a cold start does not download ~2 GB every time.
#   Uzbek:   your CTranslate2 copy of rubaistt (on your Hugging Face account)
#   English: stock Whisper (faster-whisper format)
ARG UZ_MODEL_REPO=Mavlyudabegim/rubaistt-medium-ct2
ARG EN_MODEL=small
ENV HF_HOME=/models/hf
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${UZ_MODEL_REPO}', local_dir='/models/uz')" \
    && python -c "from faster_whisper import WhisperModel; WhisperModel('${EN_MODEL}', device='cpu', compute_type='int8')"

ENV UZ_WHISPER_MODEL=/models/uz \
    WHISPER_MODEL=${EN_MODEL} \
    HF_HUB_OFFLINE=1 \
    GRADIO_ANALYTICS_ENABLED=False

COPY app.py cli.py ./
COPY subtitler ./subtitler

# Cloud Run sets $PORT (8080); app.py reads it
EXPOSE 8080
CMD ["python", "app.py"]