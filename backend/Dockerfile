# Rally Predictor backend image.
#
# Build context MUST be the repo root (so the pipeline files + models/ are
# available), with this file passed explicitly:
#
#   docker build -f backend/Dockerfile -t rally-predictor-backend .
#
# CPU image by default.  For GPU inference swap the base for an NVIDIA CUDA
# runtime (e.g. nvidia/cuda:12.4.1-runtime-ubuntu22.04 + python3) and run with
# `--gpus all`; Ultralytics will then use the GPU automatically.  See
# backend/README.md.
FROM python:3.12-slim AS base

# ffmpeg is required by utilities.create_highlights_ffmpeg; libgl/glib by OpenCV.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIPELINE_DIR=/app/pipeline \
    LOCAL_STORAGE_DIR=/data/rally-media

WORKDIR /app

# Install Python deps first for layer caching.
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Pipeline source + model weights — the entire pipeline/ folder.
COPY pipeline/ /app/pipeline/

# Backend application.
COPY backend/app /app/app

EXPOSE 8000

# Default command runs the API; the worker overrides it in compose / ECS.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
