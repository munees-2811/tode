# syntax=docker/dockerfile:1.6
#
# Video Annotation Tool — container image
#
# Build:
#   docker build -t video-annotation .
#
# Run (headless / CI — no GUI):
#   docker run --rm -it video-annotation python -c "from core import YOLOAnnotator"
#
# Run (Linux desktop with X11 forwarding):
#   xhost +local:docker
#   docker run --rm -it \
#       -e DISPLAY=$DISPLAY \
#       -v /tmp/.X11-unix:/tmp/.X11-unix \
#       -v $(pwd)/output:/app/output \
#       -v $HOME/Documents/labeled_img:/root/Documents/labeled_img \
#       video-annotation
#
# Run with GPU (NVIDIA, requires nvidia-container-toolkit on host):
#   docker run --gpus all --rm -it -e DISPLAY=$DISPLAY \
#       -v /tmp/.X11-unix:/tmp/.X11-unix video-annotation

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    OPENCV_FFMPEG_CAPTURE_OPTIONS="threads;1"

# System dependencies:
#   libgl1, libglib2.0-0       — OpenCV runtime
#   libxcb1, libxext6, libsm6  — Tk + cv2 imshow on X11
#   tk, python3-tk             — Tkinter UI
#   ffmpeg                     — yt-dlp merging (also satisfies bundled fallback)
RUN apt-get update && apt-get install --no-install-recommends -y \
        libgl1 \
        libglib2.0-0 \
        libxcb1 \
        libxext6 \
        libsm6 \
        libxrender1 \
        tk \
        python3-tk \
        ffmpeg \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Layer 1: dependency cache.  Touching code shouldn't reinstall packages.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Layer 2: application code
COPY . .

# Persisted state — annotations + frame cache + downloaded videos
VOLUME ["/app/output", "/root/Documents/labeled_img"]

CMD ["python", "main.py"]
