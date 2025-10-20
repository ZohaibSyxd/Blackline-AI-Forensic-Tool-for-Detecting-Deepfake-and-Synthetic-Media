# Use a base image with CUDA and cuDNN pre-installed for PyTorch
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODELS_DIR=/app/backend/models \
    USE_DECORD=0 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/tmp/hf \
    TORCH_HOME=/tmp/torch \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies including Python and FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3.11-venv \
    ffmpeg \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app/backend

# Expose the port
EXPOSE 8080

# Start Uvicorn
CMD ["sh", "-c", "uvicorn backend.src.api_server:app --host 0.0.0.0 --port ${PORT:-8080}"]
