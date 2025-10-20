FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps for OpenCV/mediapipe runtime
RUN apt-get update \
     && apt-get install -y --no-install-recommends \
         libgl1 \
         libglib2.0-0 \
         ffmpeg \
     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (assumes build context is backend/)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code (everything under backend/ context)
COPY . /app/backend

# Expose the port (informational; Render sets $PORT)
EXPOSE 8000

# Start Uvicorn on the Render-provided PORT (fallback 8000 locally)
CMD ["sh", "-c", "uvicorn backend.src.api_server:app --host 0.0.0.0 --port ${PORT:-8000}"]
