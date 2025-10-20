# Cloud Run deployment notes (Deep Learning enabled)

This service runs FastAPI with optional deep learning models (Xception + TimeSformer).

Key points for fast, reliable builds:

- requirements.txt now uses the official PyTorch CPU wheel index and pins `torch==2.2.2` and `torchvision==0.17.2` for Python 3.11. This avoids source builds.
- `decord` is disabled to avoid slow/fragile builds on Python 3.11. We force OpenCV sampling via `USE_DECORD=0`.
- Transformers are set to offline mode by default and caches go to `/tmp/hf` and `/tmp/torch` which are writable on Cloud Run.
- The app writes data/DB to `/tmp/data` on Cloud Run and `backend/data` locally.

Environment variables:

- `PORT=8080` (Cloud Run injects this automatically)
- `DATA_ROOT=/tmp/data` (optional; defaults to `/tmp/data` when `PORT` is present)
- `DATABASE_URL=sqlite:////tmp/data/dev.db` or your managed Postgres URL
- `USE_DECORD=0` (already set in Dockerfile)
- `TRANSFORMERS_OFFLINE=1` (already set in Dockerfile)
- `HF_HOME=/tmp/hf`, `TORCH_HOME=/tmp/torch` (already set)
- `MODELS_DIR=/app/backend/models` (bundled model checkpoints live here)

Cloud Run settings:

- Memory: 2–4 GiB recommended
- CPU: 1–2 vCPUs
- Request timeout: 300–600 seconds for large videos
- Min instances: 0–1 (set to 1 for lower cold-start latency)

Model files:

Ensure `backend/models/xception_best.pth` and `backend/models/timesformer_best.pt` are present in the container. They are included in the repo; the Dockerfile copies the entire `backend/` folder, so they will be available at runtime.

Health checks:

- `/` and `/api/health` return 200 quickly.

Troubleshooting:

- Hit `/api/buildinfo` to check ffmpeg availability and model checkpoint presence. It returns booleans indicating whether checkpoints exist.
- If you see a 500 like `Deep learning model unavailable: ...`, the message usually states the missing dependency or file.
