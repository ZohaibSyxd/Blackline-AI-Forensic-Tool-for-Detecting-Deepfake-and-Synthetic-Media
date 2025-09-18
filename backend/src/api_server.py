"""FastAPI server exposing minimal analysis pipeline for a single uploaded media file.

Endpoints:
  GET /api/health  -> {status: ok}
  POST /api/analyze (multipart/form-data file=...) -> JSON with ingest, validate, probe, metadata summary.

Processing steps for an uploaded file:
  1. Save to a temp upload directory (backend/data/uploads/tmp_<uuid>_<filename>)
  2. Ingest (content-addressed copy -> backend/data/raw/<sha>/<orig_name>) producing audit-like record (not appended yet).
  3. Probe (ffprobe + exiftool) and validate (container + decode dry run).
  4. (Optional) Future: run analyze_metadata on aggregated records – for now we synthesize minimal metadata summary from probe + validate.

Return JSON structure:
  {
    "asset": {<ingest_record>},
    "validate": {...},
    "probe": {...},
    "summary": { width, height, fps, duration_s, codec, format_valid, decode_valid, errors, deepfake_likelihood, deepfake_label }
  }

CORS is enabled for localhost dev front-end.
Run:
  uvicorn backend.src.api_server:app --reload --port 8000
"""
from __future__ import annotations
import os, uuid, shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Local imports (relative within package)
from .ingest import ingest_single_file
from .probe_media import probe_asset
from .validate_media import validate_asset
from .utils import summarize_ffprobe
from .models.df_detector import predict_deepfake

DATA_ROOT = Path("backend/data")
RAW_ROOT = DATA_ROOT / "raw"
UPLOAD_ROOT = DATA_ROOT / "uploads"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Blackline Forensic API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev; tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeResponse(BaseModel):
    asset: dict
    validate: dict | None
    probe: dict | None
    summary: dict

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)):
    # 1) Persist upload to temp path
    fn = file.filename or "upload.bin"
    suffix_safe = os.path.basename(fn)
    temp_name = f"tmp_{uuid.uuid4().hex}_{suffix_safe}"
    temp_path = UPLOAD_ROOT / temp_name
    try:
        with open(temp_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    # 2) Ingest copy/move into raw store
    try:
        ingest_rec = ingest_single_file(temp_path, RAW_ROOT, move=False)
        if ingest_rec is None:
            raise HTTPException(status_code=400, detail="Unsupported file type")
    finally:
        # remove temp file regardless (content now copied)
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass

    # 3) Validate & probe
    validate_rec = validate_asset(ingest_rec["stored_path"], ingest_rec["store_root"], ingest_rec["sha256"], ingest_rec.get("mime"))
    probe_rec = probe_asset(ingest_rec["stored_path"], ingest_rec["store_root"], ingest_rec["sha256"], ingest_rec.get("mime"), no_exif=True)

    # 4) Stub deepfake model
    video_abs = Path(ingest_rec["store_root"]) / ingest_rec["stored_path"]
    df_pred = predict_deepfake(video_abs, sha256=ingest_rec["sha256"])

    # 5) Summary
    sum_probe = summarize_ffprobe((probe_rec or {}).get("probe")) if probe_rec else {}
    summary = {
        "width": sum_probe.get("width"),
        "height": sum_probe.get("height"),
        "fps": sum_probe.get("fps"),
        "duration_s": sum_probe.get("duration_s"),
        "codec": sum_probe.get("codec"),
        "format_valid": validate_rec.get("format_valid") if validate_rec else None,
        "decode_valid": validate_rec.get("decode_valid") if validate_rec else None,
        "errors": validate_rec.get("errors") if validate_rec else [],
        "deepfake_likelihood": df_pred.get("score"),
        "deepfake_label": df_pred.get("label"),
        "deepfake_method": df_pred.get("method"),
    }

    return AnalyzeResponse(asset=ingest_rec, validate=validate_rec, probe=probe_rec, summary=summary)

# Convenience root
@app.get("/")
def root():
    return {"service": "blackline-api", "endpoints": ["/api/health", "/api/analyze"]}
