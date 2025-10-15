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
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.staticfiles import StaticFiles

# Local imports (relative within package)
from .ingest import ingest_single_file
from .probe_media import probe_asset
from .validate_media import validate_asset
from .utils import summarize_ffprobe
from .models.df_detector import predict_deepfake
# Note: heavy DL modules (torch/transformers) are imported lazily inside the route handler
from .models.copy_move_live import predict_copy_move_single
from .models.lbp_live import predict_lbp_single
from .auth import handle_login, handle_signup, get_current_user, to_public, SignupRequest

DATA_ROOT = Path("backend/data")
RAW_ROOT = DATA_ROOT / "raw"
UPLOAD_ROOT = DATA_ROOT / "uploads"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
PROGRESS_ROOT = DATA_ROOT / "derived" / "progress"
PROGRESS_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Blackline Forensic API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev; tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve derived overlays and artifacts for visualization
DERIVED_DIR = DATA_ROOT / "derived"
DERIVED_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(DERIVED_DIR)), name="static")

class AnalyzeResponse(BaseModel):
    asset: dict
    validate: dict | None
    probe: dict | None
    summary: dict

@app.get("/api/health")
def health():
    return {"status": "ok"}

# ---------------------- Lightweight Progress Tracking -----------------------
def _progress_path(job_id: str) -> Path:
    return PROGRESS_ROOT / f"{job_id}.json"

def _write_progress(job_id: str, stage: str, pct: float | int, message: str | None = None) -> None:
    try:
        payload = {
            "job_id": job_id,
            "stage": stage,
            "percent": float(max(0.0, min(100.0, pct))),
            "message": message or "",
        }
        p = _progress_path(job_id)
        p.write_text(__import__("json").dumps(payload), encoding="utf-8")
    except Exception:
        pass

@app.get("/api/progress/{job_id}")
def get_progress(job_id: str):
    p = _progress_path(job_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail="No progress yet")
    import json
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"job_id": job_id, "stage": "unknown", "percent": 0}

# ------------------------- Auth Endpoints (Prototype) -----------------------
@app.post("/api/auth/login")
def login(token = Depends(handle_login)):
    # handle_login returns Token model instance
    return token

@app.post("/api/auth/signup")
def signup(req: SignupRequest):
    return handle_signup(req)

@app.get("/api/auth/me")
def me(user = Depends(get_current_user)):
    return {"user": to_public(user).dict()}

@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...), model: str = Form("stub"), job_id: str = Form(None)):
    # 1) Persist upload to temp path
    fn = file.filename or "upload.bin"
    suffix_safe = os.path.basename(fn)
    temp_name = f"tmp_{uuid.uuid4().hex}_{suffix_safe}"
    temp_path = UPLOAD_ROOT / temp_name
    job_id = job_id or uuid.uuid4().hex
    _write_progress(job_id, "save_upload", 5, "Saving upload")
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
        _write_progress(job_id, "ingest", 15, "Ingesting asset")
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
    _write_progress(job_id, "validate", 30, "Validating format & decode")
    validate_rec = validate_asset(ingest_rec["stored_path"], ingest_rec["store_root"], ingest_rec["sha256"], ingest_rec.get("mime"))
    _write_progress(job_id, "probe", 45, "Probing media")
    probe_rec = probe_asset(ingest_rec["stored_path"], ingest_rec["store_root"], ingest_rec["sha256"], ingest_rec.get("mime"), no_exif=True)

    # 4) Model inference (selected)
    video_abs = Path(ingest_rec["store_root"]) / ingest_rec["stored_path"]
    model = (model or "stub").lower().strip()
    _write_progress(job_id, "model", 65, f"Running model: {model}")
    if model in ("copy_move", "copymove", "cm"):
        df_pred = predict_copy_move_single(video_abs, sha256=ingest_rec["sha256"])  # returns score/label/method + cm_*
    elif model in ("lbp", "lbp_rf", "lbp-model"):
        df_pred = predict_lbp_single(video_abs)
    elif model in ("dl", "deep", "deep_learning", "full", "fusion"):
        # Online fusion (Xception + TimeSformer) for a single asset
        try:
            from .models.fusion_live import predict_fusion_single  # lazy import to avoid startup torch dependency
            base = predict_fusion_single(video_abs)
            df_pred = base
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Deep learning model unavailable: {e}")
    else:
        df_pred = predict_deepfake(video_abs, sha256=ingest_rec["sha256"])  # stub fallback

    # 5) Summary
    sum_probe = summarize_ffprobe((probe_rec or {}).get("probe")) if probe_rec else {}
    _write_progress(job_id, "summary", 85, "Summarizing")
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
    # Include any additional model-specific fields (e.g., copy-move metrics/overlay)
    for k in ("cm_confidence", "cm_coverage_ratio", "cm_shift_magnitude", "cm_num_keypoints", "cm_num_matches", "overlay_uri",
              "lbp_frames", "lbp_dim", "xception_agg_score", "timesformer_score"):
        if df_pred.get(k) is not None:
            summary[k] = df_pred[k]

    _write_progress(job_id, "done", 100, "Completed")
    return AnalyzeResponse(asset=ingest_rec, validate=validate_rec, probe=probe_rec, summary=summary)

# Convenience root
@app.get("/")
def root():
    return {"service": "blackline-api", "endpoints": ["/api/health", "/api/analyze"]}
