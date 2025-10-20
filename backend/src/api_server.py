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
from .audit import append_audit_row, DEFAULT_INGEST_LOG
from .probe_media import probe_asset
from .validate_media import validate_asset
from .utils import summarize_ffprobe
from .models.df_detector import predict_deepfake
# Note: heavy DL modules (torch/transformers) are imported lazily inside the route handler
from .models.copy_move_live import predict_copy_move_single
from .models.lbp_live import predict_lbp_single
from .auth import handle_login, handle_signup, get_current_user, to_public, SignupRequest
from .db import init_db, get_db
from . import models_db  # ensure models are imported
from sqlalchemy.orm import Session
from fastapi import Request
from .storage import get_storage

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
 # Expose raw assets for video playback (dev/local only)
_storage_backend = os.getenv("STORAGE_BACKEND", "local").lower()
if _storage_backend == "local":
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    app.mount("/assets", StaticFiles(directory=str(RAW_ROOT)), name="assets")
FRAMES_DIR = DERIVED_DIR / "frames"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
STORAGE = get_storage(RAW_ROOT)

# Initialize database on startup
@app.on_event("startup")
def on_startup():
    init_db(models_module=models_db)
    # Seed guest user
    from .auth import ensure_seed_user
    from sqlalchemy.orm import Session
    for db in get_db():  # use dependency generator to get a session
        ensure_seed_user(db)
        break

class AnalyzeResponse(BaseModel):
    asset: dict
    validate: dict | None
    probe: dict | None
    summary: dict

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.get("/api/db/health")
def db_health():
    # Try a simple query
    for db in get_db():
        try:
            db.execute(__import__("sqlalchemy").text("SELECT 1"))
            return {"db": "ok"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"DB error: {e}")

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
def signup(req: SignupRequest, db: Session = Depends(get_db)):
    return handle_signup(req, db)

@app.get("/api/auth/me")
def me(user = Depends(get_current_user)):
    return {"user": to_public(user).dict()}

# ------------------------- Frame Extraction -------------------------------
@app.get("/api/frame")
def get_frame(sha256: str, stored_path: str, index: int | None = None, time_s: float | None = None):
    """Extract a frame as PNG and return its static URI.
    Provide either index (frame number) or time_s (seconds).
    stored_path is the ingest stored_path ("<sha>/<filename>").
    """
    try:
        import cv2
    except Exception:
        raise HTTPException(status_code=500, detail="OpenCV not available for frame extraction")
    try:
        if not sha256 or not stored_path:
            raise HTTPException(status_code=400, detail="sha256 and stored_path required")
        # Build absolute video path
        video_abs = Path(DATA_ROOT / "raw" / stored_path)
        if not video_abs.exists():
            raise HTTPException(status_code=404, detail="Video not found")
        # Cache path
        out_dir = FRAMES_DIR / sha256
        out_dir.mkdir(parents=True, exist_ok=True)
        key = f"idx_{index}" if index is not None else f"t_{(time_s or 0):.2f}"
        out_path = out_dir / f"{key}.png"
        # If already exists, return URI
        if out_path.exists():
            uri = f"frames/{sha256}/{out_path.name}"
            return {"uri": uri}
        cap = cv2.VideoCapture(str(video_abs))
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Failed to open video")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        # Resolve target frame
        tgt = None
        if index is not None:
            tgt = max(0, min(total - 1 if total > 0 else index, int(index)))
        elif time_s is not None and fps > 0:
            tgt = int(max(0, min(total - 1 if total > 0 else 0, round(time_s * fps))))
        else:
            cap.release(); raise HTTPException(status_code=400, detail="Provide index or valid time_s")
        # Seek and read
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(tgt))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise HTTPException(status_code=500, detail="Failed to read frame")
        # Write PNG
        ok = cv2.imwrite(str(out_path), frame)
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to write frame image")
        uri = f"frames/{sha256}/{out_path.name}"
        return {"uri": uri}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Frame extract error: {e}")

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

    # Append ingest audit row (API source) to ingest_log.jsonl
    try:
        if ingest_rec is not None:
            dest_abs = Path(ingest_rec["store_root"]) / ingest_rec["stored_path"]
            try:
                size_bytes = dest_abs.stat().st_size
            except Exception:
                size_bytes = None
            api_ingest_row = {
                "ts": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
                "user": __import__("getpass").getuser(),
                "action": ingest_rec["action"],
                "sha256": ingest_rec["sha256"],
                "stored_path": ingest_rec["stored_path"],
                "size_bytes": size_bytes,
                "mime": ingest_rec.get("mime"),
                "source": "api",
            }
            api_ingest_row = {k: v for k, v in api_ingest_row.items() if v is not None}
            append_audit_row(DEFAULT_INGEST_LOG, api_ingest_row)
    except Exception:
        # Do not fail request on audit write issues
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

    # Include optional per-frame scores timeline if produced by fusion model
    if isinstance(df_pred.get("frame_scores"), list) and df_pred.get("frame_scores"):
        summary["frame_scores"] = df_pred["frame_scores"]
        if df_pred.get("frame_fps") is not None:
            summary["frame_fps"] = df_pred["frame_fps"]
        if df_pred.get("frame_total") is not None:
            summary["frame_total"] = df_pred["frame_total"]

    _write_progress(job_id, "done", 100, "Completed")
    return AnalyzeResponse(asset=ingest_rec, validate=validate_rec, probe=probe_rec, summary=summary)

class AnalyzeAssetRequest(BaseModel):
    asset_id: int
    model: str | None = "stub"
    job_id: str | None = None

@app.post("/api/analyze/asset", response_model=AnalyzeResponse)
def analyze_asset(req: AnalyzeAssetRequest, user = Depends(get_current_user)):
    # Use client-provided job id for UI progress, or generate one
    job_id = req.job_id or uuid.uuid4().hex
    _write_progress(job_id, "validate", 5, "Preparing analysis")
    # Load the asset owned by the user
    for db in get_db():
        sess: Session = db
        a = sess.query(models_db.Asset).filter(models_db.Asset.id == req.asset_id, models_db.Asset.user_id == user.id).first()
        if not a:
            raise HTTPException(status_code=404, detail="Asset not found")
        break

    # Build ingest-like record
    ingest_rec = {
        "action": "ingest",
        "sha256": a.sha256,
        "stored_path": a.stored_path,
        "store_root": str(RAW_ROOT),
        "mime": a.mime,
        "asset_id": a.id,
    }

    # Validate & probe
    _write_progress(job_id, "validate", 20, "Validating format & decode")
    validate_rec = validate_asset(a.stored_path or "", str(RAW_ROOT), a.sha256 or "", a.mime)
    _write_progress(job_id, "probe", 35, "Probing media")
    probe_rec = probe_asset(a.stored_path or "", str(RAW_ROOT), a.sha256 or "", a.mime, no_exif=True)

    # Model inference
    video_abs = Path(RAW_ROOT) / (a.stored_path or "")
    model = (req.model or "stub").lower().strip()
    # If local file missing but remote_key present, download to temp
    temp_path = None
    if (not video_abs.exists()) and a.remote_key:
        try:
            _write_progress(job_id, "download", 50, "Fetching remote asset")
            temp_path = STORAGE.download_to_temp(a.remote_key)
            video_abs = temp_path
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to retrieve remote asset: {e}")

    _write_progress(job_id, "model", 65, f"Running model: {model}")
    if model in ("copy_move", "copymove", "cm"):
        df_pred = predict_copy_move_single(video_abs, sha256=(a.sha256 or ""))
    elif model in ("lbp", "lbp_rf", "lbp-model"):
        df_pred = predict_lbp_single(video_abs)
    elif model in ("dl", "deep", "deep_learning", "full", "fusion"):
        try:
            from .models.fusion_live import predict_fusion_single
            base = predict_fusion_single(video_abs)
            df_pred = base
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Deep learning model unavailable: {e}")
    else:
        df_pred = predict_deepfake(video_abs, sha256=(a.sha256 or ""))
    # cleanup temp
    try:
        if temp_path and Path(temp_path).exists():
            Path(temp_path).unlink()
    except Exception:
        pass

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
    for k in ("cm_confidence", "cm_coverage_ratio", "cm_shift_magnitude", "cm_num_keypoints", "cm_num_matches", "overlay_uri",
              "lbp_frames", "lbp_dim", "xception_agg_score", "timesformer_score"):
        if df_pred.get(k) is not None:
            summary[k] = df_pred[k]
    if isinstance(df_pred.get("frame_scores"), list) and df_pred.get("frame_scores"):
        summary["frame_scores"] = df_pred["frame_scores"]
        if df_pred.get("frame_fps") is not None:
            summary["frame_fps"] = df_pred["frame_fps"]
        if df_pred.get("frame_total") is not None:
            summary["frame_total"] = df_pred["frame_total"]

    _write_progress(job_id, "done", 100, "Completed")
    return AnalyzeResponse(asset=ingest_rec, validate=validate_rec, probe=probe_rec, summary=summary)

# ------------------------- Cloud Storage Helpers -------------------------
# Define AssetOut before any routes reference it to avoid forward-ref issues in Pydantic v2
class AssetOut(BaseModel):
    id: int
    original_name: str
    mime: str | None
    size_bytes: int | None
    sha256: str | None
    stored_path: str | None
    remote_key: str | None
    visibility: str
    created_at: int

def _asset_to_out(a: models_db.Asset) -> 'AssetOut':
    return AssetOut(
        id=a.id,
        original_name=a.original_name,
        mime=a.mime,
        size_bytes=a.size_bytes,
        sha256=a.sha256,
        stored_path=a.stored_path,
        remote_key=a.remote_key,
        visibility=a.visibility,
        created_at=a.created_at,
    )

class UploadUrlRequest(BaseModel):
    file_name: str
    content_type: str | None = None

class UploadUrlResponse(BaseModel):
    key: str
    upload_url: str
    headers: dict | None = None

@app.post("/api/assets/upload-url", response_model=UploadUrlResponse)
def get_upload_url(req: UploadUrlRequest, user = Depends(get_current_user)):
    # Use a per-user prefix for keys
    safe_name = os.path.basename(req.file_name)
    key = f"users/{user.username}/{uuid.uuid4().hex}_{safe_name}"
    try:
        url, headers = STORAGE.generate_upload_url(key, content_type=req.content_type or None)
        return UploadUrlResponse(key=key, upload_url=url, headers=headers or {})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate upload URL: {e}")

class ConfirmUploadRequest(BaseModel):
    key: str
    original_name: str
    mime: str | None = None

@app.post("/api/assets/confirm", response_model=AssetOut)
def confirm_uploaded(req: ConfirmUploadRequest, user = Depends(get_current_user)):
    meta = STORAGE.confirm_object(req.key)
    if not meta.get("exists"):
        raise HTTPException(status_code=400, detail="Object not found in storage")
    size_bytes = int(meta.get("size_bytes") or 0)
    # Insert DB row
    for db in get_db():
        sess: Session = db
        rec = models_db.Asset(
            user_id=user.id,
            original_name=req.original_name,
            mime=req.mime,
            size_bytes=size_bytes,
            sha256=None,
            stored_path=None,
            remote_key=req.key,
            visibility="private",
        )
        sess.add(rec)
        sess.commit()
        sess.refresh(rec)
        return _asset_to_out(rec)

class PlaybackUrlResponse(BaseModel):
    url: str

@app.get("/api/assets/{asset_id}/playback-url", response_model=PlaybackUrlResponse)
def get_playback_url(asset_id: int, user = Depends(get_current_user)):
    # Find asset and return a signed or static URL
    for db in get_db():
        sess: Session = db
        a = sess.query(models_db.Asset).filter(models_db.Asset.id == asset_id, models_db.Asset.user_id == user.id).first()
        if not a:
            raise HTTPException(status_code=404, detail="Asset not found")
        if a.stored_path:
            url = STORAGE.generate_download_url(a.stored_path) or f"/assets/{a.stored_path}"
            return PlaybackUrlResponse(url=url)
        if a.remote_key:
            url = STORAGE.generate_download_url(a.remote_key)
            if not url:
                raise HTTPException(status_code=500, detail="Download URL unavailable for remote asset")
            return PlaybackUrlResponse(url=url)
        raise HTTPException(status_code=400, detail="Asset has no storage reference")

# ------------------------- Persistent Assets (per-user) --------------------
@app.get("/api/assets", response_model=list[AssetOut])
def list_assets(user = Depends(get_current_user)):
    for db in get_db():
        sess: Session = db
        rows = sess.query(models_db.Asset).filter(models_db.Asset.user_id == user.id).order_by(models_db.Asset.created_at.desc()).all()
        return [_asset_to_out(r) for r in rows]

@app.post("/api/assets", response_model=AssetOut)
async def upload_asset(file: UploadFile = File(...), user = Depends(get_current_user)):
    """Persist an uploaded file as a user-owned asset.
    For now this stores into local RAW store and records DB metadata.
    Later we can swap to S3/Azure/GCS by saving remote_key and serving via signed URLs.
    """
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

    try:
        ingest_rec = ingest_single_file(temp_path, RAW_ROOT, move=False)
    finally:
        if temp_path.exists():
            try: temp_path.unlink()
            except Exception: pass

    # Store DB record
    for db in get_db():
        sess: Session = db
        size_bytes = None
        try:
            dest_abs = Path(ingest_rec["store_root"]) / ingest_rec["stored_path"]
            size_bytes = dest_abs.stat().st_size
        except Exception:
            pass
        rec = models_db.Asset(
            user_id=user.id,
            original_name=fn,
            mime=ingest_rec.get("mime"),
            size_bytes=size_bytes,
            sha256=ingest_rec.get("sha256"),
            stored_path=ingest_rec.get("stored_path"),
            remote_key=None,
            visibility="private",
        )
        sess.add(rec)
        sess.commit()
        sess.refresh(rec)
        return _asset_to_out(rec)

# Convenience root
@app.get("/")
def root():
    return {"service": "blackline-api", "endpoints": ["/api/health", "/api/analyze"]}
