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
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles

# Local imports (relative within package)
from .ingest import ingest_single_file
from .audit import append_audit_row, DEFAULT_INGEST_LOG
from .probe_media import probe_asset
from .validate_media import validate_asset
from .utils import summarize_ffprobe
from .models.df_detector import predict_deepfake
# Note: heavy DL modules (torch/transformers) are imported lazily inside the route handler
# Optional heavy/experimental models are imported lazily within route handlers
# to avoid startup failures in environments where the files or dependencies
# are not present (e.g., Cloud Run minimal images).
from .auth import handle_login, handle_signup, get_current_user, to_public, SignupRequest
from .db import init_db, get_db
from . import models_db  # ensure models are imported
from sqlalchemy.orm import Session
from fastapi import Request
from .storage import get_storage

# Use writable data root; on Cloud Run the image FS is read-only except /tmp
_default_data_root = "/tmp/data" if os.getenv("PORT") else "backend/data"
DATA_ROOT = Path(os.getenv("DATA_ROOT", _default_data_root))
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

# Mount the 'data' directory to serve static files
app.mount("/data", StaticFiles(directory="backend/data"), name="data")

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

@app.get("/status")
def get_status():
    """
    Returns the status of the server environment.
    """
    import torch
    import torchvision

    status = {
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if status["cuda_available"]:
        status.update({
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_current_device": torch.cuda.current_device(),
            "cuda_device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
        })
    else:
        status.update({
            "cuda_device_count": 0,
            "cuda_current_device": None,
            "cuda_device_name": None,
        })
    return status


class AnalyzeResponse(BaseModel):
    asset: dict
    validate: dict | None
    probe: dict | None
    summary: dict

class PageEntry(BaseModel):
    key: str
    label: str
    icon: str | None = None

class PagesPayload(BaseModel):
    pages: list[PageEntry]

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.get("/api/dl/info")
def dl_info():
    """Report deep learning fusion availability and which mode would be used.
    mode: "fusion-lr" if backend/models/fusion_lr.json exists, else "fusion-blend".
    """
    from .models.fusion_live import _resolve_models_root
    mroot = _resolve_models_root()
    x_ck = (mroot / "xception_best.pth").exists()
    ts_ck = (mroot / "timesformer_best.pt").exists()
    ts_cfg = (mroot / "timesformer_best.config.json").exists()
    lr_w = (mroot / "fusion_lr.json").exists()
    return {
        "models_root": str(mroot),
        "weights": {
            "xception": x_ck,
            "timesformer": ts_ck,
            "timesformer_config": ts_cfg,
            "fusion_lr": lr_w,
        },
        "mode": "fusion-lr" if lr_w else "fusion-blend",
        "method_label": "fusion-lr(xception,timesformer)" if lr_w else "fusion-blend(avg(xception,timesformer))",
    }

@app.get("/")
def root():
    return {"service": "blackline-api", "status": "ok"}

@app.get("/api/buildinfo")
def buildinfo():
    import shutil, os
    from .utils import run_command
    ffprobe_present = shutil.which("ffprobe") is not None
    ffmpeg_present = shutil.which("ffmpeg") is not None
    ffprobe_ver = None
    ffmpeg_ver = None
    if ffprobe_present:
        p = run_command(["ffprobe", "-version"])
        ffprobe_ver = (p.stdout or p.stderr).splitlines()[0] if (p.stdout or p.stderr) else None
    if ffmpeg_present:
        p = run_command(["ffmpeg", "-version"])
        ffmpeg_ver = (p.stdout or p.stderr).splitlines()[0] if (p.stdout or p.stderr) else None
    # Extra diagnostics to understand production behavior
    torch_ok = False
    decord_ok = False
    cv_ok = False
    try:
        import torch  # type: ignore
        torch_ok = True
    except Exception:
        pass
    try:
        import decord  # type: ignore
        decord_ok = True
    except Exception:
        pass
    try:
        import cv2  # type: ignore
        _ = cv2.__version__
        cv_ok = True
    except Exception:
        pass
    # Check model checkpoint presence
    from .models.fusion_live import _resolve_models_root
    mroot = _resolve_models_root()
    import pathlib
    x_ck = (mroot / "xception_best.pth").exists()
    ts_ck = (mroot / "timesformer_best.pt").exists()
    ts_cfg = (mroot / "timesformer_best.config.json").exists()
    lr_w  = (mroot / "fusion_lr.json").exists()
    return {
        "api_version": app.version,
        "storage_backend": os.getenv("STORAGE_BACKEND", "local"),
        "ffprobe_present": ffprobe_present,
        "ffmpeg_present": ffmpeg_present,
        "ffprobe_version": ffprobe_ver,
        "ffmpeg_version": ffmpeg_ver,
        "libs": {"torch": torch_ok, "decord": decord_ok, "opencv": cv_ok},
        "models": {"root": str(mroot), "xception": x_ck, "timesformer": ts_ck, "ts_config": ts_cfg, "fusion_lr_weights": lr_w},
        # Marker that this build downloads remote assets before probing
        "features": {"download_first": True, "path_progress_message": True},
        # Helpful to identify Cloud Run/Render
        "env": {"K_SERVICE": os.getenv("K_SERVICE"), "K_REVISION": os.getenv("K_REVISION"), "RENDER": os.getenv("RENDER")} 
    }

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

# -------------------------- ELA Frames Generation --------------------------
class ELAGenerateReq(BaseModel):
    asset_id: int
    frames: int | None = 12
    quality: int | None = 90
    scale: float | None = 10.0

@app.post("/api/ela/generate")
def generate_ela_frames(req: ELAGenerateReq, user = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Generate Error Level Analysis (ELA) frames for an asset.
    - For videos: sample N frames uniformly and compute ELA for each.
    - For images: compute a single ELA image.

    Returns:
      { ela_frames: [{ uri, time_s?, index? }], count: int }
    Where `uri` is relative to the /static mount (DERIVED_DIR).
    """
    from PIL import Image, ImageChops
    import io
    import cv2  # type: ignore
    import json

    # Lookup asset and authorization
    asset = db.query(models_db.Asset).filter(models_db.Asset.id == req.asset_id, models_db.Asset.user_id == user.id).one_or_none()
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")

    # Resolve filesystem path (supports both local stored_path and remote_key)
    stored_path = asset.stored_path
    abspath = None
    temp_download = None
    if stored_path:
        abspath = RAW_ROOT / stored_path
        if not abspath.exists():
            # If the local copy is missing but we have a remote key, fetch a temp copy
            if asset.remote_key:
                try:
                    temp_download = STORAGE.download_to_temp(asset.remote_key)
                    abspath = temp_download
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Failed to retrieve remote asset: {e}")
            else:
                raise HTTPException(status_code=404, detail="Stored file missing")
    elif asset.remote_key:
        try:
            temp_download = STORAGE.download_to_temp(asset.remote_key)
            abspath = temp_download
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to retrieve remote asset: {e}")
    else:
        raise HTTPException(status_code=400, detail="Asset has no stored path")

    # Output directory under derived
    key = asset.sha256 or f"asset_{asset.id}"
    out_dir = DERIVED_DIR / "ela" / key
    out_dir.mkdir(parents=True, exist_ok=True)

    def _ela_from_pil(pil_img: Image.Image, quality: int, scale: float) -> Image.Image:
        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG', quality=max(1, min(100, int(quality))))
        buf.seek(0)
        jpeg_img = Image.open(buf).convert('RGB')
        diff = ImageChops.difference(pil_img.convert('RGB'), jpeg_img)
        # scale contrast by multiplying pixel values
        def _mul(x: int) -> int:
            v = int(float(x) * float(scale))
            return 255 if v > 255 else (0 if v < 0 else v)
        return diff.point(_mul)

    quality = int(req.quality or 90)
    scale = float(req.scale or 10.0)
    frames_target = int(req.frames or 12)

    ela_items: list[dict] = []

    try:
        mime = (asset.mime or '').lower()
        if mime.startswith('video/'):
            cap = cv2.VideoCapture(str(abspath))
            if not cap.isOpened():
                raise HTTPException(status_code=500, detail="Failed to open video for ELA")
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            # Sample evenly across the duration
            n = max(1, min(frames_target, total if total > 0 else frames_target))
            indices = []
            if total and total > 0:
                for i in range(n):
                    idx = int(round((i + 0.5) * (total / n)))
                    idx = max(0, min(total - 1, idx))
                    indices.append(idx)
            else:
                indices = list(range(n))
            used = set()
            for idx in indices:
                if idx in used: continue
                used.add(idx)
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                ela_img = _ela_from_pil(pil, quality, scale)
                # Save as PNG
                t = (idx / fps) if (fps and fps > 0) else None
                name = f"f_{idx:06d}.png"
                out_path = out_dir / name
                ela_img.save(str(out_path), format='PNG')
                ela_items.append({
                    "uri": f"ela/{key}/{name}",
                    "time_s": (float(t) if t is not None else None),
                    "index": int(idx),
                })
            cap.release()
        else:
            # Treat as image
            pil = Image.open(str(abspath)).convert('RGB')
            ela_img = _ela_from_pil(pil, quality, scale)
            name = Path(stored_path).name
            stem = Path(name).stem
            out_path = out_dir / f"{stem}_ela.png"
            ela_img.save(str(out_path), format='PNG')
            ela_items.append({
                "uri": f"ela/{key}/{out_path.name}",
                "time_s": None,
                "index": None,
            })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ELA generation failed: {e}")
    finally:
        # Clean up any temp download used
        try:
            if temp_download and Path(temp_download).exists():
                Path(temp_download).unlink()
        except Exception:
            pass

    return { "ela_frames": ela_items, "count": len(ela_items) }

# List existing ELA frames for an asset (no regeneration)
@app.get("/api/ela/list")
def list_ela_frames(asset_id: int, user = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Return any ELA frames already generated for the given asset.
    Scans derived/ela/<key> and returns URIs relative to /static.
    """
    import re
    asset = db.query(models_db.Asset).filter(models_db.Asset.id == asset_id, models_db.Asset.user_id == user.id).one_or_none()
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    key = (asset.sha256 or f"asset_{asset.id}")
    out_dir = DERIVED_DIR / "ela" / key
    if not out_dir.exists():
        return {"ela_frames": [], "count": 0}
    items: list[dict] = []
    pat = re.compile(r"^f_(\d+)\.png$", re.IGNORECASE)
    for p in sorted(out_dir.glob("*.png")):
        name = p.name
        m = pat.match(name)
        idx = int(m.group(1)) if m else None
        items.append({
            "uri": f"ela/{key}/{name}",
            "time_s": None,
            "index": idx,
        })
    # If we have frame indices, sort by them
    items.sort(key=lambda d: (d.get("index") is None, d.get("index", 0), d.get("uri")))
    return {"ela_frames": items, "count": len(items)}

# -------------------------- LBP Frames Generation --------------------------
class LBPGenerateReq(BaseModel):
    asset_id: int
    frames: int | None = 12
    radius: int | None = 1

@app.post("/api/lbp/generate")
def generate_lbp_frames(req: LBPGenerateReq, user = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Generate Local Binary Pattern (LBP) frames for an asset.
    - For videos: sample N frames uniformly and compute LBP for each.
    - For images: compute a single LBP image.

    Returns:
      { lbp_frames: [{ uri, time_s?, index? }], count: int }
    Where `uri` is relative to the /static mount (DERIVED_DIR).
    """
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    from PIL import Image

    # Lookup asset and authorization
    asset = db.query(models_db.Asset).filter(models_db.Asset.id == req.asset_id, models_db.Asset.user_id == user.id).one_or_none()
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")

    # Resolve path (local or remote)
    stored_path = asset.stored_path
    abspath = None
    temp_download = None
    if stored_path:
        abspath = RAW_ROOT / stored_path
        if not abspath.exists():
            if asset.remote_key:
                try:
                    temp_download = STORAGE.download_to_temp(asset.remote_key)
                    abspath = temp_download
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Failed to retrieve remote asset: {e}")
            else:
                raise HTTPException(status_code=404, detail="Stored file missing")
    elif asset.remote_key:
        try:
            temp_download = STORAGE.download_to_temp(asset.remote_key)
            abspath = temp_download
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to retrieve remote asset: {e}")
    else:
        raise HTTPException(status_code=400, detail="Asset has no stored path")

    # Output directory
    key = asset.sha256 or f"asset_{asset.id}"
    out_dir = DERIVED_DIR / "lbp" / key
    out_dir.mkdir(parents=True, exist_ok=True)

    def _lbp_image_from_gray(gray: np.ndarray, radius: int = 1) -> np.ndarray:
        """Compute basic 8-neighbor LBP code image for a grayscale uint8 array.
        Returns an 8-bit image visualizing the LBP codes (0..255).
        """
        # Pad image to handle borders
        r = max(1, int(radius))
        g = gray.astype(np.uint8)
        h, w = g.shape[:2]
        # Create shifted neighbors
        # Offsets in clockwise order starting at (-r, -r)
        offsets = [
            (-r, -r), (0, -r), (r, -r),
            (r, 0), (r, r), (0, r),
            (-r, r), (-r, 0)
        ]
        # Build binary pattern
        center = g
        acc = np.zeros_like(center, dtype=np.uint16)
        for bit, (dx, dy) in enumerate(offsets):
            x0 = max(0, dx)
            x1 = h + min(0, dx)
            y0 = max(0, dy)
            y1 = w + min(0, dy)
            c_slice = center[x0:x1, y0:y1]
            n_slice = center[x0-dx:x1-dx, y0-dy:y1-dy]
            # Compare neighbor >= center => set bit
            mask = (n_slice >= c_slice).astype(np.uint16)
            acc[x0:x1, y0:y1] |= (mask << bit)
        # Scale to 0..255 (already 0..255 range due to 8 bits)
        return acc.astype(np.uint8)

    frames_target = int(req.frames or 12)
    radius = int(req.radius or 1)
    lbp_items: list[dict] = []
    try:
        mime = (asset.mime or '').lower()
        if mime.startswith('video/'):
            cap = cv2.VideoCapture(str(abspath))
            if not cap.isOpened():
                raise HTTPException(status_code=500, detail="Failed to open video for LBP")
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            n = max(1, min(frames_target, total if total > 0 else frames_target))
            indices = []
            if total and total > 0:
                for i in range(n):
                    idx = int(round((i + 0.5) * (total / n)))
                    idx = max(0, min(total - 1, idx))
                    indices.append(idx)
            else:
                indices = list(range(n))
            used = set()
            for idx in indices:
                if idx in used: continue
                used.add(idx)
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                lbp = _lbp_image_from_gray(gray, radius=radius)
                # Colorize slightly for visual separation (apply colormap)
                colored = cv2.applyColorMap(lbp, cv2.COLORMAP_MAGMA)
                t = (idx / fps) if (fps and fps > 0) else None
                name = f"f_{idx:06d}.png"
                out_path = out_dir / name
                ok = cv2.imwrite(str(out_path), colored)
                if not ok:
                    continue
                lbp_items.append({
                    "uri": f"lbp/{key}/{name}",
                    "time_s": (float(t) if t is not None else None),
                    "index": int(idx),
                })
            cap.release()
        else:
            # Treat as image
            # Use PIL to read cross-format, then convert to gray
            pil = Image.open(str(abspath)).convert('L')
            gray = np.array(pil, dtype=np.uint8)
            lbp = _lbp_image_from_gray(gray, radius=radius)
            import cv2 as _cv
            colored = _cv.applyColorMap(lbp, _cv.COLORMAP_MAGMA)
            name = Path(stored_path).name
            stem = Path(name).stem
            out_path = out_dir / f"{stem}_lbp.png"
            _cv.imwrite(str(out_path), colored)
            lbp_items.append({
                "uri": f"lbp/{key}/{out_path.name}",
                "time_s": None,
                "index": None,
            })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LBP generation failed: {e}")
    finally:
        try:
            if temp_download and Path(temp_download).exists():
                Path(temp_download).unlink()
        except Exception:
            pass

    return { "lbp_frames": lbp_items, "count": len(lbp_items) }

# List LBP frames for an asset
@app.get("/api/lbp/list")
def list_lbp_frames(asset_id: int, user = Depends(get_current_user), db: Session = Depends(get_db)):
    import re
    asset = db.query(models_db.Asset).filter(models_db.Asset.id == asset_id, models_db.Asset.user_id == user.id).one_or_none()
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    key = (asset.sha256 or f"asset_{asset.id}")
    out_dir = DERIVED_DIR / "lbp" / key
    if not out_dir.exists():
        return {"lbp_frames": [], "count": 0}
    items: list[dict] = []
    pat = re.compile(r"^f_(\d+)\.png$", re.IGNORECASE)
    for p in sorted(out_dir.glob("*.png")):
        name = p.name
        m = pat.match(name)
        idx = int(m.group(1)) if m else None
        items.append({
            "uri": f"lbp/{key}/{name}",
            "time_s": None,
            "index": idx,
        })
    items.sort(key=lambda d: (d.get("index") is None, d.get("index", 0), d.get("uri")))
    return {"lbp_frames": items, "count": len(items)}

# Fallback variants: by stored_path
class LBPGenerateByPathReq(BaseModel):
    stored_path: str
    sha256: str | None = None
    frames: int | None = 12
    radius: int | None = 1

@app.post("/api/lbp/generate-by-path")
def generate_lbp_frames_by_path(req: LBPGenerateByPathReq, user = Depends(get_current_user)):
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    from PIL import Image
    from pathlib import Path as _P

    # Resolve path under RAW_ROOT
    try:
        sp = (req.stored_path or "").lstrip("/")
        if not sp:
            raise HTTPException(status_code=400, detail="stored_path required")
        abspath = (RAW_ROOT / sp).resolve()
        root = RAW_ROOT.resolve()
        if not str(abspath).startswith(str(root)):
            raise HTTPException(status_code=400, detail="Invalid stored_path")
        if not abspath.exists():
            raise HTTPException(status_code=404, detail="File not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad stored_path: {e}")

    parts = _P(sp).parts
    key = (req.sha256 or (parts[0] if parts else "file"))
    out_dir = DERIVED_DIR / "lbp" / key
    out_dir.mkdir(parents=True, exist_ok=True)

    def _lbp_image_from_gray(gray: np.ndarray, radius: int = 1) -> np.ndarray:
        r = max(1, int(radius))
        g = gray.astype(np.uint8)
        h, w = g.shape[:2]
        offsets = [(-r,-r),(0,-r),(r,-r),(r,0),(r,r),(0,r),(-r,r),(-r,0)]
        center = g
        acc = np.zeros_like(center, dtype=np.uint16)
        for bit, (dx, dy) in enumerate(offsets):
            x0 = max(0, dx); x1 = h + min(0, dx)
            y0 = max(0, dy); y1 = w + min(0, dy)
            c_slice = center[x0:x1, y0:y1]
            n_slice = center[x0-dx:x1-dx, y0-dy:y1-dy]
            mask = (n_slice >= c_slice).astype(np.uint16)
            acc[x0:x1, y0:y1] |= (mask << bit)
        return acc.astype(np.uint8)

    frames_target = int(req.frames or 12)
    radius = int(req.radius or 1)
    lbp_items: list[dict] = []
    try:
        ext = abspath.suffix.lower().strip('.')
        is_video = ext in {"mp4","mov","mkv","avi","webm","m4v"}
        if is_video:
            cap = cv2.VideoCapture(str(abspath))
            if not cap.isOpened():
                raise HTTPException(status_code=500, detail="Failed to open video for LBP")
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            n = max(1, min(frames_target, total if total > 0 else frames_target))
            indices = []
            if total and total > 0:
                for i in range(n):
                    idx = int(round((i + 0.5) * (total / n)))
                    idx = max(0, min(total - 1, idx))
                    indices.append(idx)
            else:
                indices = list(range(n))
            used = set()
            for idx in indices:
                if idx in used: continue
                used.add(idx)
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                lbp = _lbp_image_from_gray(gray, radius=radius)
                colored = cv2.applyColorMap(lbp, cv2.COLORMAP_MAGMA)
                t = (idx / fps) if (fps and fps > 0) else None
                name = f"f_{idx:06d}.png"
                out_path = out_dir / name
                ok = cv2.imwrite(str(out_path), colored)
                if not ok:
                    continue
                lbp_items.append({"uri": f"lbp/{key}/{name}", "time_s": (float(t) if t is not None else None), "index": int(idx)})
            cap.release()
        else:
            from PIL import Image as _Image
            gray = _Image.open(str(abspath)).convert('L')
            import numpy as _np
            arr = _np.array(gray, dtype=_np.uint8)
            lbp = _lbp_image_from_gray(arr, radius=radius)
            import cv2 as _cv
            colored = _cv.applyColorMap(lbp, _cv.COLORMAP_MAGMA)
            stem = _P(abspath.name).stem
            out_path = out_dir / f"{stem}_lbp.png"
            _cv.imwrite(str(out_path), colored)
            lbp_items.append({"uri": f"lbp/{key}/{out_path.name}", "time_s": None, "index": None})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LBP generation failed: {e}")

    return {"lbp_frames": lbp_items, "count": len(lbp_items)}

@app.get("/api/lbp/list-by-path")
def list_lbp_frames_by_path(stored_path: str, sha256: str | None = None, user = Depends(get_current_user)):
    import re
    from pathlib import Path as _P
    sp = (stored_path or "").lstrip("/")
    if not sp:
        raise HTTPException(status_code=400, detail="stored_path required")
    parts = _P(sp).parts
    key = (sha256 or (parts[0] if parts else "file"))
    out_dir = DERIVED_DIR / "lbp" / key
    if not out_dir.exists():
        return {"lbp_frames": [], "count": 0}
    items: list[dict] = []
    pat = re.compile(r"^f_(\d+)\.png$", re.IGNORECASE)
    for p in sorted(out_dir.glob("*.png")):
        name = p.name
        m = pat.match(name)
        idx = int(m.group(1)) if m else None
        items.append({
            "uri": f"lbp/{key}/{name}",
            "time_s": None,
            "index": idx,
        })
    items.sort(key=lambda d: (d.get("index") is None, d.get("index", 0), d.get("uri")))
    return {"lbp_frames": items, "count": len(items)}

# -------------------------- Noise Frames Generation ------------------------
class NoiseGenerateReq(BaseModel):
    asset_id: int
    frames: int | None = 12
    method: str | None = "both"  # residual | fft | both
    threshold: float | None = 0.6  # high-confidence threshold on noise_score

@app.post("/api/noise/generate")
def generate_noise_frames(req: NoiseGenerateReq, user = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Generate noise residual overlays and FFT overlays for sampled frames.
    Returns per-frame noise metrics and a simple noise_score (0..1) with high-confidence indices.

    Response: { noise_frames: [{ uri, fft_uri?, time_s?, index?, residual_abs_mean, residual_std, residual_energy, fft_low_ratio, fft_high_ratio, noise_score }], count, high_indices, threshold }
    """
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    import json, time
    from PIL import Image

    # Auth + resolve file
    asset = db.query(models_db.Asset).filter(models_db.Asset.id == req.asset_id, models_db.Asset.user_id == user.id).one_or_none()
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    stored_path = asset.stored_path
    abspath = None
    temp_download = None
    if stored_path:
        abspath = RAW_ROOT / stored_path
        if not abspath.exists():
            if asset.remote_key:
                try:
                    temp_download = STORAGE.download_to_temp(asset.remote_key)
                    abspath = temp_download
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Failed to retrieve remote asset: {e}")
            else:
                raise HTTPException(status_code=404, detail="Stored file missing")
    elif asset.remote_key:
        try:
            temp_download = STORAGE.download_to_temp(asset.remote_key)
            abspath = temp_download
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to retrieve remote asset: {e}")
    else:
        raise HTTPException(status_code=400, detail="Asset has no stored path")

    key = asset.sha256 or f"asset_{asset.id}"
    out_resid = DERIVED_DIR / "noise" / key
    out_fft = DERIVED_DIR / "noise_fft" / key
    out_resid.mkdir(parents=True, exist_ok=True)
    out_fft.mkdir(parents=True, exist_ok=True)
    meta_path = out_resid / "meta.json"

    def _noise_metrics_from_gray(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        # Residual via Gaussian blur subtraction
        g = gray.astype(np.float32)
        blur = cv2.GaussianBlur(g, (0,0), sigmaX=1.0, sigmaY=1.0)
        resid = g - blur
        resid_abs = np.abs(resid)
        # Metrics
        abs_mean = float(resid_abs.mean())
        std = float(resid.std())
        energy = float(np.sqrt((resid**2).mean()))
        # FFT energy ratios
        f = np.fft.fft2(g)
        fshift = np.fft.fftshift(f)
        mag = np.log(1+np.abs(fshift))
        h, w = mag.shape
        cy, cx = h//2, w//2
        r = int(min(h,w) * 0.12)  # radius for low-frequency circle ~12%
        yy, xx = np.ogrid[:h, :w]
        mask_low = (yy-cy)**2 + (xx-cx)**2 <= r*r
        low = float(mag[mask_low].sum())
        total = float(mag.sum()) + 1e-6
        low_ratio = max(0.0, min(1.0, low/total))
        high_ratio = max(0.0, min(1.0, 1.0 - low_ratio))
        # score: emphasize high-frequency ratio and residual energy (normalized by 255)
        score = max(0.0, min(1.0, 0.5*high_ratio + 0.5*min(1.0, energy/60.0)))
        # Overlays for visualization
        resid_vis = (np.clip(resid_abs * (255.0/np.maximum(1.0, resid_abs.mean()*4.0)), 0, 255)).astype(np.uint8)
        resid_cm = cv2.applyColorMap(resid_vis, cv2.COLORMAP_TWILIGHT)
        mag_norm = (np.clip((mag/np.max(mag)) * 255.0, 0, 255)).astype(np.uint8)
        fft_cm = cv2.applyColorMap(mag_norm, cv2.COLORMAP_INFERNO)
        metrics = {
            "residual_abs_mean": abs_mean,
            "residual_std": std,
            "residual_energy": energy,
            "fft_low_ratio": low_ratio,
            "fft_high_ratio": high_ratio,
            "noise_score": score,
        }
        return resid_cm, fft_cm, metrics

    frames_target = int(req.frames or 12)
    method = (req.method or "both").lower()
    threshold = float(req.threshold or 0.6)
    items: list[dict] = []
    try:
        mime = (asset.mime or '').lower()
        if mime.startswith('video/'):
            cap = cv2.VideoCapture(str(abspath))
            if not cap.isOpened():
                raise HTTPException(status_code=500, detail="Failed to open video for noise")
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            n = max(1, min(frames_target, total if total > 0 else frames_target))
            indices = []
            if total and total > 0:
                for i in range(n):
                    idx = int(round((i + 0.5) * (total / n)))
                    idx = max(0, min(total - 1, idx))
                    indices.append(idx)
            else:
                indices = list(range(n))
            used = set()
            for idx in indices:
                if idx in used: continue
                used.add(idx)
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resid_img, fft_img, m = _noise_metrics_from_gray(gray)
                t = (idx / fps) if (fps and fps > 0) else None
                name = f"f_{idx:06d}.png"
                resid_path = out_resid / name
                fft_path = out_fft / name
                # Save overlays per requested method
                if method in ("residual","both"):
                    cv2.imwrite(str(resid_path), resid_img)
                if method in ("fft","both"):
                    cv2.imwrite(str(fft_path), fft_img)
                items.append({
                    "uri": f"noise/{key}/{name}",
                    "fft_uri": f"noise_fft/{key}/{name}",
                    "time_s": (float(t) if t is not None else None),
                    "index": int(idx),
                    **m,
                })
            cap.release()
        else:
            pil = Image.open(str(abspath)).convert('L')
            import numpy as _np
            gray = _np.array(pil, dtype=_np.uint8)
            resid_img, fft_img, m = _noise_metrics_from_gray(gray)
            name = Path(stored_path).name
            stem = Path(name).stem
            resid_path = out_resid / f"{stem}_noise.png"
            fft_path = out_fft / f"{stem}_noise_fft.png"
            if method in ("residual","both"):
                cv2.imwrite(str(resid_path), resid_img)
            if method in ("fft","both"):
                cv2.imwrite(str(fft_path), fft_img)
            items.append({
                "uri": f"noise/{key}/{resid_path.name}",
                "fft_uri": f"noise_fft/{key}/{fft_path.name}",
                "time_s": None,
                "index": None,
                **m,
            })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Noise generation failed: {e}")
    finally:
        try:
            if temp_download and Path(temp_download).exists():
                Path(temp_download).unlink()
        except Exception:
            pass

    # Persist metadata for quick listing
    try:
        meta = {"frames": items, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
    except Exception:
        pass
    high = [it.get("index") for it in items if (it.get("noise_score") or 0) >= threshold and it.get("index") is not None]
    return {"noise_frames": items, "count": len(items), "high_indices": high, "threshold": threshold}

@app.get("/api/noise/list")
def list_noise_frames(asset_id: int, user = Depends(get_current_user), db: Session = Depends(get_db)):
    """Return any noise frames and metrics for the asset if previously generated (reads meta.json)."""
    import json, re
    asset = db.query(models_db.Asset).filter(models_db.Asset.id == asset_id, models_db.Asset.user_id == user.id).one_or_none()
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    key = (asset.sha256 or f"asset_{asset.id}")
    out_resid = DERIVED_DIR / "noise" / key
    meta_path = out_resid / "meta.json"
    if meta_path.exists():
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            frames = data.get("frames") or []
            return {"noise_frames": frames, "count": len(frames)}
        except Exception:
            pass
    # Fallback: build from file list without metrics
    items: list[dict] = []
    if not out_resid.exists():
        return {"noise_frames": [], "count": 0}
    pat = re.compile(r"^f_(\d+)\.png$", re.IGNORECASE)
    for p in sorted(out_resid.glob("*.png")):
        name = p.name
        m = pat.match(name)
        idx = int(m.group(1)) if m else None
        items.append({"uri": f"noise/{key}/{name}", "fft_uri": f"noise_fft/{key}/{name}", "time_s": None, "index": idx})
    items.sort(key=lambda d: (d.get("index") is None, d.get("index", 0), d.get("uri")))
    return {"noise_frames": items, "count": len(items)}

class NoiseGenerateByPathReq(BaseModel):
    stored_path: str
    sha256: str | None = None
    frames: int | None = 12
    method: str | None = "both"
    threshold: float | None = 0.6

@app.post("/api/noise/generate-by-path")
def generate_noise_frames_by_path(req: NoiseGenerateByPathReq, user = Depends(get_current_user)):
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    import json, time
    from PIL import Image
    from pathlib import Path as _P
    # Resolve path
    try:
        sp = (req.stored_path or "").lstrip("/")
        if not sp:
            raise HTTPException(status_code=400, detail="stored_path required")
        abspath = (RAW_ROOT / sp).resolve()
        root = RAW_ROOT.resolve()
        if not str(abspath).startswith(str(root)):
            raise HTTPException(status_code=400, detail="Invalid stored_path")
        if not abspath.exists():
            raise HTTPException(status_code=404, detail="File not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad stored_path: {e}")

    parts = _P(sp).parts
    key = (req.sha256 or (parts[0] if parts else "file"))
    out_resid = DERIVED_DIR / "noise" / key
    out_fft = DERIVED_DIR / "noise_fft" / key
    out_resid.mkdir(parents=True, exist_ok=True)
    out_fft.mkdir(parents=True, exist_ok=True)
    meta_path = out_resid / "meta.json"

    def _noise_metrics_from_gray(gray: np.ndarray):
        g = gray.astype(np.float32)
        blur = cv2.GaussianBlur(g, (0,0), sigmaX=1.0, sigmaY=1.0)
        resid = g - blur
        resid_abs = np.abs(resid)
        abs_mean = float(resid_abs.mean())
        std = float(resid.std())
        energy = float(np.sqrt((resid**2).mean()))
        f = np.fft.fft2(g)
        fshift = np.fft.fftshift(f)
        mag = np.log(1+np.abs(fshift))
        h, w = mag.shape
        cy, cx = h//2, w//2
        r = int(min(h,w) * 0.12)
        yy, xx = np.ogrid[:h, :w]
        mask_low = (yy-cy)**2 + (xx-cx)**2 <= r*r
        low = float(mag[mask_low].sum())
        total = float(mag.sum()) + 1e-6
        low_ratio = max(0.0, min(1.0, low/total))
        high_ratio = max(0.0, min(1.0, 1.0 - low_ratio))
        score = max(0.0, min(1.0, 0.5*high_ratio + 0.5*min(1.0, energy/60.0)))
        resid_vis = (np.clip(resid_abs * (255.0/np.maximum(1.0, resid_abs.mean()*4.0)), 0, 255)).astype(np.uint8)
        resid_cm = cv2.applyColorMap(resid_vis, cv2.COLORMAP_TWILIGHT)
        mag_norm = (np.clip((mag/np.max(mag)) * 255.0, 0, 255)).astype(np.uint8)
        fft_cm = cv2.applyColorMap(mag_norm, cv2.COLORMAP_INFERNO)
        metrics = {
            "residual_abs_mean": abs_mean,
            "residual_std": std,
            "residual_energy": energy,
            "fft_low_ratio": low_ratio,
            "fft_high_ratio": high_ratio,
            "noise_score": score,
        }
        return resid_cm, fft_cm, metrics

    frames_target = int(req.frames or 12)
    method = (req.method or "both").lower()
    threshold = float(req.threshold or 0.6)
    items: list[dict] = []
    try:
        ext = abspath.suffix.lower().strip('.')
        is_video = ext in {"mp4","mov","mkv","avi","webm","m4v"}
        if is_video:
            cap = cv2.VideoCapture(str(abspath))
            if not cap.isOpened():
                raise HTTPException(status_code=500, detail="Failed to open video for noise")
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            n = max(1, min(frames_target, total if total > 0 else frames_target))
            indices = []
            if total and total > 0:
                for i in range(n):
                    idx = int(round((i + 0.5) * (total / n)))
                    idx = max(0, min(total - 1, idx))
                    indices.append(idx)
            else:
                indices = list(range(n))
            used = set()
            for idx in indices:
                if idx in used: continue
                used.add(idx)
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resid_img, fft_img, m = _noise_metrics_from_gray(gray)
                t = (idx / fps) if (fps and fps > 0) else None
                name = f"f_{idx:06d}.png"
                resid_path = out_resid / name
                fft_path = out_fft / name
                if method in ("residual","both"):
                    cv2.imwrite(str(resid_path), resid_img)
                if method in ("fft","both"):
                    cv2.imwrite(str(fft_path), fft_img)
                items.append({"uri": f"noise/{key}/{name}", "fft_uri": f"noise_fft/{key}/{name}", "time_s": (float(t) if t is not None else None), "index": int(idx), **m})
            cap.release()
        else:
            gray = Image.open(str(abspath)).convert('L')
            arr = np.array(gray, dtype=np.uint8)
            resid_img, fft_img, m = _noise_metrics_from_gray(arr)
            stem = _P(abspath.name).stem
            resid_path = out_resid / f"{stem}_noise.png"
            fft_path = out_fft / f"{stem}_noise_fft.png"
            if method in ("residual","both"):
                cv2.imwrite(str(resid_path), resid_img)
            if method in ("fft","both"):
                cv2.imwrite(str(fft_path), fft_img)
            items.append({"uri": f"noise/{key}/{resid_path.name}", "fft_uri": f"noise_fft/{key}/{fft_path.name}", "time_s": None, "index": None, **m})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Noise generation failed: {e}")

    try:
        meta = {"frames": items, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
    except Exception:
        pass
    high = [it.get("index") for it in items if (it.get("noise_score") or 0) >= threshold and it.get("index") is not None]
    return {"noise_frames": items, "count": len(items), "high_indices": high, "threshold": threshold}

@app.get("/api/noise/list-by-path")
def list_noise_frames_by_path(stored_path: str, sha256: str | None = None, user = Depends(get_current_user)):
    import json, re
    from pathlib import Path as _P
    sp = (stored_path or "").lstrip("/")
    if not sp:
        raise HTTPException(status_code=400, detail="stored_path required")
    parts = _P(sp).parts
    key = (sha256 or (parts[0] if parts else "file"))
    out_resid = DERIVED_DIR / "noise" / key
    meta_path = out_resid / "meta.json"
    if meta_path.exists():
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            frames = data.get("frames") or []
            return {"noise_frames": frames, "count": len(frames)}
        except Exception:
            pass
    # Fallback
    if not out_resid.exists():
        return {"noise_frames": [], "count": 0}
    items: list[dict] = []
    pat = re.compile(r"^f_(\d+)\.png$", re.IGNORECASE)
    for p in sorted(out_resid.glob("*.png")):
        name = p.name
        m = pat.match(name)
        idx = int(m.group(1)) if m else None
        items.append({"uri": f"noise/{key}/{name}", "fft_uri": f"noise_fft/{key}/{name}", "time_s": None, "index": idx})
    items.sort(key=lambda d: (d.get("index") is None, d.get("index", 0), d.get("uri")))
    return {"noise_frames": items, "count": len(items)}

# Fallback: allow generating ELA by stored_path for ad-hoc analyses (no DB asset)
class ELAGenerateByPathReq(BaseModel):
    stored_path: str
    sha256: str | None = None
    frames: int | None = 12
    quality: int | None = 90
    scale: float | None = 10.0

@app.post("/api/ela/generate-by-path")
def generate_ela_frames_by_path(req: ELAGenerateByPathReq, user = Depends(get_current_user)):
    """
    Generate ELA frames for a file referenced by its ingest stored_path (e.g., "<sha>/<name>").
    This is used for analyses created via /api/analyze that do not persist an Asset row.
    """
    from PIL import Image, ImageChops
    import io
    import cv2  # type: ignore
    from pathlib import Path as _P

    # Resolve and validate path under RAW_ROOT
    try:
        sp = (req.stored_path or "").lstrip("/")
        if not sp:
            raise HTTPException(status_code=400, detail="stored_path required")
        abspath = (RAW_ROOT / sp).resolve()
        root = RAW_ROOT.resolve()
        if not str(abspath).startswith(str(root)):
            raise HTTPException(status_code=400, detail="Invalid stored_path")
        if not abspath.exists():
            raise HTTPException(status_code=404, detail="File not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad stored_path: {e}")

    # Pick output key: prefer provided sha256; fallback to first directory segment
    parts = _P(sp).parts
    key = (req.sha256 or (parts[0] if parts else "file"))
    out_dir = DERIVED_DIR / "ela" / key
    out_dir.mkdir(parents=True, exist_ok=True)

    def _ela_from_pil(pil_img: Image.Image, quality: int, scale: float) -> Image.Image:
        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG', quality=max(1, min(100, int(quality))))
        buf.seek(0)
        jpeg_img = Image.open(buf).convert('RGB')
        diff = ImageChops.difference(pil_img.convert('RGB'), jpeg_img)
        def _mul(x: int) -> int:
            v = int(float(x) * float(scale))
            return 255 if v > 255 else (0 if v < 0 else v)
        return diff.point(_mul)

    quality = int(req.quality or 90)
    scale = float(req.scale or 10.0)
    frames_target = int(req.frames or 12)

    ela_items: list[dict] = []
    try:
        ext = abspath.suffix.lower().strip('.')
        is_video = ext in {"mp4","mov","mkv","avi","webm","m4v"}
        if is_video:
            cap = cv2.VideoCapture(str(abspath))
            if not cap.isOpened():
                raise HTTPException(status_code=500, detail="Failed to open video for ELA")
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            n = max(1, min(frames_target, total if total > 0 else frames_target))
            indices = []
            if total and total > 0:
                for i in range(n):
                    idx = int(round((i + 0.5) * (total / n)))
                    idx = max(0, min(total - 1, idx))
                    indices.append(idx)
            else:
                indices = list(range(n))
            used = set()
            for idx in indices:
                if idx in used: continue
                used.add(idx)
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                ela_img = _ela_from_pil(pil, quality, scale)
                t = (idx / fps) if (fps and fps > 0) else None
                name = f"f_{idx:06d}.png"
                out_path = out_dir / name
                ela_img.save(str(out_path), format='PNG')
                ela_items.append({"uri": f"ela/{key}/{name}", "time_s": (float(t) if t is not None else None), "index": int(idx)})
            cap.release()
        else:
            pil = Image.open(str(abspath)).convert('RGB')
            ela_img = _ela_from_pil(pil, quality, scale)
            stem = _P(abspath.name).stem
            out_path = out_dir / f"{stem}_ela.png"
            ela_img.save(str(out_path), format='PNG')
            ela_items.append({"uri": f"ela/{key}/{out_path.name}", "time_s": None, "index": None})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ELA generation failed: {e}")

    return {"ela_frames": ela_items, "count": len(ela_items)}

# List existing ELA frames by stored_path/sha (no regeneration)
@app.get("/api/ela/list-by-path")
def list_ela_frames_by_path(stored_path: str, sha256: str | None = None, user = Depends(get_current_user)):
    """
    Return ELA frames that already exist for the given stored_path/sha.
    Uses derived/ela/<key> directory where key is sha256 or first segment of stored_path.
    """
    import re
    from pathlib import Path as _P
    sp = (stored_path or "").lstrip("/")
    if not sp:
        raise HTTPException(status_code=400, detail="stored_path required")
    parts = _P(sp).parts
    key = (sha256 or (parts[0] if parts else "file"))
    out_dir = DERIVED_DIR / "ela" / key
    if not out_dir.exists():
        return {"ela_frames": [], "count": 0}
    items: list[dict] = []
    pat = re.compile(r"^f_(\d+)\.png$", re.IGNORECASE)
    for p in sorted(out_dir.glob("*.png")):
        name = p.name
        m = pat.match(name)
        idx = int(m.group(1)) if m else None
        items.append({
            "uri": f"ela/{key}/{name}",
            "time_s": None,
            "index": idx,
        })
    items.sort(key=lambda d: (d.get("index") is None, d.get("index", 0), d.get("uri")))
    return {"ela_frames": items, "count": len(items)}

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
        try:
            from .models.copy_move_live import predict_copy_move_single  # type: ignore
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Copy-Move model unavailable: {e}")
        df_pred = predict_copy_move_single(video_abs, sha256=ingest_rec["sha256"])  # returns score/label/method + cm_*
    elif model in ("lbp", "lbp_rf", "lbp-model"):
        try:
            from .models.lbp_live import predict_lbp_single  # type: ignore
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LBP model unavailable: {e}")
        df_pred = predict_lbp_single(video_abs)
    elif model in ("fusion_blend", "dl_blend"):
        try:
            from .models.fusion_live import predict_fusion_single
            df_pred = predict_fusion_single(video_abs)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Deep learning (blend) unavailable: {e}")
    elif model in ("fusion_lr", "dl_lr"):
        try:
            from .models.fusion_lr_live import predict_fusion_lr_single
            df_pred = predict_fusion_lr_single(video_abs)
        except RuntimeError as e:
            # LR weights missing or invalid
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Deep learning (LR) unavailable: {e}")
    elif model in ("dl", "deep", "deep_learning", "full", "fusion"):
        # Auto: prefer logistic regression fusion if weights exist; else blend
        try:
            from pathlib import Path as _P
            from .models.fusion_lr_live import predict_fusion_lr_single
            from .models.fusion_live import predict_fusion_single
            _root = _P(__file__).resolve().parents[1] / "models"
            lr_path = _root / "fusion_lr.json"
            df_pred = predict_fusion_lr_single(video_abs) if lr_path.exists() else predict_fusion_single(video_abs)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Deep learning model unavailable: {e}")
    else:
        # Explicitly request stub path so the method is reported as 'stub-hash'
        df_pred = predict_deepfake(video_abs, sha256=ingest_rec["sha256"], method="stub")  # stub fallback

    # 5) Summary
    sum_probe = summarize_ffprobe((probe_rec or {}).get("probe")) if probe_rec else {}
    _write_progress(job_id, "summary", 85, "Summarizing")
    # Normalize and choose the model score consistently. Models may return different keys
    # (e.g., fused_score, deepfake_score, prob_fake, fake_prob, score). Prefer fused_score
    # when available. Also support scores expressed as percentages (0..100).
    def _pick_score(pred: dict) -> float | None:
        if not isinstance(pred, dict):
            return None
        keys = ["fused_score", "deepfake_score", "prob_fake", "fake_prob", "score"]
        for k in keys:
            v = pred.get(k)
            if v is None:
                continue
            try:
                fv = float(v)
            except Exception:
                continue
            # If value looks like a percent (0-100), scale it to 0..1
            if fv > 1 and fv <= 100:
                fv = fv / 100.0
            # clamp to [0,1)
            fv = max(0.0, min(0.999999, fv))
            return fv
        # fallback to 'score' property or df_pred.get('score') if present
        try:
            v = pred.get("score")
            if v is None:
                return None
            fv = float(v)
            if fv > 1 and fv <= 100:
                fv = fv / 100.0
            return max(0.0, min(0.999999, fv))
        except Exception:
            return None

    picked_score = _pick_score(df_pred or {})
    # Default decision threshold (can be tuned per-model later)
    decision_threshold = 0.5
    label_from_score = None
    if picked_score is not None:
        label_from_score = "fake" if picked_score >= decision_threshold else "real"

    summary = {
        "width": sum_probe.get("width"),
        "height": sum_probe.get("height"),
        "fps": sum_probe.get("fps"),
        "duration_s": sum_probe.get("duration_s"),
        "codec": sum_probe.get("codec"),
        "format_valid": validate_rec.get("format_valid") if validate_rec else None,
        "decode_valid": validate_rec.get("decode_valid") if validate_rec else None,
        "errors": validate_rec.get("errors") if validate_rec else [],
        "deepfake_likelihood": (picked_score if picked_score is not None else df_pred.get("score")),
        "deepfake_label": (df_pred.get("label") or label_from_score),
        "deepfake_method": df_pred.get("method"),
        "decision_threshold": decision_threshold,
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
    # Bubble up fusion components debug if present
    if df_pred.get("components") is not None:
        summary["components"] = df_pred["components"]

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

    # Resolve local path for this asset. If it's only in remote storage, download to temp
    video_abs = Path(RAW_ROOT) / (a.stored_path or "")
    temp_path = None
    store_root_for_meta = str(RAW_ROOT)
    stored_path_for_meta = a.stored_path or ""
    # Consider a missing or directory-only path as not having a local file
    has_local_file = bool(a.stored_path) and video_abs.is_file()
    if (not has_local_file) and a.remote_key:
        try:
            _write_progress(job_id, "download", 18, "Fetching remote asset")
            temp_path = STORAGE.download_to_temp(a.remote_key)
            video_abs = temp_path
            store_root_for_meta = str(Path(temp_path).parent)
            stored_path_for_meta = Path(temp_path).name
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to retrieve remote asset: {e}")
    elif not has_local_file and not a.remote_key:
        # No local file and no remote reference; cannot proceed
        raise HTTPException(status_code=404, detail="Asset content not available")

    # Emit a quick check so the UI can see path details
    try:
        size_bytes = None
        try:
            size_bytes = Path(video_abs).stat().st_size if Path(video_abs).exists() else None
        except Exception:
            size_bytes = None
        _write_progress(
            job_id,
            "prepare",
            20,
            f"Path resolved: {str(video_abs)} | exists={Path(video_abs).exists()} | size={size_bytes}"
        )
    except Exception:
        pass

    # Build ingest-like record (reflecting the path used for analysis/probing)
    ingest_rec = {
        "action": "ingest",
        "sha256": a.sha256,
        "stored_path": stored_path_for_meta,
        "store_root": store_root_for_meta,
        "mime": a.mime,
        "asset_id": a.id,
    }

    # Validate & probe using the resolved local path (RAW or temp)
    _write_progress(job_id, "validate", 22, "Validating format & decode")
    validate_rec = validate_asset(stored_path_for_meta, store_root_for_meta, a.sha256 or "", a.mime)
    _write_progress(job_id, "probe", 38, "Probing media")
    probe_rec = probe_asset(stored_path_for_meta, store_root_for_meta, a.sha256 or "", a.mime, no_exif=True)

    # Model inference
    model = (req.model or "stub").lower().strip()

    _write_progress(job_id, "model", 65, f"Running model: {model}")
    if model in ("copy_move", "copymove", "cm"):
        try:
            from .models.copy_move_live import predict_copy_move_single  # type: ignore
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Copy-Move model unavailable: {e}")
        df_pred = predict_copy_move_single(video_abs, sha256=(a.sha256 or ""))
    elif model in ("lbp", "lbp_rf", "lbp-model"):
        try:
            from .models.lbp_live import predict_lbp_single  # type: ignore
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LBP model unavailable: {e}")
        df_pred = predict_lbp_single(video_abs)
    elif model in ("fusion_blend", "dl_blend"):
        try:
            from .models.fusion_live import predict_fusion_single
            df_pred = predict_fusion_single(video_abs)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Deep learning (blend) unavailable: {e}")
    elif model in ("fusion_lr", "dl_lr"):
        try:
            from .models.fusion_lr_live import predict_fusion_lr_single
            df_pred = predict_fusion_lr_single(video_abs)
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Deep learning (LR) unavailable: {e}")
    elif model in ("dl", "deep", "deep_learning", "full", "fusion"):
        try:
            from pathlib import Path as _P
            from .models.fusion_lr_live import predict_fusion_lr_single
            from .models.fusion_live import predict_fusion_single
            _root = _P(__file__).resolve().parents[1] / "models"
            lr_path = _root / "fusion_lr.json"
            df_pred = predict_fusion_lr_single(video_abs) if lr_path.exists() else predict_fusion_single(video_abs)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Deep learning model unavailable: {e}")
    else:
        # Explicitly request stub path so the method is reported as 'stub-hash'
        df_pred = predict_deepfake(video_abs, sha256=(a.sha256 or ""), method="stub")
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
    if df_pred.get("components") is not None:
        summary["components"] = df_pred["components"]

    _write_progress(job_id, "done", 100, "Completed")
    return AnalyzeResponse(asset=ingest_rec, validate=validate_rec, probe=probe_rec, summary=summary)

# ------------------------- Media Asset Viewing -------------------------
@app.get("/api/v1/asset/{asset_id}/view")
def view_asset(asset_id: int, db: Session = Depends(get_db)):
    """
    Get a media asset for viewing
    """
    db_asset = db.query(models_db.Asset).filter(models_db.Asset.id == asset_id).first()
    if not db_asset:
        raise HTTPException(status_code=404, detail="Asset not found")

    # TODO: check permissions
    # if db_asset.user_id != current_user.id:
    #     raise HTTPException(status_code=403, detail="Not authorized")

    file_path = os.path.join(RAW_ROOT, db_asset.stored_path)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

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

# ------------------------ User Pages Persistence ---------------------------
@app.get("/api/pages", response_model=PagesPayload)
def get_pages(user = Depends(get_current_user)):
    for db in get_db():
        sess: Session = db
        row = sess.query(models_db.UserPrefs).filter(models_db.UserPrefs.user_id == user.id).first()
        if not row or not (row.pages_json or '').strip():
            return PagesPayload(pages=[])
        try:
            data = __import__('json').loads(row.pages_json or '[]')
        except Exception:
            data = []
        # Coerce entries to expected shape
        pages = []
        for p in data:
            if isinstance(p, dict) and isinstance(p.get('key'), str) and isinstance(p.get('label'), str):
                pages.append(PageEntry(key=p['key'], label=p['label'], icon=p.get('icon')))
        return PagesPayload(pages=pages)

@app.put("/api/pages", response_model=PagesPayload)
def put_pages(payload: PagesPayload, user = Depends(get_current_user)):
    # sanitize input
    pages = []
    for p in payload.pages:
        if not p.key or not p.label:
            continue
        pages.append({'key': p.key, 'label': p.label, 'icon': p.icon})
    txt = __import__('json').dumps(pages)
    now = int(__import__('time').time())
    for db in get_db():
        sess: Session = db
        row = sess.query(models_db.UserPrefs).filter(models_db.UserPrefs.user_id == user.id).first()
        if row:
            row.pages_json = txt
            row.updated_at = now
            sess.add(row)
            sess.commit()
        else:
            row = models_db.UserPrefs(user_id=user.id, pages_json=txt, updated_at=now)
            sess.add(row)
            sess.commit()
        return PagesPayload(pages=[PageEntry(**p) for p in pages])

# Convenience root
@app.get("/")
def root():
    return {"service": "blackline-api", "endpoints": ["/api/health", "/api/analyze"]}
