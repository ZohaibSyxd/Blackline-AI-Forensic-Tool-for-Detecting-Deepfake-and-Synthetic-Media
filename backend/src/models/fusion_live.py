from __future__ import annotations
"""
Lightweight single-asset fusion inference for the web API.

This runs:
  - Xception (timm) on a small, evenly-sampled set of frames
  - TimeSformer on a short clip worth of frames
Then fuses the two probabilities with a simple blend (avg) to produce a
"full deep learning" score. This is intended for online predictions in the
API and mirrors the offline scripts (xception_infer.py, timesformer_infer.py)
at a smaller scale suitable for real-time usage.

Returned dict contains at least keys: {score, label, method} and exposes
sub-scores for transparency.
"""
import json
from pathlib import Path
import os
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

# Try decord first for efficient sampling; fall back to OpenCV
try:
    import decord
    from decord import VideoReader, cpu as decord_cpu
    HAS_DECORD = True
except Exception:  # pragma: no cover - optional
    HAS_DECORD = False
    import cv2

from .xception_infer import XceptionDeepfakeDetector, IMAGENET_MEAN, IMAGENET_STD
from torchvision import transforms

from transformers import AutoImageProcessor, TimesformerModel, TimesformerConfig
import torch.nn as nn


# ------------------------ Model singletons ------------------------
_xception: Optional[XceptionDeepfakeDetector] = None
_ts_processor = None
_ts_model: Optional[nn.Module] = None

TS_DEFAULT_MODEL_ID = "facebook/timesformer-base-finetuned-k400"


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_xception(checkpoint: Optional[Path] = None) -> XceptionDeepfakeDetector:
    global _xception
    if _xception is None:
        _xception = XceptionDeepfakeDetector(checkpoint_path=str(checkpoint) if checkpoint else None,
                                             device=str(pick_device()))
    return _xception


class TSBinary(nn.Module):
    def __init__(self, base: TimesformerModel, hidden: int):
        super().__init__()
        self.base = base
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden, 1))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.base(pixel_values=pixel_values, return_dict=True)
        tokens = out.last_hidden_state  # (B,N,H)
        pooled = tokens[:, 0]
        logit = self.head(pooled)
        return logit.squeeze(1)


def _ensure_timesformer(config_json: Optional[Path] = None, checkpoint: Optional[Path] = None):
    global _ts_processor, _ts_model
    if _ts_processor is not None and _ts_model is not None:
        return _ts_processor, _ts_model
    model_name = TS_DEFAULT_MODEL_ID
    frames = 8
    size = 224
    if config_json and config_json.exists():
        try:
            cfg = json.loads(config_json.read_text(encoding="utf-8"))
            model_name = cfg.get("model_name", model_name)
            frames = int(cfg.get("frames", frames))
            size = int(cfg.get("size", size))
        except Exception:
            pass
    processor = None
    base = None
    # Prefer HF weights; fallback to local config offline if unavailable
    try:
        processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)
        base = TimesformerModel.from_pretrained(model_name, use_safetensors=True)
    except Exception:
        try:
            if config_json and config_json.exists():
                cfg_dict = json.loads(config_json.read_text(encoding="utf-8"))
                config = TimesformerConfig(**cfg_dict) if isinstance(cfg_dict, dict) else TimesformerConfig()
            else:
                config = TimesformerConfig()
            base = TimesformerModel(config)
            processor = None  # we'll build pixel_values manually
        except Exception:
            # leave base None to signal failure to caller
            base = None
    hidden = base.config.hidden_size
    model = TSBinary(base, hidden)
    if checkpoint and checkpoint.exists():
        try:
            ck = torch.load(str(checkpoint), map_location="cpu")
            model.load_state_dict(ck, strict=False)
        except Exception:
            pass
    model.eval().to(pick_device())
    _ts_processor = processor
    _ts_model = model
    return processor, model


def _sample_frames_video_with_meta(vpath: Path, num: int) -> Tuple[List[Image.Image], List[int], float, int]:
    """Sample ~num frames evenly from the video.
    Returns (images, frame_indices, fps, total_frames)
    """
    if HAS_DECORD:
        try:
            vr = VideoReader(str(vpath), ctx=decord_cpu(0))
            if len(vr) <= 0:
                return [], [], 0.0, 0
            import numpy as np
            idxs = np.linspace(0, len(vr) - 1, num=num, dtype=int)
            arr = vr.get_batch(idxs).asnumpy()
            fps = 0.0
            try:
                fps = float(vr.get_avg_fps())
            except Exception:
                fps = 0.0
            return [Image.fromarray(fr) for fr in arr], idxs.tolist(), fps, int(len(vr))
        except Exception:
            pass
    # Fallback: OpenCV
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        return [], [], 0.0, 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    out: List[Image.Image] = []
    idxs: List[int] = []
    if total > 0:
        import numpy as np
        samp = np.linspace(0, total - 1, num=num, dtype=int)
        for i in samp:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out.append(Image.fromarray(frame))
            idxs.append(int(i))
    else:
        # Some OpenCV backends don't report frame count; fall back to sequential sampling.
        # Read frames with a simple stride until we collect `num` frames or hit EOF.
        stride = max(1, int(round(fps / 2.0))) if fps and fps > 0 else 5
        current_index = 0
        while len(out) < num:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out.append(Image.fromarray(frame_rgb))
            idxs.append(current_index)
            # Skip `stride-1` frames quickly
            for _ in range(stride - 1):
                if not cap.grab():
                    break
                current_index += 1
            current_index += 1
        # Best-effort estimate: total remains unknown -> 0
        total = 0
    cap.release()
    return out, idxs, fps, int(total)


_x_tf = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def _predict_xception_detail(vpath: Path, checkpoint: Optional[Path], frames: int = 16, agg: str = "p95") -> Tuple[float, List[float], List[int], float, int]:
    """Return (agg_score, per_frame_probs, sampled_indices, fps, total_frames)."""
    det = _ensure_xception(checkpoint)
    imgs, idxs, fps, total = _sample_frames_video_with_meta(vpath, frames)
    if not imgs:
        return 0.5, [], idxs, fps, total
    batch = torch.stack([_x_tf(im) for im in imgs], dim=0)
    with torch.no_grad():
        probs = det.predict_batch(batch).cpu().numpy().tolist()
    import numpy as np
    a = np.asarray(probs, float) if probs else np.asarray([0.5], float)
    if agg == "median":
        agg_score = float(np.median(a))
    elif agg == "max":
        agg_score = float(np.max(a))
    elif agg == "p90":
        agg_score = float(np.percentile(a, 90))
    elif agg == "p95":
        agg_score = float(np.percentile(a, 95))
    else:
        agg_score = float(np.mean(a))
    return agg_score, [float(x) for x in probs], idxs, fps, total


def _predict_timesformer(vpath: Path, config_json: Optional[Path], checkpoint: Optional[Path], frames: int = 8, size: int = 224) -> float:
    processor, model = _ensure_timesformer(config_json, checkpoint)
    if model is None:
        return 0.5
    imgs, _, _, _ = _sample_frames_video_with_meta(vpath, frames)
    if not imgs:
        return 0.5
    device = pick_device()
    with torch.no_grad():
        if processor is not None:
            inputs = processor([imgs], return_tensors="pt", size={"shortest_edge": size})
            pixel_values = inputs["pixel_values"].to(device)  # (1, T, C, H, W)
        else:
            # Manual preprocessing when HF processor is unavailable (offline)
            from torchvision import transforms as tvt
            tfm = tvt.Compose([tvt.Resize((size, size)), tvt.ToTensor(), tvt.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
            tensors = [tfm(im) for im in imgs]  # list (C,H,W)
            pixel_values = torch.stack(tensors, dim=0).unsqueeze(0).to(device)  # (1, T, C, H, W)
        logits = model(pixel_values)
        prob = torch.sigmoid(logits)[0].item()
        return float(prob)


def _resolve_models_root(models_root: Path | None = None) -> Path:
    """Resolve a robust models directory across different Docker contexts.
    Tries in order: explicit arg, $MODELS_DIR, ../../models (relative to this file),
    CWD/backend/models, ../../../backend/models.
    """
    if models_root and Path(models_root).exists():
        return Path(models_root)
    env_dir = os.getenv("MODELS_DIR")
    if env_dir and Path(env_dir).exists():
        return Path(env_dir)
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "models",                # /app/backend/models
        Path.cwd() / "backend" / "models",         # CWD-based
        here.parents[3] / "backend" / "models",    # /app/backend/backend/models (bad context)
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback to first candidate even if missing (to preserve path shape)
    return candidates[0]


def predict_fusion_single(video_path: Path, models_root: Path | None = None) -> Dict[str, object]:
    """Run Xception + TimeSformer and return a fused probability dict.

    models_root: folder containing model files (defaults to resolved models directory)
    """
    root = _resolve_models_root(models_root)
    x_ckpt = root / "xception_best.pth"
    ts_ckpt = root / "timesformer_best.pt"
    ts_cfg  = root / "timesformer_best.config.json"

    try:
        xs, xs_list, xs_idxs, xs_fps, xs_total = _predict_xception_detail(video_path, x_ckpt if x_ckpt.exists() else None, frames=24, agg="p95")
    except Exception:
        xs, xs_list, xs_idxs, xs_fps, xs_total = 0.5, [], [], 0.0, 0
    ts_valid = True
    try:
        ts = _predict_timesformer(video_path, ts_cfg if ts_cfg.exists() else None, ts_ckpt if ts_ckpt.exists() else None, frames=8, size=224)
    except Exception:
        ts = 0.5
        ts_valid = False

    xs_valid = bool(xs_list) and (xs_fps is not None)
    if xs_valid and ts_valid:
        fused = (xs + ts) / 2.0
    elif xs_valid:
        fused = xs
    elif ts_valid:
        fused = ts
    else:
        fused = 0.5
    label = "fake" if fused >= 0.5 else "real"
    # Prepare per-frame timeline for Xception samples
    frame_scores = []
    if xs_list and xs_idxs:
        for idx, prob in zip(xs_idxs, xs_list):
            t = (float(idx) / xs_fps) if xs_fps and xs_fps > 0 else None
            frame_scores.append({"index": int(idx), "time_s": t, "prob": float(prob)})

    out = {
        "score": float(fused),
        "label": label,
        "method": "fusion-blend(avg(xception,timesformer))",
        "xception_agg_score": float(xs),
        "timesformer_score": float(ts),
        "frame_scores": frame_scores,
        "frame_fps": float(xs_fps) if xs_fps else None,
        "frame_total": int(xs_total) if xs_total else None,
    }
    # Debug components: whether we believe each branch contributed meaningfully
    out["components"] = {
        "xception": {"ckpt_present": x_ckpt.exists(), "valid": xs_valid},
        "timesformer": {"ckpt_present": ts_ckpt.exists(), "valid": ts_valid},
    }
    return out
