from __future__ import annotations
"""
Live Logistic Regression fusion of Xception + TimeSformer for single-asset inference.

Loads weights from backend/models/fusion_lr.json (coef/intercept) and uses the
same quick sampling helpers as fusion_blend to produce per-asset scores online.

Output keys mirror other predictors: {score, label, method, xception_agg_score,
timesformer_score, frame_scores?, frame_fps?, frame_total?, components}
"""
import json
from pathlib import Path
from typing import Dict, Optional

import math

from .fusion_live import _predict_xception_detail, _predict_timesformer, _resolve_models_root


def _load_lr_weights(models_root: Path | None = None) -> Optional[Dict[str, float]]:
    root = _resolve_models_root(models_root)
    f = root / "fusion_lr.json"
    if not f.exists():
        return None
    try:
        obj = json.loads(f.read_text(encoding="utf-8"))
        coef = obj.get("coef") or obj.get("coefs")
        intercept = obj.get("intercept")
        if (not isinstance(coef, (list, tuple))) or len(coef) < 2 or intercept is None:
            return None
        return {"w0": float(coef[0]), "w1": float(coef[1]), "b": float(intercept)}
    except Exception:
        return None


def _sigmoid(x: float) -> float:
    try:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        else:
            z = math.exp(x)
            return z / (1.0 + z)
    except Exception:
        return 0.5


def predict_fusion_lr_single(video_path: Path, models_root: Path | None = None) -> Dict[str, object]:
    """Run quick Xception+TimeSformer then fuse via LR weights.

    If fusion_lr.json is missing or invalid, raises RuntimeError.
    """
    w = _load_lr_weights(models_root)
    if not w:
        raise RuntimeError("LR fusion weights not found (backend/models/fusion_lr.json)")

    root = _resolve_models_root(models_root)
    x_ckpt = root / "xception_best.pth"
    ts_ckpt = root / "timesformer_best.pt"
    ts_cfg  = root / "timesformer_best.config.json"

    try:
        xs, xs_list, xs_idxs, xs_fps, xs_total = _predict_xception_detail(video_path, x_ckpt if x_ckpt.exists() else None, frames=24, agg="p95")
        xs_valid = True
    except Exception:
        xs, xs_list, xs_idxs, xs_fps, xs_total = 0.5, [], [], 0.0, 0
        xs_valid = False
    try:
        ts = _predict_timesformer(video_path, ts_cfg if ts_cfg.exists() else None, ts_ckpt if ts_ckpt.exists() else None, frames=8, size=224)
        ts_valid = True
    except Exception:
        ts = 0.5
        ts_valid = False

    # LR fuse: logit = w0*xs + w1*ts + b
    logit = (w["w0"] * float(xs)) + (w["w1"] * float(ts)) + w["b"]
    prob = float(_sigmoid(logit))
    label = "fake" if prob >= 0.5 else "real"

    frame_scores = []
    if xs_list and xs_idxs:
        for idx, probf in zip(xs_idxs, xs_list):
            t = (float(idx) / xs_fps) if xs_fps and xs_fps > 0 else None
            frame_scores.append({"index": int(idx), "time_s": t, "prob": float(probf)})

    out = {
        "score": prob,
        "label": label,
        "method": "fusion-lr(xception,timesformer)",
        "xception_agg_score": float(xs),
        "timesformer_score": float(ts),
        "frame_scores": frame_scores,
        "frame_fps": float(xs_fps) if xs_fps else None,
        "frame_total": int(xs_total) if xs_total else None,
    }
    out["components"] = {
        "xception": {"ckpt_present": x_ckpt.exists(), "valid": xs_valid},
        "timesformer": {"ckpt_present": ts_ckpt.exists(), "valid": ts_valid},
    }
    return out
