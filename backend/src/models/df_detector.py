"""Stub deepfake detector.

This provides a placeholder likelihood score so the front-end can display a
"Deepfake Likelihood" percentage. Replace implementation with a real model
later (e.g., frame sampling + CNN/transformer inference).

Design choices for the stub:
Deterministic pseudo-score based on SHA-256 (stable across runs for same file)
Produces fields: {"score": float 0..1, "label": "real"|"fake", "method": "stub-hash"}

To extend:
Load model weights globally (module import time)
Add a predict(video_path: Path) -> dict that performs frame extraction & inference
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Dict

def _hash_to_score(sha256: str) -> float:
    # Take first 8 hex chars, convert to int, normalize to [0,1)
    try:
        n = int(sha256[:8], 16)
        return (n % 10_000_000) / 10_000_000.0  # 7 significant digits
    except Exception:
        return 0.5

def predict_deepfake(video_path: Path, sha256: str | None = None, method: str | None = None) -> Dict[str, object]:
    """Return a pseudo deepfake score.

    Args:
        video_path: path to the ingested video file.
        sha256: optional known content hash; if absent we derive from file bytes (slow for large files).
        method: optional method for prediction; if None, defaults to "fusion-blend".
    """
    # Lazily import heavy DL libs; guard imports so the API still works in dev
    predict_on_frames = None
    predict_on_clips = None
    get_video_frames = None
    get_video_clips = None
    try:
        from ..xception import predict_on_frames as _pof
        predict_on_frames = _pof
    except Exception as e:
        predict_on_frames = None
        xception_import_error = str(e)
    else:
        xception_import_error = None
    try:
        from ..timesformer import predict_on_clips as _pot
        predict_on_clips = _pot
    except Exception as e:
        predict_on_clips = None
        timesformer_import_error = str(e)
    else:
        timesformer_import_error = None
    try:
        from ..utils import get_video_frames as _gvf, get_video_clips as _gvc
        get_video_frames = _gvf
        get_video_clips = _gvc
    except Exception:
        # utils should normally be available; if not, leave None and let callers handle
        get_video_frames = None
        get_video_clips = None

    # Default to fusion-blend if no method is specified
    if method is None:
        method = "fusion-blend"

    # Component-wise prediction
    components = {}
    if "xception" in method:
        if predict_on_frames is None or get_video_frames is None:
            components["xception"] = {"valid": False, "error": f"xception unavailable: {xception_import_error or 'not installed'}"}
        else:
            try:
                frame_batches = get_video_frames(video_path, num_frames=16, batch_size=8)
                xception_results = predict_on_frames(frame_batches)
                components["xception"] = xception_results
            except Exception as e:
                components["xception"] = {"valid": False, "error": str(e)}

    if "timesformer" in method:
        if predict_on_clips is None or get_video_clips is None:
            components["timesformer"] = {"valid": False, "error": f"timesformer unavailable: {timesformer_import_error or 'not installed'}"}
        else:
            try:
                clips = get_video_clips(video_path, clip_len=16, num_clips=5)
                timesformer_results = predict_on_clips(clips)
                components["timesformer"] = timesformer_results
            except Exception as e:
                components["timesformer"] = {"valid": False, "error": str(e)}

    # Fusion
    if method == "fusion-blend":
# ... existing code ...
        x_score = components.get("xception", {}).get("agg_score", 0.5)
        t_score = components.get("timesformer", {}).get("agg_score", 0.5)
        # simple average blend
        score = (x_score + t_score) / 2.0
        label = "fake" if score >= 0.5 else "real"
        return {
            "score": score,
            "label": label,
            "method": "fusion-blend(avg(xception,timesformer))",
            "components": components,
            "xception_agg_score": x_score,
            "timesformer_score": t_score,
        }
    elif method == "xception":
        score = components.get("xception", {}).get("agg_score", 0.5)
        label = "fake" if score >= 0.5 else "real"
        return {
            "score": score,
            "label": label,
            "method": "xception",
            "components": components,
        }
    elif method == "timesformer":
        score = components.get("timesformer", {}).get("agg_score", 0.5)
        label = "fake" if score >= 0.5 else "real"
        return {
            "score": score,
            "label": label,
            "method": "timesformer",
            "components": components,
        }

    # Fallback to stub-hash if method is unknown or components failed
    if sha256 is None:
        h = hashlib.sha256()
        with open(video_path, 'rb') as f:
            for chunk in iter(lambda: f.read(1024*1024), b''):
                h.update(chunk)
        sha256 = h.hexdigest()
    score = _hash_to_score(sha256)
    label = 'fake' if score >= 0.5 else 'real'
    return {
        'score': score,
        'label': label,
        'method': 'stub-hash',
    }