"""
Live Copy-Move forgery detector for a single uploaded asset.

Implementation details:
- For videos: sample up to N frames uniformly across duration via OpenCV VideoCapture.
- For images: run once on the image.
- Use detect_copy_move() from compute_copy_move.py (ORB + translation clustering) to compute
  metrics and an overlay visualization highlighting suspected duplicated regions.
- Save best overlay under backend/data/derived/overlays/copy_move/<sha>/live_best.jpg
- Return a dict containing a normalized score in [0,1], label ('fake' if score>thr), method id,
  and a subset of metrics plus a relative overlay URI that the API serves under /static.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, Generator

import cv2
import numpy as np

from .compute_copy_move import detect_copy_move  # reuse algorithm

DERIVED_BASE = Path("backend/data/derived")
OVERLAYS_DIR = DERIVED_BASE / "overlays" / "copy_move"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _score_from_metrics(m: Dict[str, float]) -> float:
    """Combine copy-move metrics into a unit score.
    Heuristic: emphasize consensus (cm_confidence) and spatial extent (cm_coverage_ratio).
    """
    conf = float(m.get("cm_confidence", 0.0) or 0.0)
    cov = float(m.get("cm_coverage_ratio", 0.0) or 0.0)
    shift = float(m.get("cm_shift_magnitude", 0.0) or 0.0)
    # Normalize shift into [0,1] with soft cap at 40px (avoid rewarding tiny shifts)
    shift_norm = min(1.0, shift / 40.0)
    score = 0.6 * conf + 0.3 * min(1.0, cov * 6.0) + 0.1 * shift_norm
    return float(max(0.0, min(1.0, score)))


def _read_image_bgr(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


def _iter_sampled_frames(cap: cv2.VideoCapture, max_samples: int = 10) -> Generator[Tuple[int, np.ndarray], None, None]:
    """Yield (index, frame_bgr) for up to max_samples frames uniformly spaced across the stream."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        # Fallback: attempt to read sequentially
        idx = 0
        while idx < max_samples:
            ok, frame = cap.read()
            if not ok:
                break
            yield (idx, frame)
            idx += 1
        return
    step = max(1, total // max_samples)
    for idx in range(0, total, step):
        if max_samples <= 0:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        yield (idx, frame)
        max_samples -= 1


def predict_copy_move_single(asset_path: Path, sha256: str | None = None) -> Dict[str, object]:
    sha = sha256 or "nohash"
    out_dir = OVERLAYS_DIR / sha
    _ensure_dir(out_dir)

    best_score = 0.0
    best_metrics: Dict[str, float] = {}
    best_overlay: Path | None = None

    # Try treat as video first
    cap = cv2.VideoCapture(str(asset_path))
    if cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) > 1:
        for sample_idx, frame in _iter_sampled_frames(cap, max_samples=10):
            overlay, metrics = detect_copy_move(frame)
            # write temp overlay
            tmp_path = out_dir / f"live_{sample_idx:06d}.jpg"
            cv2.imwrite(str(tmp_path), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            s = _score_from_metrics(metrics)
            if s > best_score:
                best_score = s
                best_metrics = metrics
                best_overlay = tmp_path
        cap.release()
    else:
        # Image path
        cap.release()
        img = _read_image_bgr(asset_path)
        if img is not None:
            overlay, metrics = detect_copy_move(img)
            tmp_path = out_dir / "live_image.jpg"
            cv2.imwrite(str(tmp_path), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            best_score = _score_from_metrics(metrics)
            best_metrics = metrics
            best_overlay = tmp_path

    # If nothing processed, return neutral
    if best_overlay is None:
        return {
            "score": 0.0,
            "label": "real",
            "method": "copy-move-orb",
            "cm_confidence": 0.0,
            "cm_coverage_ratio": 0.0,
            "cm_shift_magnitude": 0.0,
            "cm_num_keypoints": 0,
            "cm_num_matches": 0,
        }

    # Keep only the best overlay to avoid clutter
    # (Optionally we could clean others, but they can help debugging; keep them for now.)
    rel_overlay = best_overlay.relative_to(DERIVED_BASE).as_posix()  # e.g., overlays/copy_move/<sha>/live_xxx.jpg

    label = "fake" if best_score >= 0.4 else "real"

    result: Dict[str, object] = {
        "score": best_score,
        "label": label,
        "method": "copy-move-orb",
        "overlay_uri": rel_overlay,
    }
    # merge subset of metrics
    for k in ("cm_confidence", "cm_coverage_ratio", "cm_shift_magnitude", "cm_num_keypoints", "cm_num_matches"):
        if k in best_metrics:
            result[k] = best_metrics[k]

    return result
