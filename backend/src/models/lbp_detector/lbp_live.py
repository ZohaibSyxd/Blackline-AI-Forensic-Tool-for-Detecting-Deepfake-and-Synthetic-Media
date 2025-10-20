"""
Live LBP-based deepfake detector for a single asset (image or video).

Pipeline:
1) Read frames: if video, uniformly sample up to N frames; if image, use once.
2) Preprocess: convert to grayscale, optionally resize to a manageable size.
3) Compute LBP histogram per frame using skimage.feature.local_binary_pattern
   with parameters (radius=2, neighbors=16, method="uniform"), then a normalized
   histogram with bins [0..neighbors+1].
4) Aggregate video descriptor as the mean histogram across frames.
5) Load a pre-trained scikit-learn classifier (joblib) and predict FAKE probability.

Returns:
  {
    "score": float in [0,1]  # probability of FAKE
    "label": "fake"|"real",
    "method": "lbp_rf",
    "lbp_frames": int,
    "lbp_dim": int
  }
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import cv2
import joblib


DEFAULT_MODEL_PATH = Path("backend/src/LBP/lbp_model_rf.joblib")


def _to_gray_uint8(bgr: np.ndarray, max_size: int = 512) -> np.ndarray:
    if bgr is None or bgr.size == 0:
        return np.zeros((1,1), dtype=np.uint8)
    h, w = bgr.shape[:2]
    scale = 1.0
    if max(h, w) > max_size:
        scale = max_size / float(max(h, w))
        nh, nw = int(h * scale), int(w * scale)
        bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return gray


def _lbp_hist(gray_u8: np.ndarray, radius: int = 2, neighbors: int = 16) -> np.ndarray:
    """Compute uniform LBP histogram with P=neighbors, R=radius.
    Returns a normalized histogram of length neighbors+2 (uniform bins 0..P plus one for non-uniform).
    """
    h, w = gray_u8.shape[:2]
    if h < 2*radius+1 or w < 2*radius+1:
        return np.zeros((neighbors + 2,), dtype=np.float32)

    # Precompute neighbor offsets (dx, dy) in image coordinates (x right, y down)
    thetas = np.linspace(0, 2*np.pi, neighbors, endpoint=False)
    dx = radius * np.cos(thetas)
    dy = -radius * np.sin(thetas)  # negative because y increases downward

    # Histogram of length P+2
    hist = np.zeros((neighbors + 2,), dtype=np.float64)

    # Iterate excluding radius border
    for y in range(radius, h - radius):
        for x in range(radius, w - radius):
            c = float(gray_u8[y, x])
            bits = []
            # Sample P neighbors with bilinear interpolation
            for k in range(neighbors):
                xf = x + dx[k]
                yf = y + dy[k]
                x0 = int(np.floor(xf)); x1 = x0 + 1
                y0 = int(np.floor(yf)); y1 = y0 + 1
                ax = xf - x0
                ay = yf - y0
                # Clamp indices (safety)
                x0c = min(max(0, x0), w-1); x1c = min(max(0, x1), w-1)
                y0c = min(max(0, y0), h-1); y1c = min(max(0, y1), h-1)
                v00 = float(gray_u8[y0c, x0c])
                v10 = float(gray_u8[y0c, x1c])
                v01 = float(gray_u8[y1c, x0c])
                v11 = float(gray_u8[y1c, x1c])
                v0 = v00 * (1-ax) + v10 * ax
                v1 = v01 * (1-ax) + v11 * ax
                vn = v0 * (1-ay) + v1 * ay
                bits.append(1 if vn >= c else 0)
            # Uniform pattern check (number of 0-1 transitions in circular sequence)
            transitions = 0
            for i in range(neighbors):
                if bits[i] != bits[(i+1) % neighbors]:
                    transitions += 1
            if transitions <= 2:
                bin_idx = sum(bits)
            else:
                bin_idx = neighbors + 1
            hist[bin_idx] += 1.0

    # Normalize
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist.astype(np.float32)


def _iter_sample_frames(asset_path: Path, max_samples: int = 16) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    cap = cv2.VideoCapture(str(asset_path))
    if cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) > 1:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            # sequential fallback
            for _ in range(max_samples):
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
        else:
            step = max(1, total // max_samples)
            for idx in range(0, total, step):
                if len(frames) >= max_samples:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if ok:
                    frames.append(frame)
        cap.release()
        return frames
    # Not a video (or single-frame video) → try reading as image
    cap.release()
    img = cv2.imread(str(asset_path), cv2.IMREAD_COLOR)
    if img is not None:
        frames.append(img)
    return frames


def _load_classifier(model_path: Path = DEFAULT_MODEL_PATH):
    try:
        clf = joblib.load(model_path)
        return clf
    except Exception as e:
        raise RuntimeError(f"Failed to load LBP model at {model_path}: {e}")


def predict_lbp_single(asset_path: Path, model_path: Path | None = None) -> Dict[str, object]:
    clf = _load_classifier(model_path or DEFAULT_MODEL_PATH)
    frames = _iter_sample_frames(asset_path, max_samples=16)
    if not frames:
        return {"score": 0.5, "label": "real", "method": "lbp_rf", "lbp_frames": 0, "lbp_dim": 0}

    vecs: List[np.ndarray] = []
    for bgr in frames:
        gray = _to_gray_uint8(bgr)
        hist = _lbp_hist(gray)
        vecs.append(hist)
    X = np.stack(vecs, axis=0)  # (N, D)
    v = X.mean(axis=0, keepdims=True)  # (1, D)

    # Expect classifier with predict_proba; y=1 corresponds to REAL (per training convention)
    try:
        proba = clf.predict_proba(v)[0]  # [P(y=0), P(y=1)]
        p_real = float(proba[1]) if len(proba) > 1 else 1.0 - float(proba[0])
    except Exception:
        # Fallback to decision_function or predict if proba not available
        try:
            decision = float(clf.decision_function(v))
            # Map decision to [0,1] via sigmoid as a heuristic
            p_real = 1.0 / (1.0 + np.exp(-decision))
        except Exception:
            pred = int(clf.predict(v)[0])  # 1=REAL, 0=FAKE
            p_real = 1.0 if pred == 1 else 0.0

    p_fake = float(max(0.0, min(1.0, 1.0 - p_real)))
    label = "fake" if p_fake >= 0.5 else "real"
    return {
        "score": p_fake,
        "label": label,
        "method": "lbp_rf",
        "lbp_frames": int(len(frames)),
        "lbp_dim": int(v.shape[1] if v.ndim == 2 else 0),
    }
