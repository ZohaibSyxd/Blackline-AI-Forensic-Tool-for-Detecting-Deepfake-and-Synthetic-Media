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
from future import annotations
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

def predict_deepfake(video_path: Path, sha256: str | None = None) -> Dict[str, object]:
    """Return a pseudo deepfake score.

    Args:
        video_path: path to the ingested video file.
        sha256: optional known content hash; if absent we derive from file bytes (slow for large files).
    """
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