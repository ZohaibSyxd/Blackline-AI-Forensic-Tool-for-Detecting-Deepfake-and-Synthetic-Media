"""
Predict FAKE/REAL per asset using a trained noise model and frames_noise.jsonl.

Inputs:
  - frames_noise.jsonl: per-frame noise metrics (same as train)
  - noise_model.joblib: saved model from train_noise.py

Outputs:
  - predictions.jsonl (default): lines with {group_key (sha256), pred_label (1=REAL), pred_prob_fake}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
from ...audit import audit_step
import numpy as np


METRIC_KEYS = [
    "residual_abs_mean",
    "residual_std",
    "residual_energy",
    "fft_low_ratio",
    "fft_high_ratio",
]


def aggregate_frames_noise(frames_noise_path: Path) -> Dict[str, List[float]]:
    by_sha: Dict[str, Dict[str, List[float]]] = {}
    counts: Dict[str, int] = {}
    with open(frames_noise_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            sha = row.get("sha256")
            if not sha:
                continue
            d = by_sha.setdefault(sha, {k: [] for k in METRIC_KEYS})
            for k in METRIC_KEYS:
                v = row.get(k)
                if v is None:
                    continue
                try:
                    d[k].append(float(v))
                except Exception:
                    continue
            counts[sha] = counts.get(sha, 0) + 1

    def summarize(vals: List[float]) -> List[float]:
        if not vals:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        arr = np.array(vals, dtype=np.float32)
        return [
            float(np.mean(arr)),
            float(np.std(arr)),
            float(np.min(arr)),
            float(np.max(arr)),
            float(np.median(arr)),
        ]

    feats_by_sha: Dict[str, List[float]] = {}
    for sha, metrics in by_sha.items():
        feat_row: List[float] = []
        for k in METRIC_KEYS:
            feat_row.extend(summarize(metrics.get(k, [])))
        feat_row.append(float(counts.get(sha, 0)))  # frame count
        feats_by_sha[sha] = feat_row
    return feats_by_sha


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict with noise model from frames_noise.jsonl")
    ap.add_argument("--frames-noise", default="backend/data/derived/frames_noise.jsonl", help="Path to frames_noise.jsonl")
    ap.add_argument("--model", default="backend/data/derived/noise_model.joblib", help="Path to trained model (joblib)")
    ap.add_argument("--out", default="backend/data/derived/predictions.jsonl", help="Output predictions JSONL")
    args = ap.parse_args()

    frames_noise_path = Path(args.frames_noise)
    model_path = Path(args.model)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    feats_by_sha = aggregate_frames_noise(frames_noise_path)
    if not feats_by_sha:
        print("No aggregated features found; did you run compute_noise.py?")
        return

    bundle = joblib.load(model_path)
    clf = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle
    classes_ = list(clf.classes_)
    idx_real = classes_.index(1) if 1 in classes_ else 0

    shas = list(feats_by_sha.keys())
    X = np.array([feats_by_sha[s] for s in shas], dtype=np.float32)
    proba = clf.predict_proba(X)
    prob_real = proba[:, idx_real]
    prob_fake = 1.0 - prob_real
    pred_real = (prob_real >= 0.5).astype(int)  # 1=REAL else 0

    with audit_step("predict_noise", params=vars(args), inputs={"frames_noise": args.frames_noise, "model": args.model}) as outputs:
      with open(out_path, "w", encoding="utf-8") as w:
        for sha, yhat, pf in zip(shas, pred_real, prob_fake):
            row = {
                "group_key": sha,
                "pred_label": int(yhat),  # 1=REAL
                "pred_prob_fake": float(pf),
            }
            w.write(json.dumps(row) + "\n")
      outputs["noise_predictions"] = {"path": args.out}
    print(f"Wrote {len(shas)} predictions -> {out_path}")


if __name__ == "__main__":
    main()
