"""
Train a simple noise-based classifier from per-frame noise metrics.

Inputs:
  - frames_noise.jsonl: per-frame metrics with keys:
      asset_id, sha256, shot_index, frame_index, residual_abs_mean,
      residual_std, residual_energy, fft_low_ratio, fft_high_ratio
  - manifest.csv: per-asset rows with sha256 and label_num (1=REAL, 0=FAKE)

Pipeline:
  1) Aggregate per-frame metrics to per-asset features (by sha256):
     For each metric compute: mean, std, min, max, median, and frame_count.
  2) Join with labels from manifest.csv.
  3) Fit a LogisticRegression (class_weight=balanced).
  4) Report CV metrics and save final model trained on all data.

Outputs:
  - noise_model.joblib (default under backend/data/derived)
  - noise_model_report.json (classification report, confusion matrix, roc_auc)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold


METRIC_KEYS = [
    "residual_abs_mean",
    "residual_std",
    "residual_energy",
    "fft_low_ratio",
    "fft_high_ratio",
]


def read_manifest_labels(manifest_path: Path) -> Dict[str, int]:
    labels: Dict[str, int] = {}
    with open(manifest_path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            sha = row.get("sha256")
            lab = row.get("label_num")
            if not sha:
                continue
            if lab is None or lab == "":
                continue
            try:
                labels[sha] = int(lab)
            except Exception:
                continue
    return labels


def aggregate_frames_noise(frames_noise_path: Path) -> Tuple[List[str], np.ndarray, np.ndarray]:
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

    shas: List[str] = []
    features: List[List[float]] = []
    frame_counts: List[int] = []

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

    for sha, metrics in by_sha.items():
        feat_row: List[float] = []
        for k in METRIC_KEYS:
            feat_row.extend(summarize(metrics.get(k, [])))
        # append frame count as a feature
        feat_row.append(float(counts.get(sha, 0)))
        shas.append(sha)
        features.append(feat_row)
        frame_counts.append(counts.get(sha, 0))

    if not features:
        return [], np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    X = np.array(features, dtype=np.float32)
    return shas, X, np.array(frame_counts, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train noise classifier from frames_noise.jsonl and manifest.csv labels")
    ap.add_argument("--frames-noise", default="backend/data/derived/frames_noise.jsonl", help="Path to frames_noise.jsonl")
    ap.add_argument("--manifest", default="backend/data/derived/manifest.csv", help="Path to manifest.csv with labels")
    ap.add_argument("--out-model", default="backend/data/derived/noise_model.joblib", help="Output model path (joblib)")
    ap.add_argument("--report", default="backend/data/derived/noise_model_report.json", help="Output JSON report path")
    ap.add_argument("--cv-folds", type=int, default=5, help="Stratified K-Folds for CV (reduced if not enough samples)")
    args = ap.parse_args()

    frames_noise_path = Path(args.frames_noise)
    manifest_path = Path(args.manifest)
    out_model = Path(args.out_model)
    out_report = Path(args.report)
    out_model.parent.mkdir(parents=True, exist_ok=True)

    # Aggregate features by asset (sha256)
    shas, X, frame_counts = aggregate_frames_noise(frames_noise_path)
    if X.shape[0] == 0:
        print("No aggregated features found. Did you run compute_noise.py?")
        return

    # Load labels
    label_map = read_manifest_labels(manifest_path)
    y_list: List[int] = []
    keep_idx: List[int] = []
    for i, sha in enumerate(shas):
        if sha in label_map:
            y_list.append(label_map[sha])
            keep_idx.append(i)
    if not keep_idx:
        print("No labels matched between frames_noise and manifest.csv")
        return

    X = X[keep_idx]
    y = np.array(y_list, dtype=np.int64)
    shas = [shas[i] for i in keep_idx]

    # Ensure at least 2 classes exist
    classes = np.unique(y)
    if classes.size < 2:
        print("Only one class present in labels; cannot train a classifier.")
        return

    # Decide whether to do CV or train-only evaluation
    class_counts = [int(np.sum(y == c)) for c in classes]
    min_class = int(np.min(class_counts))
    do_cv = (min_class >= 2) and (X.shape[0] >= 4)

    y_true_all: List[int] = []
    y_prob_all: List[float] = []
    y_pred_all: List[int] = []

    if do_cv:
        folds = min(args.cv_folds, min_class)
        folds = max(folds, 2)
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        for train_idx, test_idx in skf.split(X, y):
            clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
            clf.fit(X[train_idx], y[train_idx])
            proba = clf.predict_proba(X[test_idx])
            classes_ = list(clf.classes_)
            idx_real = classes_.index(1) if 1 in classes_ else 0
            prob_real = proba[:, idx_real]
            prob_fake = 1.0 - prob_real
            pred = (prob_fake >= 0.5).astype(int)  # 1=FAKE, 0=REAL
            y_pred = (1 - pred).astype(int)       # back to 1=REAL

            y_true_all.extend(y[test_idx].tolist())
            y_prob_all.extend(prob_real.tolist())
            y_pred_all.extend(y_pred.tolist())
        eval_mode = "cv"
    else:
        # Train-only evaluation (small dataset): fit and evaluate on training set
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        classes_ = list(clf.classes_)
        idx_real = classes_.index(1) if 1 in classes_ else 0
        prob_real = proba[:, idx_real]
        prob_fake = 1.0 - prob_real
        pred = (prob_fake >= 0.5).astype(int)
        y_pred = (1 - pred).astype(int)

        y_true_all = y.tolist()
        y_prob_all = prob_real.tolist()
        y_pred_all = y_pred.tolist()
        eval_mode = "train"

    # Build report
    try:
        report = classification_report(y_true_all, y_pred_all, output_dict=True, zero_division=0)
    except Exception:
        report = {"error": "classification_report_failed"}

    try:
        roc_auc = float(roc_auc_score(y_true_all, y_prob_all))
    except Exception:
        roc_auc = None

    try:
        cm = confusion_matrix(y_true_all, y_pred_all).tolist()
    except Exception:
        cm = None

    summary = {
        "classes": ["REAL", "FAKE"],
        "encoding": {"REAL": 1, "FAKE": 0},
        "n_assets": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "evaluation": eval_mode,
        "classification_report": report,
        "confusion_matrix": cm,
        "roc_auc": roc_auc,
    }

    out_report.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Fit final model on all data and save
    final_clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    final_clf.fit(X, y)
    joblib.dump({
        "model": final_clf,
        "feature_version": 1,
        "metrics": METRIC_KEYS,
        "agg": ["mean","std","min","max","median","count"],
    }, out_model)
    print(f"Saved model to {out_model}")
    print(f"Wrote report to {out_report}")


if __name__ == "__main__":
    main()
