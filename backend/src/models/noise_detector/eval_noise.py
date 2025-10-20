"""
Evaluate noise predictions against ground truth labels in manifest.csv.

Inputs:
  - predictions.jsonl: rows with {group_key (sha256), pred_label (1=REAL), pred_prob_fake}
  - manifest.csv: provides label_num (1=REAL, 0=FAKE) for each sha256

Outputs:
  - noise_eval.json: summary metrics (accuracy, precision for FAKE, recall, f1, confusion matrix)
  - Prints concise summary to stdout.

Thresholding:
  - By default, uses pred_prob_fake >= 0.5 -> predict FAKE, else REAL.
  - Can adjust via --threshold.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_manifest_labels(path: Path) -> Dict[str, int]:
    out: Dict[str, int] = {}
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            sha = row.get("sha256")
            lab = row.get("label_num")
            if not sha or lab is None or lab == "":
                continue
            try:
                out[sha] = int(lab)
            except Exception:
                continue
    return out


def load_predictions(path: Path) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            sha = row.get("group_key")
            if not sha:
                continue
            out[sha] = {
                "pred_label": int(row.get("pred_label", 0)),   # 1=REAL
                "pred_prob_fake": float(row.get("pred_prob_fake", 0.0)),
            }
    return out


def compute_metrics(y_true: np.ndarray, y_score_fake: np.ndarray, threshold: float) -> Tuple[Dict[str, float], List[List[int]]]:
    # Map to binary preds: 1=FAKE, 0=REAL for conventional confusion-matrix order
    y_pred_fake = (y_score_fake >= threshold).astype(int)
    # But y_true uses 1=REAL, 0=FAKE -> convert to 1=FAKE for consistency
    y_true_fake = (1 - y_true).astype(int)

    tp = int(np.sum((y_true_fake == 1) & (y_pred_fake == 1)))
    tn = int(np.sum((y_true_fake == 0) & (y_pred_fake == 0)))
    fp = int(np.sum((y_true_fake == 0) & (y_pred_fake == 1)))
    fn = int(np.sum((y_true_fake == 1) & (y_pred_fake == 0)))

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)  # precision for FAKE class
    recall = tp / max(tp + fn, 1)     # recall for FAKE class
    f1 = (2 * precision * recall) / max(precision + recall, 1e-9)

    cm = [[tn, fp], [fn, tp]]  # rows: true REAL/FAKE (0/1), cols: pred REAL/FAKE (0/1)

    return {
        "accuracy": accuracy,
        "precision_fake": precision,
        "recall_fake": recall,
        "f1_fake": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }, cm


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate noise predictions against manifest labels")
    ap.add_argument("--pred", default="backend/data/derived/predictions.jsonl", help="Path to predictions.jsonl")
    ap.add_argument("--manifest", default="backend/data/derived/manifest.csv", help="Path to manifest.csv")
    ap.add_argument("--out", default="backend/data/derived/noise_eval.json", help="Path to write eval summary JSON")
    ap.add_argument("--threshold", type=float, default=0.5, help="Threshold on pred_prob_fake for FAKE")
    args = ap.parse_args()

    pred_path = Path(args.pred)
    manifest_path = Path(args.manifest)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = load_manifest_labels(manifest_path)
    preds = load_predictions(pred_path)

    y_true: List[int] = []
    y_score_fake: List[float] = []
    matched = 0
    for sha, lab in labels.items():
        if sha in preds:
            matched += 1
            y_true.append(lab)  # 1=REAL, 0=FAKE
            y_score_fake.append(preds[sha]["pred_prob_fake"])  # probability of FAKE

    if matched == 0:
        print("No overlap between predictions and manifest labels.")
        return

    y_true_arr = np.array(y_true, dtype=np.int64)
    y_score_arr = np.array(y_score_fake, dtype=np.float32)
    metrics, cm = compute_metrics(y_true_arr, y_score_arr, threshold=args.threshold)

    summary = {
        "n_total_labels": int(len(labels)),
        "n_with_predictions": int(matched),
        "threshold": args.threshold,
        "metrics": metrics,
        "confusion_matrix": cm,
    }

    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Eval (threshold={args.threshold}): accuracy={metrics['accuracy']:.3f} precision_fake={metrics['precision_fake']:.3f} recall_fake={metrics['recall_fake']:.3f} f1_fake={metrics['f1_fake']:.3f}")
    print(f"Wrote evaluation summary -> {out_path}")


if __name__ == "__main__":
    main()
