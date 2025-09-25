"""
Evaluate ELA metrics for deepfake detection using simple per-asset aggregation
and a threshold classifier.

Inputs:
  - frames_ela.jsonl (from compute_ela.py)
  - manifest.csv (from build_manifest.py) containing columns:
        sha256, label, label_num (REAL=1, FAKE=0), split (train/test/val)

Outputs:
  - Prints metrics to stdout and writes a JSON file with results/params.

Usage (PowerShell):
  # 1) Ensure manifest.csv exists (with labels)
  # python .\backend\src\build_manifest.py `
  #   --audit .\backend\data\audit\ingest_log.jsonl `
  #   --out   .\backend\data\derived\manifest.csv `
  #   --meta  .\backend\datasets\train\metadata.json

  # 2) Run ELA if not already done
  # python .\backend\src\models\compute_ela.py `
  #   --frames .\backend\data\derived\frames_normalized.jsonl `
  #   --root   .\backend\data\derived\normalized `
  #   --out    .\backend\data\derived\frames_ela.jsonl

  # 3) Evaluate
  python .\backend\src\models\eval_ela.py `
    --ela       .\backend\data\derived\frames_ela.jsonl `
    --manifest  .\backend\data\derived\manifest.csv `
    --agg       median `
    --metric    ela_error_mean `
    --out       .\backend\data\derived\ela_eval.json

    # 3b) Stratified K-fold CV (no sklearn required)
    # Runs k stratified folds on the labeled subset and reports per-fold and mean/std metrics
    python .\backend\src\models\eval_ela.py `
        --ela       .\backend\data\derived\frames_ela.jsonl `
        --manifest  .\backend\data\derived\manifest.csv `
        --agg       p95 `
        --metric    ela_error_mean `
        --cv-k      5 `
        --seed      42 `
        --out       .\backend\data\derived\ela_eval_cv.json

Notes:
  - This uses only numpy + stdlib. No sklearn dependency required.
  - If split is missing in manifest, a random train/test split is generated.
  - Positive class is FAKE by default (label_num 0). You can flip with --positive REAL.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def read_manifest(path: Path) -> Dict[str, dict]:
    """Return mapping sha256 -> manifest row with label_num, label, split."""
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    out: Dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            sha = row.get("sha256")
            if not sha:
                continue
            # label_num may be '', '0', '1'
            lab_num_raw = row.get("label_num")
            lab_num = None
            if lab_num_raw is not None and str(lab_num_raw).strip() != "":
                try:
                    lab_num = int(lab_num_raw)
                except Exception:
                    lab_num = None
            out[sha] = {
                "label": (row.get("label") or "").strip().upper(),
                "label_num": lab_num,
                "split": (row.get("split") or "").strip().lower(),
                "asset_id": row.get("asset_id") or "",
            }
    return out


def read_ela_jsonl(path: Path) -> Dict[str, List[dict]]:
    """Group ELA frame rows by sha256."""
    if not path.exists():
        raise FileNotFoundError(f"ELA jsonl not found: {path}")
    groups: Dict[str, List[dict]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            sha = rec.get("sha256")
            if not sha:
                continue
            groups[sha].append(rec)
    return groups


def aggregate_feature(values: List[float], agg: str) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    agg = agg.lower()
    if agg == "mean":
        return float(np.mean(arr))
    if agg == "max":
        return float(np.max(arr))
    if agg == "median":
        return float(np.median(arr))
    if agg in {"p95", "q95"}:
        return float(np.percentile(arr, 95))
    if agg in {"p90", "q90"}:
        return float(np.percentile(arr, 90))
    if agg in {"p99", "q99"}:
        return float(np.percentile(arr, 99))
    # default
    return float(np.mean(arr))


def train_threshold(xs: np.ndarray, ys: np.ndarray, pos_label: int) -> Tuple[float, Dict[str, float]]:
    """Choose threshold t that maximizes accuracy on training data.

    Predict pos if x >= t when positive is FAKE (pos_label=0) and higher means more fake-like.
    For REAL positive, we invert the direction.
    """
    assert xs.shape == ys.shape
    mask = ~np.isnan(xs)
    xs = xs[mask]
    ys = ys[mask].astype(int)
    if xs.size == 0:
        return float("nan"), {"acc": float("nan")}
    uniq = np.unique(xs)
    # Test midpoints between sorted unique values (and edges)
    uniq.sort()
    cands = []
    if uniq.size == 1:
        cands = [uniq[0]]
    else:
        mids = (uniq[:-1] + uniq[1:]) / 2.0
        cands = [uniq[0] - 1e-6, *mids.tolist(), uniq[-1] + 1e-6]

    best_t = None
    best_acc = -1.0
    for t in cands:
        if pos_label == 0:  # FAKE positive; higher -> fake
            yhat = (xs >= t).astype(int)
        else:  # REAL positive; higher -> real
            yhat = (xs < t).astype(int)
        acc = (yhat == ys).mean()
        if acc > best_acc:
            best_acc = float(acc)
            best_t = float(t)
    return best_t if best_t is not None else float("nan"), {"acc": best_acc}


def _metrics_binary(y_true: np.ndarray, scores: np.ndarray, y_pred: np.ndarray, pos_label: int) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    # Confusion matrix components for pos_label
    tp = int(np.sum((y_true == pos_label) & (y_pred == pos_label)))
    tn = int(np.sum((y_true != pos_label) & (y_pred != pos_label)))
    fp = int(np.sum((y_true != pos_label) & (y_pred == pos_label)))
    fn = int(np.sum((y_true == pos_label) & (y_pred != pos_label)))
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, (prec + rec)) if (prec + rec) > 0 else 0.0
    # Balanced accuracy
    tpr = rec
    tnr = tn / max(1, tn + fp)
    bacc = 0.5 * (tpr + tnr)

    # ROC-AUC via ranking (only if both classes present)
    try:
        y_pos = (y_true == pos_label).astype(int)
        if y_pos.sum() in (0, y_pos.size):
            auc = float("nan")
        else:
            order = np.argsort(scores)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(1, len(scores) + 1)
            # Use Mann–Whitney U: AUC = (sum ranks of positives - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
            n_pos = int(y_pos.sum())
            n_neg = int(y_pos.size - n_pos)
            sum_r_pos = float(ranks[y_pos == 1].sum())
            auc = (sum_r_pos - n_pos * (n_pos + 1) / 2.0) / max(1, n_pos * n_neg)
    except Exception:
        auc = float("nan")

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "balanced_accuracy": float(bacc),
        "roc_auc": float(auc),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def _stratified_kfold_indices(y: np.ndarray, k: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return list of (train_mask, test_mask) for stratified k-fold on labels y.

    If a class has fewer samples than k, we reduce k to the minimum class count.
    Requires at least 2 folds to proceed.
    """
    y = y.astype(int)
    classes, counts = np.unique(y, return_counts=True)
    if classes.size < 2:
        raise ValueError("Stratified K-fold requires at least two classes")
    max_k = int(min([counts.min(), k]))
    if max_k < 2:
        raise ValueError(f"Not enough samples per class for k={k} (min class count={counts.min()})")
    k = max_k

    rng = np.random.default_rng(seed)
    # Build per-class folds
    per_fold_indices: List[List[int]] = [ [] for _ in range(k) ]
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        parts = np.array_split(idx, k)
        for fi in range(k):
            per_fold_indices[fi].extend(parts[fi].tolist())

    masks: List[Tuple[np.ndarray, np.ndarray]] = []
    n = y.size
    for fi in range(k):
        test_idx = np.array(sorted(per_fold_indices[fi]), dtype=int)
        test_mask = np.zeros(n, dtype=bool)
        test_mask[test_idx] = True
        train_mask = ~test_mask
        masks.append((train_mask, test_mask))
    return masks


def _mean_std_metrics(fold_metrics: List[Dict[str, float]]) -> Tuple[Dict[str, float], Dict[str, float]]:
    keys = [k for k in fold_metrics[0].keys() if isinstance(fold_metrics[0][k], (int, float))]
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    for k in keys:
        vals = np.array([float(m[k]) for m in fold_metrics], dtype=float)
        means[k] = float(np.nanmean(vals))
        stds[k] = float(np.nanstd(vals))
    return means, stds


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate ELA metrics as a classifier.")
    ap.add_argument("--ela", default="backend/data/derived/frames_ela.jsonl", help="Path to frames_ela.jsonl")
    ap.add_argument("--manifest", default="backend/data/derived/manifest.csv", help="Path to manifest.csv (w/ labels)")
    ap.add_argument("--agg", choices=["mean", "max", "median", "p90", "p95", "p99"], default="p95", help="Per-asset aggregation")
    ap.add_argument("--metric", choices=["ela_error_mean", "ela_error_max"], default="ela_error_mean", help="ELA field to use")
    ap.add_argument("--positive", choices=["FAKE", "REAL"], default="FAKE", help="Which class is positive for metrics")
    ap.add_argument("--use-split", action="store_true", help="Use manifest split train/test/val; otherwise random split")
    ap.add_argument("--val-as-test", action="store_true", help="When using split, treat VAL as test if TEST empty")
    ap.add_argument("--test-size", type=float, default=0.2, help="Test fraction for random split (0-1)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv-k", type=int, default=0, help="If >=2, run stratified K-fold CV on labeled items (ignores --use-split)")
    ap.add_argument("--out", default="backend/data/derived/ela_eval.json", help="Path to save metrics JSON")
    args = ap.parse_args()

    ela_path = Path(args.ela)
    man_path = Path(args.manifest)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(man_path)
    groups = read_ela_jsonl(ela_path)

    # Build per-asset feature table
    xs = []
    ys = []
    sha_list = []
    for sha, frames in groups.items():
        meta = manifest.get(sha)
        if not meta:
            continue
        lab_num = meta.get("label_num")
        if lab_num is None:
            continue
        vals = []
        for fr in frames:
            v = fr.get(args.metric)
            if v is None:
                continue
            try:
                vals.append(float(v))
            except Exception:
                continue
        if not vals:
            continue
        feat = aggregate_feature(vals, args.agg)
        if np.isnan(feat):
            continue
        xs.append(float(feat))
        ys.append(int(lab_num))
        sha_list.append(sha)

    if not xs:
        print("No overlapping assets with labels and ELA metrics. Ensure manifest has labels and ELA exists.")
        return

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=int)

    pos_label = 0 if args.positive == "FAKE" else 1

    # Cross-validation branch
    if args.cv_k and args.cv_k >= 2:
        try:
            folds = _stratified_kfold_indices(ys, args.cv_k, args.seed)
        except ValueError as e:
            print(f"CV configuration error: {e}")
            return

        fold_results = []
        for fi, (tr, te) in enumerate(folds):
            t, _ = train_threshold(xs[tr], ys[tr], pos_label)
            if np.isnan(t):
                print(f"Fold {fi+1}: failed to find threshold (insufficient training data)")
                return
            scores = xs.copy()
            if pos_label == 1:
                scores = -scores
            if pos_label == 0:
                yhat_tr = (xs[tr] >= t).astype(int)
                yhat_te = (xs[te] >= t).astype(int)
            else:
                yhat_tr = (xs[tr] < t).astype(int)
                yhat_te = (xs[te] < t).astype(int)
            m_tr = _metrics_binary(ys[tr], scores[tr], yhat_tr, pos_label)
            m_te = _metrics_binary(ys[te], scores[te], yhat_te, pos_label)
            fold_results.append({
                "fold": fi + 1,
                "threshold": float(t),
                "train": m_tr,
                "test": m_te,
            })

        test_metrics = [fr["test"] for fr in fold_results]
        mean_m, std_m = _mean_std_metrics(test_metrics)
        result = {
            "params": {
                "metric": args.metric,
                "agg": args.agg,
                "positive": args.positive,
                "cv_k": int(len(folds)),
            },
            "cv": {
                "folds": fold_results,
                "test_mean": mean_m,
                "test_std": std_m,
            },
        }
        with open(out_path, "w", encoding="utf-8") as w:
            json.dump(result, w, indent=2)

        print("ELA evaluation (Stratified K-fold)")
        print(f"  metric={args.metric} agg={args.agg} positive={args.positive} k={len(folds)}")
        print("  Test mean: acc={accuracy:.3f} f1={f1:.3f} auc={roc_auc:.3f} bal_acc={balanced_accuracy:.3f}".format(**mean_m))
        print("  Test  std: acc={accuracy:.3f} f1={f1:.3f} auc={roc_auc:.3f} bal_acc={balanced_accuracy:.3f}".format(**std_m))
        return

    # Split (single train/test)
    if args.use_split:
        splits = []
        has_test = False
        for sha in sha_list:
            spl = (manifest.get(sha, {}).get("split") or "").lower()
            splits.append(spl)
            if spl == "test":
                has_test = True
        splits = np.array(splits)
        if has_test:
            tr = (splits == "train")
            te = (splits == "test")
        elif args.val_as_test and np.any(splits == "val"):
            tr = (splits == "train")
            te = (splits == "val")
        else:
            # Fallback to random
            rng = np.random.default_rng(args.seed)
            idx = rng.permutation(xs.size)
            n_test = max(1, int(round(args.test_size * xs.size)))
            te_idx = idx[:n_test]
            tr_idx = idx[n_test:]
            tr = np.zeros(xs.size, dtype=bool); tr[tr_idx] = True
            te = np.zeros(xs.size, dtype=bool); te[te_idx] = True
    else:
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(xs.size)
        n_test = max(1, int(round(args.test_size * xs.size)))
        te_idx = idx[:n_test]
        tr_idx = idx[n_test:]
        tr = np.zeros(xs.size, dtype=bool); tr[tr_idx] = True
        te = np.zeros(xs.size, dtype=bool); te[te_idx] = True

    # Train threshold on training split
    t, train_info = train_threshold(xs[tr], ys[tr], pos_label)
    if np.isnan(t):
        print("Failed to find threshold (insufficient training data).")
        return

    # Scores for ROC use raw feature; if positive is REAL we flip the sign so higher -> positive consistently
    scores = xs.copy()
    if pos_label == 1:
        scores = -scores

    # Predict
    if pos_label == 0:
        yhat_tr = (xs[tr] >= t).astype(int)
        yhat_te = (xs[te] >= t).astype(int)
    else:
        yhat_tr = (xs[tr] < t).astype(int)
        yhat_te = (xs[te] < t).astype(int)

    # Metrics
    m_tr = _metrics_binary(ys[tr], scores[tr], yhat_tr, pos_label)
    m_te = _metrics_binary(ys[te], scores[te], yhat_te, pos_label)

    result = {
        "params": {
            "metric": args.metric,
            "agg": args.agg,
            "positive": args.positive,
            "threshold": t,
            "train_size": int(tr.sum()),
            "test_size": int(te.sum()),
        },
        "train": m_tr,
        "test": m_te,
    }

    with open(out_path, "w", encoding="utf-8") as w:
        json.dump(result, w, indent=2)

    # Pretty print summary
    print("ELA evaluation")
    print(f"  metric={args.metric} agg={args.agg} positive={args.positive}")
    print(f"  train n={tr.sum()} test n={te.sum()} threshold={t:.4f}")
    print("  Train: acc={accuracy:.3f} f1={f1:.3f} auc={roc_auc:.3f} tp={tp} fp={fp} tn={tn} fn={fn}".format(**m_tr))
    print("  Test : acc={accuracy:.3f} f1={f1:.3f} auc={roc_auc:.3f} tp={tp} fp={fp} tn={tn} fn={fn}".format(**m_te))


if __name__ == "__main__":
    main()
