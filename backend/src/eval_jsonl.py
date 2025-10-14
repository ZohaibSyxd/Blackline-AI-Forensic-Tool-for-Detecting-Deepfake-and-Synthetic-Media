#!/usr/bin/env python3
"""
Evaluate deepfake predictions (JSONL) against a manifest with labels.

Works with your single-model or fused outputs:
- Looks for a score in this order: fused_score, deepfake_score, prob_fake, fake_prob, score
- Assumes score = P(FAKE). Manifest labels: label_num 0=FAKE, 1=REAL.

Examples:
  # Use test split, use threshold from file if present (else 0.5)
  python backend/src/eval/eval_jsonl.py \
    --preds backend/data/derived/fusion_predictions.jsonl \
    --manifest backend/data/derived/manifest.csv \
    --split test --use-file-threshold

  # Learn threshold on TRAIN, evaluate on TEST, also save ROC/PR plots
  python backend/src/eval/eval_jsonl.py \
    --preds backend/data/derived/fusion_predictions.jsonl \
    --manifest backend/data/derived/manifest.csv \
    --learn-threshold train --split test \
    --plots-out-dir backend/data/derived/plots \
    --out-metrics backend/data/derived/metrics_fusion.json
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Optional plotting / sklearn (install if you want curves)
try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False

try:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, precision_recall_fscore_support
    )
    HAS_SK = True
except Exception:
    HAS_SK = False

SCORE_KEYS = ["fused_score","deepfake_score","prob_fake","fake_prob","score"]

def read_manifest(path: Path):
    lab, split = {}, {}
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            sha = (row.get("sha256") or "").strip()
            if not sha: continue
            lab[sha] = row.get("label_num", "").strip()
            split[sha] = (row.get("split") or "").strip().lower()
    return lab, split

def read_preds(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                d=json.loads(line)
            except Exception:
                continue
            sha=d.get("sha256")
            if not sha: continue
            # pick a score
            score=None
            for k in SCORE_KEYS:
                if k in d and d[k] is not None:
                    try:
                        score=float(d[k])
                        break
                    except Exception:
                        pass
            if score is None or np.isnan(score): 
                continue
            thr = d.get("decision_threshold", None)
            rows.append((sha, score, thr))
    return rows

def choose_threshold(xs: np.ndarray, y_pos: np.ndarray) -> float:
    # y_pos: 1=FAKE positive, 0=REAL
    m = ~np.isnan(xs)
    xs = xs[m]; y_pos = y_pos[m]
    if xs.size == 0:
        return 0.5
    xs_sorted = np.unique(xs)
    # midpoints (and margins) search
    cands = [xs_sorted[0]-1e-6, *(((xs_sorted[:-1]+xs_sorted[1:])/2.0).tolist() if xs_sorted.size>1 else []), xs_sorted[-1]+1e-6]
    best_t, best_bal = 0.5, -1.0
    for t in cands:
        yhat = (xs >= t).astype(int)
        tp = int(np.sum((y_pos==1)&(yhat==1)))
        tn = int(np.sum((y_pos==0)&(yhat==0)))
        fp = int(np.sum((y_pos==0)&(yhat==1)))
        fn = int(np.sum((y_pos==1)&(yhat==0)))
        tpr = tp / max(1, tp+fn)  # recall (FAKE)
        tnr = tn / max(1, tn+fp)  # specificity (REAL)
        bal = 0.5*(tpr+tnr)
        if bal > best_bal:
            best_bal, best_t = bal, float(t)
    return best_t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="test", help="Which split to evaluate: test|val|train|all")
    ap.add_argument("--use-file-threshold", action="store_true", help="Use decision_threshold from preds file (median if multiple).")
    ap.add_argument("--threshold", type=float, default=None, help="Fixed threshold to apply (overrides others).")
    ap.add_argument("--learn-threshold", choices=["train","all"], default=None,
                    help="Learn threshold on TRAIN (or ALL labeled) to maximize balanced acc, then evaluate on --split.")
    ap.add_argument("--plots-out-dir", default="", help="If set and matplotlib available, writes ROC/PR plots.")
    ap.add_argument("--out-metrics", default="", help="If set, writes a JSON metrics summary.")
    args = ap.parse_args()

    preds = read_preds(Path(args.preds))
    lab, split = read_manifest(Path(args.manifest))

    # Join preds with labels
    shas, scores, labels, splits = [], [], [], []
    for sha, s, thr in preds:
        lab_s = lab.get(sha, "")
        if lab_s == "":  # unlabeled → skip for metrics
            continue
        shas.append(sha); scores.append(float(s))
        labels.append(int(lab_s))  # 0=FAKE, 1=REAL
        splits.append((split.get(sha, "") or ""))

    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    # Pos class (for metrics): FAKE
    y_pos = (labels == 0).astype(int)

    if scores.size == 0:
        print("No labeled predictions found. Check your manifest and preds.")
        return

    # Build masks
    sp = args.split.lower()
    if sp == "all":
        eval_mask = np.ones_like(y_pos, dtype=bool)
    else:
        eval_mask = np.array([s == sp for s in splits], bool)
        if not eval_mask.any():
            # fallback: evaluate on all
            print(f"[warn] No rows with split='{sp}'. Evaluating on all labeled.")
            eval_mask = np.ones_like(y_pos, dtype=bool)

    # Decide threshold
    if args.threshold is not None:
        thr = float(args.threshold)
        thr_src = "fixed"
    elif args.learn_threshold is not None:
        if args.learn_threshold == "train":
            tr_mask = np.array([s == "train" for s in splits], bool)
            if not tr_mask.any():
                print("[warn] No TRAIN rows; learning on all labeled instead.")
                tr_mask = np.ones_like(y_pos, dtype=bool)
            thr = choose_threshold(scores[tr_mask], y_pos[tr_mask])
            thr_src = "learned(train)"
        else:
            thr = choose_threshold(scores, y_pos)
            thr_src = "learned(all)"
    elif args.use_file_threshold:
        # take the median of any decision_thresholds present in file; fallback 0.5
        file_thrs = [r[2] for r in preds if r[2] is not None]
        thr = float(np.median(file_thrs)) if len(file_thrs) else 0.5
        thr_src = "file(median)" if len(file_thrs) else "default(0.5)"
    else:
        thr = 0.5
        thr_src = "default(0.5)"

    # Metrics at threshold
    m = eval_mask
    yhat = (scores[m] >= thr).astype(int)  # 1=FAKE predicted
    yy = y_pos[m]

    tp = int(np.sum((yy==1)&(yhat==1)))
    tn = int(np.sum((yy==0)&(yhat==0)))
    fp = int(np.sum((yy==0)&(yhat==1)))
    fn = int(np.sum((yy==1)&(yhat==0)))

    acc = (tp+tn)/max(1,tp+tn+fp+fn)
    recall = tp/max(1,tp+fn)         # FAKE recall (TPR)
    specificity = tn/max(1,tn+fp)    # REAL recall (TNR)
    bal_acc = 0.5*(recall+specificity)
    precision = tp/max(1,tp+fp)
    f1 = 0.0 if (precision+recall)==0 else 2*precision*recall/(precision+recall)

    # Curve metrics
    if HAS_SK and len(np.unique(yy))>1:
        roc_auc = roc_auc_score(yy, scores[m])
        pr_auc  = average_precision_score(yy, scores[m])  # AP = PR-AUC for positive (FAKE)
    else:
        roc_auc = float("nan"); pr_auc = float("nan")

    # Print
    n_eval = int(m.sum())
    print(f"Evaluated on split='{args.split}'  (n={n_eval})")
    print(f"Threshold = {thr:.4f}  [{thr_src}]")
    print(f"ACC={acc:.3f}  BAL_ACC={bal_acc:.3f}  PREC(FAKE)={precision:.3f}  REC(FAKE)={recall:.3f}  SPEC(REAL)={specificity:.3f}  F1(FAKE)={f1:.3f}")
    print(f"ROC_AUC={roc_auc:.3f}  PR_AUC(FAKE)={pr_auc:.3f}")
    print(f"Confusion matrix (FAKE=positive):  TP={tp}  FP={fp}  TN={tn}  FN={fn}")

    # Optional: plots
    if args.plots_out_dir and HAS_PLT and HAS_SK and len(np.unique(yy))>1:
        out_dir = Path(args.plots_out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        # ROC
        from sklearn.metrics import roc_curve, precision_recall_curve
        fpr, tpr, _ = roc_curve(yy, scores[m])
        plt.figure()
        plt.plot(fpr, tpr, lw=2)
        plt.plot([0,1],[0,1],'--')
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC (AUC={roc_auc:.3f})")
        plt.tight_layout(); plt.savefig(out_dir/"roc.png"); plt.close()
        # PR
        prec, rec, _ = precision_recall_curve(yy, scores[m])
        plt.figure()
        plt.plot(rec, prec, lw=2)
        plt.xlabel("Recall (FAKE)"); plt.ylabel("Precision (FAKE)"); plt.title(f"PR (AP={pr_auc:.3f})")
        plt.tight_layout(); plt.savefig(out_dir/"pr.png"); plt.close()
        print(f"Wrote ROC/PR plots → {out_dir}")

    # Optional: dump metrics JSON
    if args.out_metrics:
        out = {
            "split": args.split,
            "n_eval": n_eval,
            "threshold": thr, "threshold_source": thr_src,
            "accuracy": acc, "balanced_accuracy": bal_acc,
            "precision_fake": precision, "recall_fake": recall,
            "specificity_real": specificity, "f1_fake": f1,
            "roc_auc": roc_auc, "pr_auc_fake": pr_auc,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }
        Path(args.out_metrics).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_metrics, "w", encoding="utf-8") as w:
            json.dump(out, w, indent=2)
        print(f"Wrote metrics JSON → {args.out_metrics}")

if __name__ == "__main__":
    main()
