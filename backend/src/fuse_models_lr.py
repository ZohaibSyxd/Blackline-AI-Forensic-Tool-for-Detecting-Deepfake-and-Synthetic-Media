#!/usr/bin/env python3
"""
Stacking fusion for Xception + TimeSformer using Logistic Regression.

Inputs:
  --xception    backend/data/derived/xception_scores_frames.jsonl   (per-frame)
  --timesformer backend/data/derived/timesformer_scores.jsonl       (per-clip)
  --manifest    backend/data/derived/manifest.csv                   (sha256,label_num,split,...)

It aggregates each model to one score per sha256 (configurable), then fits a
logistic regression on TRAIN to predict P(FAKE). Evaluation is on TEST (or VAL
if you pass --val-as-test), and predictions for all assets are written to JSONL.

Usage:
  python backend/src/fuse_models_lr.py \
    --xception backend/data/derived/xception_scores_frames.jsonl \
    --timesformer backend/data/derived/timesformer_scores.jsonl \
    --manifest backend/data/derived/manifest.csv \
    --agg-frame p95 --agg-clip median \
    --C 1.0 --class-weight balanced --calibrate sigmoid \
    --use-split --val-as-test \
    --out backend/data/derived/fusion_predictions_lr.jsonl
"""
#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
from .audit import audit_step

SCORE_KEYS=["deepfake_score","prob_fake","fake_prob","score"]

def _get_score(d):
    for k in SCORE_KEYS:
        if k in d:
            try: return float(d[k])
            except: pass
    return None

def read_jsonl_scores(p: Path):
    by_sha={}
    with p.open(encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try: r=json.loads(line)
            except: continue
            sha=r.get("sha256"); s=_get_score(r)
            if not sha or s is None or np.isnan(s): continue
            by_sha.setdefault(sha,[]).append(float(s))
    return by_sha

def agg(vals, how):
    a=np.asarray(vals,float)
    if a.size==0: return np.nan
    how=how.lower()
    if how=="median": return float(np.median(a))
    if how=="max":    return float(np.max(a))
    if how=="p95":    return float(np.percentile(a,95))
    if how=="p90":    return float(np.percentile(a,90))
    return float(np.mean(a))

def agg_map(m, how): return {k: agg(v, how) for k,v in m.items()}

def read_manifest(p: Path):
    lab, split, aid = {}, {}, {}
    with p.open(newline="",encoding="utf-8") as f:
        r=csv.DictReader(f)
        for row in r:
            sha=(row.get("sha256") or "").strip()
            if not sha: continue
            lab[sha]=(row.get("label_num") or "").strip()
            split[sha]=(row.get("split") or "").strip().lower()
            aid[sha]=(row.get("asset_id") or "").strip()
    return lab, split, aid

def main():
    ap=argparse.ArgumentParser("Stack Xception+TimeSformer via Logistic Regression")
    ap.add_argument("--xception", required=True)
    ap.add_argument("--timesformer", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--agg-frame", default="p95", choices=["mean","max","median","p90","p95"])
    ap.add_argument("--agg-clip",  default="median", choices=["mean","max","median","p90","p95"])
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--class-weight", default="balanced", choices=["balanced","none"])
    ap.add_argument("--calibrate", default="sigmoid", choices=["none","sigmoid","isotonic"])
    ap.add_argument("--use-split", action="store_true")
    ap.add_argument("--val-as-test", action="store_true")
    ap.add_argument("--out", required=True)
    args=ap.parse_args()

    x=agg_map(read_jsonl_scores(Path(args.xception)), args.agg_frame)
    t=agg_map(read_jsonl_scores(Path(args.timesformer)), args.agg_clip)

    shas=sorted(set(x)|set(t))
    if not shas: print("No assets."); return
    X=np.array([[x.get(s,np.nan), t.get(s,np.nan)] for s in shas], float)

    # impute: if one is NaN, use the other; else mean 0.5
    for j in (0,1):
        col=X[:,j]; m=np.isnan(col)
        if m.any():
            other=X[:,1-j]
            repl=np.where(~np.isnan(other), other, 0.5)
            col[m]=repl[m]; X[:,j]=col

    lab, split, aid = read_manifest(Path(args.manifest))
    y=np.array([int(lab.get(s,"-1")) for s in shas], int)
    mask=(y>=0)
    if not mask.any():
        print("No labels in manifest."); return
    # convention: 0=FAKE (positive), 1=REAL
    y_pos=(y==0).astype(int)

    # train/test split via manifest
    te_mask=np.array([split.get(s,"")=="test" for s in shas], bool)
    if not te_mask.any() and args.val_as_test:
        te_mask=np.array([split.get(s,"")=="val" for s in shas], bool)
    tr_mask=np.array([split.get(s,"")=="train" for s in shas], bool)
    if not tr_mask.any():
        tr_mask=mask; te_mask=np.zeros_like(tr_mask)

    cw=None if args.class_weight=="none" else "balanced"
    base=LogisticRegression(C=args.C, class_weight=cw, solver="liblinear", max_iter=200)
    clf=base if args.calibrate=="none" else CalibratedClassifierCV(base, cv=5, method=args.calibrate)

    clf.fit(X[tr_mask&mask], y_pos[tr_mask&mask])
    print("Fitting logistic regression...")
    if (te_mask&mask).any():
        p=clf.predict_proba(X[te_mask&mask])[:,1]
        from math import isnan
        auc = roc_auc_score(y_pos[te_mask&mask], p) if len(np.unique(y_pos[te_mask&mask]))>1 else float("nan")
        acc = accuracy_score(y_pos[te_mask&mask], (p>=0.5).astype(int))
        print(f"[stack] TEST: AUC={auc:.3f} ACC={acc:.3f}")

    probs=clf.predict_proba(X)[:,1]  # P(FAKE)
    out=Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with audit_step("fuse_models_lr", params=vars(args), inputs={"xception": args.xception, "timesformer": args.timesformer, "manifest": args.manifest}) as outputs:
      with out.open("w",encoding="utf-8") as w:
        for s,(sx,st),pf in zip(shas,X,probs):
            w.write(json.dumps({
                "asset_id": (aid.get(s) or None),
                "sha256": s,
                "xception_agg_score": None if np.isnan(sx) else float(sx),
                "timesformer_agg_score": None if np.isnan(st) else float(st),
                "fused_score": float(pf),                # P(FAKE)
                "predicted_label": "FAKE" if pf>=0.5 else "REAL",
                "decision_threshold": 0.5,
                "model_name": "stack: logistic(xception,timesformer)",
                "model_version": "v1",
            })+"\n")
      outputs["fusion_predictions_lr"] = {"path": args.out}
    print(f"Wrote → {out}")

if __name__=="__main__":
    main()
