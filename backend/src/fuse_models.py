#!/usr/bin/env python3
"""
Fuse Xception (per-frame) and TimeSformer (per-clip) scores into one per-asset score.

Inputs (JSONL):
  --xception   backend/data/derived/xception_scores.jsonl
               Rows can be per-frame OR per-asset. Expected keys tried in order:
               ["deepfake_score","prob_fake","fake_prob","score"] plus sha256.
  --timesformer backend/data/derived/timesformer_scores.jsonl
               Per-clip rows with the same score key options + sha256.

Optional labels (CSV):
  --manifest   backend/data/derived/manifest.csv
               Columns: sha256, label_num (REAL=1, FAKE=0), split (train/test/val) [optional]

What it does:
  1) Aggregates each model to one score per sha256 (median/p95).
  2) Either blends with a fixed --alpha (fused = a*x + (1-a)*t), or
     grid-searches alpha in [0..1] to maximize validation accuracy
     (threshold chosen on train split).
  3) Writes per-asset JSONL with scores and predicted label.

Usage:
  python backend/src/fuse_models.py \
    --xception backend/data/derived/xception_scores.jsonl \
    --timesformer backend/data/derived/timesformer_scores.jsonl \
    --manifest backend/data/derived/manifest.csv \
    --agg-frame p95 --agg-clip median \
    --learn-alpha \
    --use-split --val-as-test \
    --out backend/data/derived/fusion_predictions.jsonl
"""
#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path
import numpy as np
from tqdm import tqdm

SCORE_KEYS = ["deepfake_score","prob_fake","fake_prob","score"]

def _get_score(d):
    for k in SCORE_KEYS:
        if k in d:
            try: return float(d[k])
            except: pass
    return None

def read_jsonl_scores(p: Path):
    by_sha = {}
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

def read_manifest(mpath: Path):
    lab, split, aid = {}, {}, {}
    if not mpath.exists(): return lab, split, aid
    with mpath.open(newline="",encoding="utf-8") as f:
        r=csv.DictReader(f)
        for row in r:
            sha=(row.get("sha256") or "").strip()
            if not sha: continue
            lab[sha]=(row.get("label_num") or "").strip()
            split[sha]=(row.get("split") or "").strip().lower()
            aid[sha]=(row.get("asset_id") or "").strip()
    return lab, split, aid

def choose_threshold(scores, y01):
    # y01: 1 for FAKE (pos), 0 for REAL
    x=np.asarray(scores,float); y=np.asarray(y01,int)
    m=~np.isnan(x); x=x[m]; y=y[m]
    if x.size==0: return 0.5
    xs=np.unique(x)
    cands=[xs[0]-1e-6, *(((xs[:-1]+xs[1:])/2.0).tolist() if xs.size>1 else []), xs[-1]+1e-6]
    best_t,best_acc=0.5,-1
    for t in cands:
        yhat=(x>=t).astype(int)
        acc=(yhat==y).mean()
        if acc>best_acc: best_acc, best_t=acc, float(t)
    return best_t

def main():
    ap=argparse.ArgumentParser("Blend Xception+TimeSformer")
    ap.add_argument("--xception", required=True)
    ap.add_argument("--timesformer", required=True)
    ap.add_argument("--manifest", default="")
    ap.add_argument("--agg-frame", default="p95", choices=["mean","max","median","p90","p95"])
    ap.add_argument("--agg-clip",  default="median", choices=["mean","max","median","p90","p95"])
    ap.add_argument("--alpha", type=float, default=None, help="Weight for Xception (else grid-search 0..1)")
    ap.add_argument("--use-split", action="store_true")
    ap.add_argument("--val-as-test", action="store_true")
    ap.add_argument("--out", required=True)
    args=ap.parse_args()

    x=agg_map(read_jsonl_scores(Path(args.xception)), args.agg_frame)
    t=agg_map(read_jsonl_scores(Path(args.timesformer)), args.agg_clip)

    shas=sorted(set(x)|set(t))
    if not shas: 
        print("No overlapping assets."); return

    X=np.array([x.get(s,np.nan) for s in shas],float)
    T=np.array([t.get(s,np.nan) for s in shas],float)

    # simple impute: if one side NaN, use the other; if both NaN → drop later
    both=np.isnan(X)&np.isnan(T)
    X[both]=0.5; T[both]=0.5

    lab, split, aid = read_manifest(Path(args.manifest)) if args.manifest else ({},{},{})
    y = np.array([int(lab[s]) if lab.get(s,"")!="" else -1 for s in shas], int)
    # convention: label_num 0=FAKE (pos), 1=REAL (neg)
    y_pos = (y==0).astype(int)

    def fuse(a):
        a=float(np.clip(a,0,1))
        return a*np.nan_to_num(X, nan=0.0)+(1-a)*np.nan_to_num(T, nan=0.0)

    # pick alpha
    if args.alpha is None and (y>=0).any():
        # use split if available else use everything to pick alpha
        test_mask = np.array([(split.get(s,"")==("test" if any(v=="test" for v in split.values()) else ("val" if args.val_as_test else ""))) for s in shas], bool)
        train_mask= np.array([split.get(s,"")=="train" for s in shas], bool)
        if not train_mask.any(): train_mask = (y>=0)
        best_a,best_bal=-1,-1.0; best_thr=0.5
        for a in tqdm(np.linspace(0,1,21), desc="grid search alpha"):
            F=fuse(a)
            thr=choose_threshold(F[train_mask], y_pos[train_mask])
            ev_mask=test_mask if test_mask.any() else train_mask
            yhat=(F[ev_mask]>=thr).astype(int); yy=y_pos[ev_mask]
            tp=((yhat==1)&(yy==1)).sum(); tn=((yhat==0)&(yy==0)).sum()
            fp=((yhat==1)&(yy==0)).sum(); fn=((yhat==0)&(yy==1)).sum()
            tpr=tp/max(1,tp+fn); tnr=tn/max(1,tn+fp); bal=0.5*(tpr+tnr)
            if bal>best_bal: best_bal, best_a, best_thr = bal, a, thr
        alpha, thr = best_a, best_thr
        print(f"[blend] learned alpha={alpha:.2f} thr={thr:.4f} bal_acc={best_bal:.3f}")
    else:
        alpha=float(args.alpha if args.alpha is not None else 0.5)
        F=fuse(alpha)
        thr=choose_threshold(F[(y>=0)], y_pos[(y>=0)]) if (y>=0).any() else 0.5

    F=fuse(alpha)
    out=Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w",encoding="utf-8") as w:
        for s, sx, st, sf in zip(shas, X, T, F):
            rec={
                "asset_id": (aid.get(s) or None),
                "sha256": s,
                "xception_agg_score": None if np.isnan(sx) else float(sx),
                "timesformer_agg_score": None if np.isnan(st) else float(st),
                "fused_score": float(sf),
                "alpha_xception": float(alpha),
                "decision_threshold": float(thr),
                "predicted_label": "FAKE" if sf>=thr else "REAL",
                "model_name": "xception+timesformer (blend)",
                "model_version": "v1",
            }
            w.write(json.dumps(rec)+"\n")
    print(f"Wrote → {out}")

if __name__=="__main__":
    main()
