"""
src/models/predict_lbp.py

Load a trained LBP classifier (joblib) and run predictions on aggregated
per-video LBP histograms. Input is JSONL written by aggregate_lbp.py.

Usage (PowerShell)
    python .\src\LBP\predict_lbp.py \
        --model .\src\LBP\lbp_model.joblib \
        --features .\data\derived\lbp_videos.jsonl \
        --out .\data\derived\lbp_predictions.jsonl

Notes
- Input rows contain {"asset_id": <id>, "lbp": <aggregated histogram>}.
- The model is loaded from joblib (saved in train_lbp.py).
- Output rows contain {"asset_id": <id>, "pred_label": <REAL/FAKE>, "pred_score": <probability of FAKE>}.
"""

import argparse
import json
import joblib
import numpy as np
from pathlib import Path
from ..audit import audit_step


def load_features(path):
    """Load aggregated LBP features from JSONL."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            asset_id = row.get("asset_id")
            lbp = row.get("lbp")
            if asset_id is None or lbp is None:
                continue
            rows.append((asset_id, np.array(lbp)))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Predict REAL/FAKE labels from LBP features.")
    parser.add_argument("--model", required=True, help="Path to trained model (joblib).")
    parser.add_argument("--features", required=True, help="Input JSONL (per-video LBP features).")
    parser.add_argument("--out", required=True, help="Output JSONL (predictions).")
    args = parser.parse_args()

    # Load model
    model = joblib.load(args.model)

    # Load features
    rows = load_features(args.features)
    if not rows:
        print("No features found.")
        return

    asset_ids, X = zip(*rows)
    X = np.vstack(X)

    # Mapping: 0 = FAKE, 1 = REAL
    label_map = {0: "FAKE", 1: "REAL"}

    # Predict
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        preds = model.predict(X)
        fake_index = list(model.classes_).index(0)  # numeric 0 = FAKE
        scores = probs[:, fake_index]  # probability of FAKE
    else:
        preds = model.predict(X)
        scores = np.array([1.0 if p == 0 else 0.0 for p in preds])

    # Write output
    Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
    with audit_step("predict_lbp", params=vars(args), inputs={"features": args.features, "model": args.model}) as outputs:
      with open(args.out, "w", encoding="utf-8") as f:
        for asset_id, pred, score in zip(asset_ids, preds, scores):
            json.dump({
                "asset_id": asset_id,
                "pred_label": label_map.get(int(pred), str(pred)),
                "pred_score": float(score),
            }, f)
            f.write("\n")
      outputs["lbp_predictions"] = {"path": args.out}
    print(f"Wrote {len(asset_ids)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
