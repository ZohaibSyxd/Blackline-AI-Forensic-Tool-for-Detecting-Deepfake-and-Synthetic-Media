"""
src/train_lbp.py

Train a classifier on per-video aggregated LBP histograms with REAL/FAKE labels.

Input is manifest.csv (built by build_manifest.py), which must contain:
- asset_id
- lbp (JSON stringified histogram list)
- label (REAL/FAKE)
- label_num (1 for REAL,0 for FAKE)

We parse the LBP vectors, split into train/val, fit a classifier, and save
the trained model to disk.

Usage (PowerShell)
    python .\backend\src\LBP\train_lbp.py \
        --manifest .\backend\data\derived\manifest.csv \
        --model_out .\backend\src\LBP\lbp_model.joblib \
        --limit 1000

Notes
- Uses scikit-learn LogisticRegression by default.
- Can switch to RandomForest via --model rf.
"""

import argparse, csv, json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import joblib


def load_manifest(path, limit=None):
    """Load manifest.csv and return X (features) and y (labels)."""
    X, y, ids = [], [], []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            lbp_str = row.get("lbp")
            label_num = row.get("label_num")
            if not lbp_str or label_num == "":
                continue  # skip if missing
            try:
                hist = json.loads(lbp_str)
            except Exception:
                continue
            X.append(hist)
            y.append(int(label_num))
            ids.append(row.get("asset_id"))
    return np.array(X), np.array(y), ids


def main():
    ap = argparse.ArgumentParser(description="Train classifier on per-video LBP features.")
    ap.add_argument("--manifest", required=True, help="Path to manifest.csv with LBP + labels")
    ap.add_argument("--model_out", required=True, help="Path to save trained model (.joblib)")
    ap.add_argument("--limit", type=int, default=None, help="Optional row limit for debugging")
    ap.add_argument("--model", choices=["logreg", "rf"], default="logreg", help="Classifier type")
    args = ap.parse_args()

    X, y, ids = load_manifest(args.manifest, args.limit)
    if X.size == 0:
        print("No usable rows in manifest. Did you attach lbp + labels?")
        return

    print(f"Loaded {len(X)} samples from {args.manifest}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    reports = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if args.model == "logreg":
            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        else:  # rf
            clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        print(f"\nFold {fold+1} validation report:")
        print(classification_report(y_val, y_pred, target_names=["FAKE", "REAL"]))
        reports.append(classification_report(y_val, y_pred, target_names=["FAKE", "REAL"], output_dict=True))

    # Retrain on all data and save the final model
    if args.model == "logreg":
        final_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    else:
        final_clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    final_clf.fit(X, y)
    Path(Path(args.model_out).parent).mkdir(parents=True, exist_ok=True)
    joblib.dump(final_clf, args.model_out)
    print(f"\nSaved final model trained on all data -> {args.model_out}")


if __name__ == "__main__":
    main()
