"""
src/aggregate_lbp.py

Aggregate per-frame LBP histograms into per-video features for downstream
classification. Input is JSONL written by lbp_features.py (one row per frame).

Usage (PowerShell)
        python .\src\LBP\aggregate_lbp.py \
                --features .\data\derived\lbp_features.jsonl \
                --out .\data\derived\lbp_videos.jsonl \
                --method mean

Notes
- Input rows contain {"video": <id>, "frame": <int>, "lbp": <histogram list>}.
- We group by video and aggregate histograms using the chosen method.
- Supported methods:
    * mean  → average histogram across frames
    * median → elementwise median
    * max   → elementwise max
- Output rows contain {"video": <id>, "lbp": <aggregated histogram>}.
"""

import argparse
import json
from collections import defaultdict
import numpy as np


def load_features(path):
    data = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue  # skip blank lines
            row = json.loads(line)
            key = row.get("asset_id")
            if key is None:
                raise KeyError(f"Row missing asset_id field: {row}")
            if "lbp_hist" not in row:
                raise KeyError(f"Row missing lbp_hist field: {row}")
            data[key].append(row["lbp_hist"])
    return data



def aggregate(hist_list, method="mean"):
    arr = np.array(hist_list)
    if method == "mean":
        return arr.mean(axis=0).tolist()
    elif method == "median":
        return np.median(arr, axis=0).tolist()
    elif method == "max":
        return arr.max(axis=0).tolist()
    else:
        raise ValueError(f"Unknown method: {method}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Input JSONL (per-frame LBP features)")
    parser.add_argument("--out", required=True, help="Output JSONL (per-video LBP features)")
    parser.add_argument("--method", default="mean", choices=["mean", "median", "max"], help="Aggregation method")
    args = parser.parse_args()

    features = load_features(args.features)

    with open(args.out, "w", encoding="utf-8") as f:
        for asset_id, hist_list in features.items():
            agg = aggregate(hist_list, method=args.method)
            json.dump({"asset_id": asset_id, "lbp": agg}, f)
            f.write("\n")



if __name__ == "__main__":
    main()
