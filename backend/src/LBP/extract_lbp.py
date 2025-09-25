"""
src/extract_lbp.py

Extract Local Binary Pattern (LBP) texture features from sampled frames.
LBP histograms capture local micro-texture details that can reveal artifacts
common in manipulated or deepfake videos.

Usage (PowerShell)
        python .\src\LBP\extract_lbp.py \
                --frames .\data\derived\frames.jsonl \
                --frames-root .\data\derived\frames \
                --out .\data\derived\lbp_features.jsonl \
                --radius 2 \
                --neighbors 16 \
                --method uniform

Notes
- The frames.jsonl file is written by sample_frames.py and contains per-frame
  metadata including asset_id, shot_index, frame_index, approx_t_ms, and URI.
- Each frame image (JPEG) is loaded and converted to grayscale.
- LBP is computed using skimage.feature.local_binary_pattern with the specified
  parameters.
- For each frame, we compute a normalized histogram of LBP codes.
- Output is JSONL: one row per frame, keyed by asset_id/shot/frame_index with
  the LBP histogram as a feature vector.
- These features can be aggregated later at the shot- or video-level and fed
  into classifiers (e.g., SVM, logistic regression, or neural nets).
"""

import argparse
import json
import os
from skimage import io, color
from skimage.util import img_as_ubyte
from skimage.feature import local_binary_pattern
import numpy as np
from tqdm import tqdm

def extract_lbp_histogram(img_path, radius=2, neighbors=16, method="uniform"):
    img = io.imread(img_path)
    # grayscale
    gray = color.rgb2gray(img)

    # convert to uint8
    gray_uint8 = img_as_ubyte(gray)

    # compute LBP
    lbp = local_binary_pattern(gray_uint8, neighbors, radius, method)
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, neighbors + 3),
        range=(0, neighbors + 2),
        density=True
    )
    return hist.tolist()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True, help="Path to frames.jsonl from sample_frames.py")
    parser.add_argument("--frames-root", required=True, help="Root directory where frame images are stored")
    parser.add_argument("--out", required=True, help="Output JSONL file with LBP features")
    parser.add_argument("--radius", type=int, default=2, help="Radius of circular LBP neighborhood")
    parser.add_argument("--neighbors", type=int, default=16, help="Number of sampling points in LBP")
    parser.add_argument("--method", type=str, default="uniform", choices=["default", "ror", "uniform", "var"],
                        help="LBP method variant")
    args = parser.parse_args()

    with open(args.frames, "r") as f_in, open(args.out, "w") as f_out:
        for line in tqdm(f_in, desc="Extracting LBP"):
            row = json.loads(line)

            # pick face_uri if present, else uri
            img_key = "face_uri" if "face_uri" in row else "uri"
            img_rel = row.get(img_key)
            if not img_rel:
                continue

            # join cleanly: frames-root or faces-root should point to the top dir
            img_path = os.path.join(args.frames_root, img_rel.replace("faces/", "").replace("frames/", "").lstrip("/\\"))

            if not os.path.exists(img_path):
                print(f"Missing image: {img_path}")
                continue

            try:
                hist = extract_lbp_histogram(img_path, args.radius, args.neighbors, args.method)
                out_row = {
                    "asset_id": row.get("asset_id"),
                    "sha256": row.get("sha256"),
                    "shot_index": row.get("shot_index"),
                    "frame_index": row.get("frame_index"),
                    "approx_t_ms": row.get("approx_t_ms"),
                    "lbp_hist": hist
                }
                f_out.write(json.dumps(out_row) + "\n")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")


if __name__ == "__main__":
    main()
