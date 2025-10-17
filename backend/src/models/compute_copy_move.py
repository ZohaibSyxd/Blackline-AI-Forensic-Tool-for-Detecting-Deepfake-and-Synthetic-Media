"""
Copy-Move Forgery Detection over sampled frames (keypoint + translation clustering).

Inputs:
  - frames.jsonl (from sample_frames.py) with rows containing:
      sha256, shot_index, frame_index, uri (relative to frames root)
  - frames root directory containing JPEGs under frames/<sha>/<shot>/<nnnnnn>.jpg

Outputs:
  - Overlays: overlays/copy_move/<sha>/<shot>/<frame>.jpg (visualization)
  - Metrics JSONL: frames_copy_move.jsonl with per-frame metrics:
      { asset_id, sha256, shot_index, frame_index, uri,
        overlay_uri, cm_num_keypoints, cm_num_matches,
        cm_dominant_dx, cm_dominant_dy, cm_shift_magnitude,
        cm_coverage_ratio, cm_confidence }

Notes:
  - Simple, fast approach using ORB keypoints + Hamming matcher within the same image,
    followed by translation vector clustering (integer binning). This highlights likely
    duplicated regions with a consistent shift.
  - This is a heuristic detector; high scores warrant further analysis rather than final proof.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from ..audit import audit_step

import numpy as np
import cv2


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def detect_copy_move(
    bgr: np.ndarray,
    max_features: int = 1200,
    min_shift: float = 10.0,
    bin_size: float = 4.0,
    min_pairs: int = 8,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Detect copy-move via intra-image keypoint matching and translation clustering.

    Args:
        bgr: Input color image (H,W,3) in BGR
        max_features: ORB feature cap
        min_shift: minimum pixel shift to consider as copy-move (avoid trivial neighbors)
        bin_size: quantization step for translation clustering
        min_pairs: minimum matched pairs supporting a dominant translation

    Returns:
        overlay_bgr: visualization (BGR) with matches and mask overlay
        metrics: dict with cm_* fields
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=max_features)
    kps, des = orb.detectAndCompute(gray, None)

    metrics: Dict[str, Any] = {
        "cm_num_keypoints": int(len(kps) if kps else 0),
        "cm_num_matches": 0,
        "cm_dominant_dx": 0.0,
        "cm_dominant_dy": 0.0,
        "cm_shift_magnitude": 0.0,
        "cm_coverage_ratio": 0.0,
        "cm_confidence": 0.0,
    }

    overlay = bgr.copy()
    if des is None or len(kps) < 2:
        return overlay, metrics

    # Intra-image matching: match each descriptor to the nearest different keypoint
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    # knnMatch against itself; we'll filter i==j and small spatial shifts
    knn = bf.knnMatch(des, des, k=2)

    # Collect plausible pairs with their translation vectors
    pairs: List[Tuple[int, int, float, float, float]] = []  # (i, j, dx, dy, dist)
    for i, nbrs in enumerate(knn):
        if not nbrs:
            continue
        # choose the best neighbor that's not itself
        best = None
        for m in nbrs:
            if m.trainIdx != i:
                best = m
                break
        if best is None:
            continue
        j = best.trainIdx
        pi = np.array(kps[i].pt)
        pj = np.array(kps[j].pt)
        dxy = pj - pi
        shift = float(np.hypot(dxy[0], dxy[1]))
        if shift < min_shift:
            continue
        pairs.append((i, j, float(dxy[0]), float(dxy[1]), float(best.distance)))

    if not pairs:
        return overlay, metrics

    # Bin translations to find dominant shift
    def quant(v: float) -> int:
        return int(np.round(v / bin_size))

    bins: Dict[Tuple[int, int], List[int]] = {}
    for idx, (_, _, dx, dy, _) in enumerate(pairs):
        key = (quant(dx), quant(dy))
        bins.setdefault(key, []).append(idx)

    # Choose bin with most support
    dom_key = max(bins.keys(), key=lambda k: len(bins[k]))
    supporter_idx = bins[dom_key]
    if len(supporter_idx) < min_pairs:
        # Not enough consensus; return early with low-confidence overlay
        return overlay, metrics

    # Compute refined average shift from supporters
    sel = [pairs[t] for t in supporter_idx]
    dxs = np.array([p[2] for p in sel], dtype=np.float32)
    dys = np.array([p[3] for p in sel], dtype=np.float32)
    dmean = np.array([dxs.mean(), dys.mean()])

    # Build mask from both source and destination keypoints for the dominant translation
    mask = np.zeros((h, w), dtype=np.uint8)
    for (i, j, _, _, _) in sel:
        xi, yi = kps[i].pt
        xj, yj = kps[j].pt
        cv2.circle(mask, (int(round(xi)), int(round(yi))), 6, 255, -1)
        cv2.circle(mask, (int(round(xj)), int(round(yj))), 6, 255, -1)
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)

    # Visualization: draw matched pairs and semi-transparent mask
    vis = overlay
    # Blend mask in red
    red = np.zeros_like(vis)
    red[:, :, 2] = 255
    alpha = 0.25
    vis = cv2.addWeighted(vis, 1.0, red, alpha, 0, dst=vis, mask=mask)

    # Draw lines for a subset (to avoid clutter)
    max_draw = min(200, len(sel))
    for (i, j, _, _, dist) in sel[:max_draw]:
        pi = tuple(int(round(v)) for v in kps[i].pt)
        pj = tuple(int(round(v)) for v in kps[j].pt)
        cv2.circle(vis, pi, 3, (0, 255, 0), -1)
        cv2.circle(vis, pj, 3, (0, 255, 0), -1)
        cv2.line(vis, pi, pj, (0, 255, 255), 1)

    coverage = float(mask.sum() / 255) / float(h * w)
    dx, dy = float(dmean[0]), float(dmean[1])
    shift_mag = float((dmean[0] ** 2 + dmean[1] ** 2) ** 0.5)
    confidence = float(min(1.0, len(sel) / max(1, len(kps))))

    metrics.update({
        "cm_num_matches": int(len(sel)),
        "cm_dominant_dx": dx,
        "cm_dominant_dy": dy,
        "cm_shift_magnitude": shift_mag,
        "cm_coverage_ratio": coverage,
        "cm_confidence": confidence,
    })

    return vis, metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Copy-Move detection over sampled frames (ORB + clustering)")
    ap.add_argument("--frames", default="backend/data/derived/frames.jsonl", help="Path to frames JSONL")
    ap.add_argument("--root", default="backend/data/derived/frames", help="Root folder containing frames")
    ap.add_argument("--out", default="backend/data/derived/frames_copy_move.jsonl", help="Output metrics JSONL")
    ap.add_argument("--overlays", default="backend/data/derived/overlays", help="Base output dir for overlays")
    ap.add_argument("--max-features", type=int, default=1200)
    ap.add_argument("--min-shift", type=float, default=10.0)
    ap.add_argument("--bin-size", type=float, default=4.0)
    ap.add_argument("--min-pairs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on frames processed")
    args = ap.parse_args()

    frames_root = Path(args.root)
    overlays_root = Path(args.overlays) / "copy_move"
    ensure_dir(overlays_root)

    processed = 0
    with audit_step("compute_copy_move", params=vars(args), inputs={"frames": args.frames}) as outputs:
        with open(args.out, "w", encoding="utf-8") as fw, open(args.frames, encoding="utf-8") as fr:
            for line in fr:
                if not line.strip():
                    continue
                try:
                    row: Dict[str, Any] = json.loads(line)
                except Exception:
                    continue

                uri = row.get("uri")
                sha = row.get("sha256")
                shot_idx = row.get("shot_index")
                frame_idx = row.get("frame_index")
                if not uri or sha is None or shot_idx is None or frame_idx is None:
                    continue

                src_path = frames_root / Path(uri).relative_to("frames")  # uri starts with frames/
                if not src_path.exists():
                    continue

                try:
                    bgr = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
                    if bgr is None:
                        continue
                    vis, m = detect_copy_move(
                        bgr,
                        max_features=args.max_features,
                        min_shift=args.min_shift,
                        bin_size=args.bin_size,
                        min_pairs=args.min_pairs,
                    )

                    # Save overlay
                    out_dir = overlays_root / str(sha) / str(shot_idx)
                    ensure_dir(out_dir)
                    out_path = out_dir / f"{int(frame_idx):06d}.jpg"
                    cv2.imwrite(str(out_path), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                    out_row = {
                        "asset_id": row.get("asset_id"),
                        "sha256": sha,
                        "shot_index": shot_idx,
                        "frame_index": frame_idx,
                        "uri": uri,
                        "overlay_uri": str(out_path.relative_to(overlays_root.parent).as_posix()),
                        **m,
                    }
                    fw.write(json.dumps(out_row) + "\n")
                    processed += 1
                except Exception:
                    # Skip on any per-frame failure
                    continue

                if args.limit and processed >= args.limit:
                    break

        outputs["frames_copy_move"] = {"path": args.out}
    print(f"Copy-Move processed {processed} frames → {args.out}")


if __name__ == "__main__":
    main()
