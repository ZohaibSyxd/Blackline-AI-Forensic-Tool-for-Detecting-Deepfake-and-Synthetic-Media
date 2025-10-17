"""
Select and save only the top-K most suspicious frames by ELA score.

- Reads frames.jsonl (from sample_frames.py): each row needs sha256, shot_index, frame_index, uri
- Computes ELA metrics for all frames, ranks by chosen metric (mean or max)
- Saves overlays and metrics only for the top-K frames

Outputs:
  - Overlays: overlays/ela_topk/<sha>/<shot>/<frame>.jpg
  - Metrics JSONL: frames_ela_topk.jsonl (only top-K rows)

Notes:
  - Ranking uses the unscaled ELA difference (scale=1.0). Saved overlays can use a visual scale.
  - Non-destructive.
"""

from __future__ import annotations

import argparse
import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageChops


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def compute_ela_image(img: Image.Image, recompress_quality: int = 85, scale: float = 10.0) -> Image.Image:
    """Compute ELA image by recompressing and taking absolute difference."""
    from io import BytesIO

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=recompress_quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")

    diff = ImageChops.difference(img, recompressed)
    if scale != 1.0:
        diff = Image.fromarray(
            np.clip(np.array(diff, dtype=np.float32) * scale, 0, 255).astype(np.uint8)
        )
    return diff


def ela_metrics(ela_img: Image.Image) -> Dict[str, float]:
    arr = np.asarray(ela_img, dtype=np.uint8)
    if arr.ndim == 3:
        gray = arr.max(axis=2)
    else:
        gray = arr
    return {
        "ela_error_mean": float(gray.mean()),
        "ela_error_max": float(gray.max()),
    }


@dataclass
class FrameRef:
    score: float
    row: Dict[str, Any]


def main() -> None:
    ap = argparse.ArgumentParser(description="Save only top-K most suspicious frames by ELA score.")
    ap.add_argument("--frames", default="backend/data/derived/frames.jsonl", help="Path to frames JSONL")
    ap.add_argument("--root", default="backend/data/derived/frames", help="Root folder containing frames")
    ap.add_argument("--out", default="backend/data/derived/frames_ela_topk.jsonl", help="Output metrics JSONL (top-K only)")
    ap.add_argument("--overlays", default="backend/data/derived/overlays", help="Base output dir for overlays")
    ap.add_argument("--jpeg-quality", type=int, default=85, help="Recompress quality for ELA (60-95)")
    ap.add_argument("--scale", type=float, default=10.0, help="Intensity scale for saved ELA overlays")
    ap.add_argument("-k", "--topk", type=int, required=True, help="Number of most suspicious frames to keep")
    ap.add_argument("--metric", choices=["max", "mean"], default="max", help="ELA metric used for ranking")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on frames considered")
    args = ap.parse_args()

    frames_root = Path(args.root)
    overlays_root = Path(args.overlays) / "ela_topk"
    ensure_dir(overlays_root)

    # First pass: scan all frames, compute ELA score (unscaled), maintain min-heap of size K
    heap: List[Tuple[float, int, Dict[str, Any]]] = []
    processed = 0
    tie = 0

    with open(args.frames, encoding="utf-8") as fr:
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

            try:
                src_path = frames_root / Path(uri).relative_to("frames")  # uri starts with frames/
            except Exception:
                continue

            if not src_path.exists():
                continue

            try:
                # Compute score using unscaled ELA
                img = Image.open(src_path).convert("RGB")
                ela_img_unscaled = compute_ela_image(img, recompress_quality=args.jpeg_quality, scale=1.0)
                m = ela_metrics(ela_img_unscaled)
                score = m["ela_error_max"] if args.metric == "max" else m["ela_error_mean"]

                # Keep smallest on top; if heap not full, push; else replace if score better
                if len(heap) < args.topk:
                    heapq.heappush(heap, (score, tie, {**row, **m}))
                else:
                    if score > heap[0][0]:
                        heapq.heapreplace(heap, (score, tie, {**row, **m}))
                tie += 1
                processed += 1
            except Exception:
                # Skip corrupted images
                continue

            if args.limit and processed >= args.limit:
                break

    if not heap:
        print("No frames ranked; nothing to save.")
        return

    # Sort selected frames by score descending
    selected: List[Tuple[float, int, Dict[str, Any]]] = sorted(heap, key=lambda x: (-x[0], x[1]))

    # Second pass: save overlays only for top-K and write metrics JSONL
    with open(args.out, "w", encoding="utf-8") as fw:
        for rank, (score, _, row) in enumerate(selected, start=1):
            uri = row["uri"]
            sha = row["sha256"]
            shot_idx = row["shot_index"]
            frame_idx = row["frame_index"]

            src_path = frames_root / Path(uri).relative_to("frames")
            if not src_path.exists():
                # If missing now, skip saving but still write row
                continue

            try:
                img = Image.open(src_path).convert("RGB")
                ela_img = compute_ela_image(img, recompress_quality=args.jpeg_quality, scale=args.scale)

                out_dir = overlays_root / str(sha) / str(shot_idx)
                ensure_dir(out_dir)
                out_path = out_dir / f"{int(frame_idx):06d}.jpg"
                ela_img.save(out_path, format="JPEG", quality=95)

                out_row = {
                    "asset_id": row.get("asset_id"),
                    "sha256": sha,
                    "shot_index": shot_idx,
                    "frame_index": frame_idx,
                    "uri": uri,
                    "overlay_uri": str(out_path.relative_to(overlays_root.parent).as_posix()),
                    "ela_error_mean": float(row.get("ela_error_mean", 0.0)),
                    "ela_error_max": float(row.get("ela_error_max", 0.0)),
                    "rank": rank,
                    "score_metric": f"ela_error_{args.metric}",
                    "score": float(score),
                }
                fw.write(json.dumps(out_row) + "\n")
            except Exception:
                # Skip if anything goes wrong for this frame
                continue

    print(f"Ranked {processed} frames, saved top-{len(selected)} overlays → {args.out}")


if __name__ == "__main__":
    main()