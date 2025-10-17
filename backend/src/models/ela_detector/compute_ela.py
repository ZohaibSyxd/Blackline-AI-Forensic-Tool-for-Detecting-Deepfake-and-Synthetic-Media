"""
Error Level Analysis (ELA) over sampled frames.

Inputs:
  - frames.jsonl (from sample_frames.py) with rows containing:
      sha256, shot_index, frame_index, uri (relative to frames root)
  - frames root directory containing JPEGs under frames/<sha>/<shot>/<nnnnnn>.jpg

Outputs:
  - Overlays: overlays/ela/<sha>/<shot>/<frame>.jpg (visual ELA map)
  - Metrics JSONL: frames_ela.jsonl with per-frame metrics:
      { asset_id, sha256, shot_index, frame_index, uri, ela_error_mean, ela_error_max }

Notes:
  - Non-destructive: creates new files only; does not modify existing assets.
  - ELA implementation: recompress at a target quality, diff absolute, optionally scale.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict
from ...audit import audit_step

import numpy as np
from PIL import Image, ImageChops


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def compute_ela_image(img: Image.Image, recompress_quality: int = 85, scale: float = 10.0) -> Image.Image:
    """Compute ELA image by recompressing and taking absolute difference.

    Args:
        img: PIL Image in RGB
        recompress_quality: JPEG quality to recompress at (60-95 sensible)
        scale: multiply diff to enhance visibility
    Returns:
        ELA visualization image (RGB)
    """
    # Recompress into memory buffer
    from io import BytesIO

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=recompress_quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")

    diff = ImageChops.difference(img, recompressed)
    if scale != 1.0:
        # Enhance visibility; convert to float, scale, clip
        diff = Image.fromarray(
            np.clip(np.array(diff, dtype=np.float32) * scale, 0, 255).astype(np.uint8)
        )
    return diff


def ela_metrics(ela_img: Image.Image) -> Dict[str, float]:
    arr = np.asarray(ela_img, dtype=np.uint8)
    if arr.ndim == 3:
        # Convert to grayscale energy by max across channels (common for ELA)
        gray = arr.max(axis=2)
    else:
        gray = arr
    return {
        "ela_error_mean": float(gray.mean()),
        "ela_error_max": float(gray.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute Error Level Analysis (ELA) overlays and metrics.")
    ap.add_argument("--frames", default="backend/data/derived/frames.jsonl", help="Path to frames JSONL")
    ap.add_argument("--root", default="backend/data/derived/frames", help="Root folder containing frames")
    ap.add_argument("--out", default="backend/data/derived/frames_ela.jsonl", help="Output metrics JSONL")
    ap.add_argument("--overlays", default="backend/data/derived/overlays", help="Base output dir for overlays")
    ap.add_argument("--jpeg-quality", type=int, default=85, help="Recompress quality for ELA (60-95)")
    ap.add_argument("--scale", type=float, default=10.0, help="Intensity scale for ELA diff")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on frames processed")
    args = ap.parse_args()

    frames_root = Path(args.root)
    overlays_root = Path(args.overlays) / "ela"
    ensure_dir(overlays_root)

    processed = 0
    with audit_step("compute_ela", params=vars(args), inputs={"frames": args.frames}) as outputs:
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
                    img = Image.open(src_path).convert("RGB")
                    ela_img = compute_ela_image(img, recompress_quality=args.jpeg_quality, scale=args.scale)
                    m = ela_metrics(ela_img)

                    # Save overlay
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
                        **m,
                    }
                    fw.write(json.dumps(out_row) + "\n")
                    processed += 1
                except Exception:
                    # Skip corrupted images
                    continue

                if args.limit and processed >= args.limit:
                    break

        outputs["frames_ela"] = {"path": args.out}
    print(f"ELA processed {processed} frames → {args.out}")


if __name__ == "__main__":
    main()
