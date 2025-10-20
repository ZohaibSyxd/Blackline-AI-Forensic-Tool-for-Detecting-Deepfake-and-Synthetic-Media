"""
Extract high-confidence frames from ELA per-frame metrics for later reporting.

Inputs:
  - frames_ela.jsonl (from compute_ela.py) with rows containing at least:
      { sha256, shot_index, frame_index, uri, overlay_uri, ela_error_mean, ela_error_max }

Behavior:
  - Select frames whose ELA metric is above a very high threshold.
  - Threshold can be given explicitly or derived from a high quantile (default q=0.995).
  - Copy overlay and/or original frame images to an output folder and write an index JSONL.

Usage (PowerShell):
  # Using 99.5th percentile of ela_error_mean
  python .\backend\src\models\ela_detector\extract_high_conf_frames.py `
    --ela        .\backend\data\derived\frames_ela.jsonl `
    --out-dir    .\backend\data\derived\report\ela_top_frames `
    --metric     ela_error_mean `
    --quantile   0.995 `
    --copy       overlay

  # Or set an explicit numeric threshold
  # python .\backend\src\models\ela_detector\extract_high_conf_frames.py `
  #   --threshold 70.0 --copy both
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract high-confidence frames from ELA metrics.")
    ap.add_argument("--ela", default="backend/data/derived/frames_ela.jsonl", help="Path to frames_ela.jsonl")
    ap.add_argument("--metric", choices=["ela_error_mean", "ela_error_max"], default="ela_error_mean")
    ap.add_argument("--threshold", type=float, default=None, help="Absolute threshold to select frames (overrides --quantile)")
    ap.add_argument("--quantile", type=float, default=0.995, help="Quantile in [0,1] for automatic threshold if --threshold not set")
    ap.add_argument("--frames-root", default="backend/data/derived/frames", help="Root folder containing original frames")
    ap.add_argument("--overlays-root", default="backend/data/derived/overlays", help="Base folder containing ELA overlays")
    ap.add_argument("--out-dir", default="backend/data/derived/report/ela_top_frames", help="Output directory to copy selected frames")
    ap.add_argument("--copy", choices=["overlay", "original", "both"], default="overlay", help="Which images to copy")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on number of frames to copy")
    return ap.parse_args()


def load_ela_rows(path: Path, metric: str) -> List[Tuple[Dict[str, Any], float]]:
    rows: List[Tuple[Dict[str, Any], float]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec: Dict[str, Any] = json.loads(line)
            except Exception:
                continue
            v = rec.get(metric)
            if v is None:
                continue
            try:
                rows.append((rec, float(v)))
            except Exception:
                continue
    return rows


def main() -> None:
    args = parse_args()

    ela_path = Path(args.ela)
    frames_root = Path(args.frames_root)
    overlays_root = Path(args.overlays_root)  # overlay_uri is relative to this base
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read rows and collect metric values
    pairs = load_ela_rows(ela_path, args.metric)
    if not pairs:
        print("No ELA rows found or metric missing; nothing to extract.")
        return

    vals = np.array([v for _, v in pairs], dtype=float)
    thr = float(args.threshold) if args.threshold is not None else float(np.quantile(vals, min(max(args.quantile, 0.0), 1.0)))

    copied = 0
    index_path = out_dir / "selected_ela_frames.jsonl"
    # Create subdirs on demand
    def _dst(kind: str, sha: str, shot: int, frame_idx: int) -> Path:
        d = out_dir / kind / str(sha) / str(shot)
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{int(frame_idx):06d}.jpg"

    with open(index_path, "w", encoding="utf-8") as fout:
        for rec, v in pairs:
            if v < thr:
                continue

            sha = rec.get("sha256")
            shot = rec.get("shot_index")
            frame_idx = rec.get("frame_index")
            if sha is None or shot is None or frame_idx is None:
                continue

            copied_any = False
            src_overlay = None
            src_frame = None

            if args.copy in ("overlay", "both"):
                ov_uri = rec.get("overlay_uri")
                if isinstance(ov_uri, str) and ov_uri:
                    src_overlay = overlays_root / ov_uri
                    if src_overlay.exists():
                        dst = _dst("overlay", str(sha), int(shot), int(frame_idx))
                        shutil.copy2(src_overlay, dst)
                        copied_any = True

            if args.copy in ("original", "both"):
                uri = rec.get("uri")
                if isinstance(uri, str) and uri:
                    # uri starts with "frames/"; resolve under frames_root
                    try:
                        src_frame = frames_root / Path(uri).relative_to("frames")
                    except Exception:
                        src_frame = frames_root / uri
                    if src_frame.exists():
                        dst = _dst("frames", str(sha), int(shot), int(frame_idx))
                        shutil.copy2(src_frame, dst)
                        copied_any = True

            if copied_any:
                fout.write(json.dumps({
                    "sha256": sha,
                    "shot_index": shot,
                    "frame_index": frame_idx,
                    "metric": args.metric,
                    "value": float(v),
                    "threshold": thr,
                    "overlay_src": str(src_overlay) if src_overlay else None,
                    "frame_src": str(src_frame) if src_frame else None,
                }) + "\n")
                copied += 1
                if args.limit and copied >= args.limit:
                    break

    print(f"Selected {copied} frames with {args.metric} >= {thr:.4f} → {out_dir}")


if __name__ == "__main__":
    main()
