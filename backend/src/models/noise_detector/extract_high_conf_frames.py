"""
Extract high-confidence frames from Noise per-frame metrics for later reporting.

Inputs:
  - frames_noise.jsonl (from compute_noise.py) with rows containing at least:
      {
        sha256, shot_index, frame_index, uri,
        overlay_residual_uri, overlay_fft_uri,
        residual_abs_mean, residual_std, residual_energy,
        fft_low_ratio, fft_high_ratio
      }

Behavior:
  - Select frames whose chosen Noise metric is above (or below) a threshold.
  - Threshold can be given explicitly or derived from a high/low quantile.
  - Copy overlay(s) (residual and/or FFT) and/or original frame images to an output folder and write an index JSONL.

Usage (PowerShell):
  # Using 99.5th percentile of residual_energy, copying residual overlays
  python .\backend\src\models\noise_detector\extract_high_conf_frames.py `
    --noise       .\backend\data\derived\frames_noise.jsonl `
    --out-dir     .\backend\data\derived\report\noise_top_frames `
    --metric      residual_energy `
    --quantile    0.995 `
    --select      high `
    --copy        overlay `
    --overlay-kind residual

  # Copy both overlays and originals using an explicit threshold
  # python .\backend\src\models\noise_detector\extract_high_conf_frames.py `
  #   --metric residual_abs_mean --threshold 12.5 --copy both --overlay-kind both
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


NOISE_METRICS = [
    "residual_abs_mean",
    "residual_std",
    "residual_energy",
    "fft_low_ratio",
    "fft_high_ratio",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract high-confidence frames from Noise metrics.")
    ap.add_argument("--noise", default="backend/data/derived/frames_noise.jsonl", help="Path to frames_noise.jsonl")
    ap.add_argument("--metric", choices=NOISE_METRICS, default="residual_energy")
    ap.add_argument("--threshold", type=float, default=None, help="Absolute threshold to select frames (overrides --quantile)")
    ap.add_argument("--quantile", type=float, default=0.995, help="Quantile in [0,1] for automatic threshold if --threshold not set")
    ap.add_argument("--select", choices=["high", "low"], default="high", help="Select values above (high) or below (low) threshold")
    ap.add_argument("--frames-root", default="backend/data/derived/frames", help="Root folder containing original frames")
    ap.add_argument("--overlays-root", default="backend/data/derived/overlays", help="Base folder containing noise overlays")
    ap.add_argument("--out-dir", default="backend/data/derived/report/noise_top_frames", help="Output directory to copy selected frames")
    ap.add_argument("--copy", choices=["overlay", "original", "both"], default="overlay", help="Which images to copy")
    ap.add_argument("--overlay-kind", choices=["residual", "fft", "both"], default="residual", help="Which overlay(s) to copy if --copy includes overlays")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on number of frames to copy")
    return ap.parse_args()


def load_noise_rows(path: Path, metric: str) -> List[Tuple[Dict[str, Any], float]]:
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

    noise_path = Path(args.noise)
    frames_root = Path(args.frames_root)
    overlays_root = Path(args.overlays_root)  # overlay_*_uri is relative to this base (e.g., "noise/..." or "noise_fft/...")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read rows and collect metric values
    pairs = load_noise_rows(noise_path, args.metric)
    if not pairs:
        print("No Noise rows found or metric missing; nothing to extract.")
        return

    vals = np.array([v for _, v in pairs], dtype=float)
    q = float(np.clip(args.quantile, 0.0, 1.0))
    if args.threshold is not None:
        thr = float(args.threshold)
    else:
        thr = float(np.quantile(vals, q if args.select == "high" else (1.0 - q)))

    cmp = (lambda v: v >= thr) if args.select == "high" else (lambda v: v <= thr)

    copied = 0
    index_path = out_dir / "selected_noise_frames.jsonl"

    # Create subdirs on demand
    def _dst(kind: str, sha: str, shot: int, frame_idx: int) -> Path:
        d = out_dir / kind / str(sha) / str(shot)
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{int(frame_idx):06d}.jpg"

    with open(index_path, "w", encoding="utf-8") as fout:
        for rec, v in pairs:
            if not cmp(v):
                continue

            sha = rec.get("sha256")
            shot = rec.get("shot_index")
            frame_idx = rec.get("frame_index")
            if sha is None or shot is None or frame_idx is None:
                continue

            copied_any = False
            src_overlay_residual = None
            src_overlay_fft = None
            src_frame = None

            if args.copy in ("overlay", "both"):
                if args.overlay_kind in ("residual", "both"):
                    ov_uri = rec.get("overlay_residual_uri")
                    if isinstance(ov_uri, str) and ov_uri:
                        p = overlays_root / ov_uri
                        if p.exists():
                            src_overlay_residual = p
                            dst = _dst("overlay_residual", str(sha), int(shot), int(frame_idx))
                            shutil.copy2(p, dst)
                            copied_any = True

                if args.overlay_kind in ("fft", "both"):
                    ov_uri = rec.get("overlay_fft_uri")
                    if isinstance(ov_uri, str) and ov_uri:
                        p = overlays_root / ov_uri
                        if p.exists():
                            src_overlay_fft = p
                            dst = _dst("overlay_fft", str(sha), int(shot), int(frame_idx))
                            shutil.copy2(p, dst)
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
                    "select": args.select,
                    "value": float(v),
                    "threshold": thr,
                    "overlay_residual_src": str(src_overlay_residual) if src_overlay_residual else None,
                    "overlay_fft_src": str(src_overlay_fft) if src_overlay_fft else None,
                    "frame_src": str(src_frame) if src_frame else None,
                }) + "\n")
                copied += 1
                if args.limit and copied >= args.limit:
                    break

    direction = ">=" if args.select == "high" else "<="
    print(f"Selected {copied} frames with {args.metric} {direction} {thr:.6f} → {out_dir}")


if __name__ == "__main__":
    main()
