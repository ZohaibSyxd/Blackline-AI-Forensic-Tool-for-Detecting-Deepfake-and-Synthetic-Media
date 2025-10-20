"""
Noise Residual and Frequency Analysis over sampled frames.

Inputs:
  - frames.jsonl (from sample_frames.py) with rows:
      sha256, shot_index, frame_index, uri (relative to frames root)
  - Frames root directory containing JPEGs under frames/<sha>/<shot>/<nnnnnn>.jpg

Outputs:
  - Overlays:
      overlays/noise/<sha>/<shot>/<frame>.jpg      (visual residual map)
      overlays/noise_fft/<sha>/<shot>/<frame>.jpg  (log-magnitude FFT spectrum)
  - Metrics JSONL: frames_noise.jsonl with per-frame metrics:
      {
        asset_id, sha256, shot_index, frame_index, uri,
        residual_abs_mean, residual_std, residual_energy,
        fft_low_ratio, fft_high_ratio
      }

Notes:
  - Residual can be computed via Gaussian subtraction, median subtraction, or NLM (fastNlMeans) denoising residuals.
  - FFT metrics are simple low/high frequency energy ratios, which can indicate over-smoothing or high-frequency artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import cv2
from PIL import Image
from ...audit import audit_step


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_gray(path: Path) -> np.ndarray:
    # Load RGB using PIL for consistency, then to grayscale float32
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return gray


def compute_residual(gray: np.ndarray, method: str, gaussian_sigma: float, median_ksize: int, nlm_h: float) -> np.ndarray:
    # Return residual in float32 (can be negative)
    if method == "gaussian":
        # Subtract a Gaussian-smoothed version (high-pass)
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=gaussian_sigma, sigmaY=gaussian_sigma)
        residual = gray - blurred
    elif method == "median":
        k = max(3, median_ksize | 1)  # odd
        blurred = cv2.medianBlur(gray.astype(np.uint8), k)
        residual = gray - blurred.astype(np.float32)
    elif method == "nlm":
        base_u8 = gray.astype(np.uint8)
        den = cv2.fastNlMeansDenoising(base_u8, None, h=float(nlm_h), templateWindowSize=7, searchWindowSize=21)
        residual = base_u8.astype(np.float32) - den.astype(np.float32)
    else:
        # default to gaussian
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=gaussian_sigma, sigmaY=gaussian_sigma)
        residual = gray - blurred
    return residual


def residual_vis(residual: np.ndarray) -> np.ndarray:
    # Visualization: absolute residual -> normalize -> colorize (optional)
    abs_res = np.abs(residual)
    if abs_res.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    # Normalize to 0-255
    vmin, vmax = np.percentile(abs_res, [1, 99])
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = np.clip((abs_res - vmin) / (vmax - vmin), 0.0, 1.0)
    vis = (norm * 255.0).astype(np.uint8)
    # Optional: apply a colormap for easier inspection
    vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
    return vis_color


def fft_spectrum(residual: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    # Remove DC component
    x = residual - float(np.mean(residual))
    # Compute 2D FFT and shift zero frequency to center
    F = np.fft.fft2(x)
    F = np.fft.fftshift(F)
    mag = np.abs(F)
    # Visualization: log magnitude
    logmag = np.log1p(mag)
    if logmag.size == 0:
        spec_vis = np.zeros((1, 1), dtype=np.uint8)
        metrics = {"fft_low_ratio": 0.0, "fft_high_ratio": 0.0}
        return spec_vis, metrics
    # Normalize log-mag to 0..255 for viewing
    vmin, vmax = np.percentile(logmag, [5, 99.5])
    if vmax <= vmin:
        vmax = vmin + 1.0
    spec = np.clip((logmag - vmin) / (vmax - vmin), 0.0, 1.0)
    spec_vis = (spec * 255.0).astype(np.uint8)
    spec_vis = cv2.applyColorMap(spec_vis, cv2.COLORMAP_VIRIDIS)

    # Low/High frequency energy ratios
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = np.sqrt((max(cy, h - cy - 1)) ** 2 + (max(cx, w - cx - 1)) ** 2)
    cutoff = 0.25 * r_max  # inner quarter as "low" band
    low_mask = r <= cutoff
    high_mask = r > cutoff
    # Energy ~ sum of squared magnitude (Parseval related)
    total = np.maximum(np.sum(mag ** 2), 1e-9)
    low_energy = float(np.sum((mag ** 2)[low_mask])) / total
    high_energy = float(np.sum((mag ** 2)[high_mask])) / total
    metrics = {
        "fft_low_ratio": low_energy,
        "fft_high_ratio": high_energy,
    }
    return spec_vis, metrics


def residual_metrics(residual: np.ndarray) -> Dict[str, float]:
    abs_mean = float(np.mean(np.abs(residual)))
    std = float(np.std(residual))
    energy = float(np.mean(residual ** 2))  # per-pixel energy
    return {
        "residual_abs_mean": abs_mean,
        "residual_std": std,
        "residual_energy": energy,
    }


def _to_py_scalar(v: Any) -> Any:
    """Convert numpy scalar types to native Python types for JSON serialization."""
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def _sanitize_for_json(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _to_py_scalar(v) for k, v in d.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute Noise Residual and FFT overlays + metrics.")
    ap.add_argument("--frames", default="backend/data/derived/frames.jsonl", help="Path to frames JSONL")
    ap.add_argument("--root", default="backend/data/derived/frames", help="Root folder containing frames")
    ap.add_argument("--out", default="backend/data/derived/frames_noise.jsonl", help="Output metrics JSONL")
    ap.add_argument("--overlays", default="backend/data/derived/overlays", help="Base output dir for overlays")
    ap.add_argument("--method", choices=["gaussian", "median", "nlm"], default="gaussian", help="Residual method")
    ap.add_argument("--gaussian-sigma", type=float, default=2.0, help="Sigma for Gaussian blur (if method=gaussian)")
    ap.add_argument("--median-ksize", type=int, default=5, help="Kernel size for median blur (if method=median)")
    ap.add_argument("--nlm-h", type=float, default=10.0, help="Filter strength for NLM residual (if method=nlm)")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on frames processed")
    ap.add_argument("--verbose", action="store_true", help="Print debug info for skipped/processed frames")
    ap.add_argument("--debug-limit", type=int, default=10, help="Max number of verbose skip logs to print")
    args = ap.parse_args()

    frames_root = Path(args.root)
    overlays_noise_root = Path(args.overlays) / "noise"
    overlays_fft_root = Path(args.overlays) / "noise_fft"
    ensure_dir(overlays_noise_root)
    ensure_dir(overlays_fft_root)

    processed = 0
    total = 0
    parse_errors = 0
    missing_fields = 0
    missing_src = 0
    exceptions = 0
    vcount = 0
    with audit_step("compute_noise", params=vars(args), inputs={"frames": args.frames}) as outputs:
        with open(args.out, "w", encoding="utf-8") as fw, open(args.frames, encoding="utf-8") as fr:
            for line in fr:
                if not line.strip():
                    continue
                try:
                    row: Dict[str, Any] = json.loads(line)
                except Exception:
                    parse_errors += 1
                    if args.verbose and vcount < args.debug_limit:
                        print("[skip] parse error for line")
                        vcount += 1
                    continue

                uri = row.get("uri")
                sha = row.get("sha256")
                shot_idx = row.get("shot_index")
                frame_idx = row.get("frame_index")
                if not uri or sha is None or shot_idx is None or frame_idx is None:
                    missing_fields += 1
                    if args.verbose and vcount < args.debug_limit:
                        print("[skip] missing required fields in row:", row)
                        vcount += 1
                    continue

                src_path = frames_root / Path(uri).relative_to("frames")  # uri starts with frames/
                if not src_path.exists():
                    missing_src += 1
                    if args.verbose and vcount < args.debug_limit:
                        print(f"[skip] src not found: {src_path}")
                        vcount += 1
                    continue

                try:
                    gray = load_gray(src_path)
                    res = compute_residual(gray, method=args.method, gaussian_sigma=args.gaussian_sigma,
                                           median_ksize=args.median_ksize, nlm_h=args.nlm_h)
                    res_vis = residual_vis(res)
                    spec_vis, fft_metrics = fft_spectrum(res)
                    m = residual_metrics(res)

                    # Save overlays
                    out_dir_res = overlays_noise_root / str(sha) / str(shot_idx)
                    out_dir_fft = overlays_fft_root / str(sha) / str(shot_idx)
                    ensure_dir(out_dir_res)
                    ensure_dir(out_dir_fft)
                    out_path_res = out_dir_res / f"{int(frame_idx):06d}.jpg"
                    out_path_fft = out_dir_fft / f"{int(frame_idx):06d}.jpg"
                    cv2.imwrite(str(out_path_res), res_vis)
                    cv2.imwrite(str(out_path_fft), spec_vis)

                    out_row = {
                        "asset_id": row.get("asset_id"),
                        "sha256": sha,
                        "shot_index": shot_idx,
                        "frame_index": frame_idx,
                        "uri": uri,
                        "overlay_residual_uri": str(out_path_res.relative_to(overlays_noise_root.parent).as_posix()),
                        "overlay_fft_uri": str(out_path_fft.relative_to(overlays_fft_root.parent).as_posix()),
                        **m,
                        **fft_metrics,
                    }
                    out_row = _sanitize_for_json(out_row)
                    fw.write(json.dumps(out_row) + "\n")
                    processed += 1
                except Exception:
                    # Skip corrupted images or processing failures
                    exceptions += 1
                    if args.verbose and vcount < args.debug_limit:
                        import traceback
                        print(f"[skip] exception processing {src_path}:")
                        traceback.print_exc(limit=1)
                        vcount += 1
                    continue

                if args.limit and processed >= args.limit:
                    break
                total += 1

        outputs["frames_noise"] = {"path": args.out}
    print(f"Noise analysis processed {processed} frames → {args.out}")
    print(f"Summary: parsed_ok={processed + missing_src + missing_fields + exceptions} parse_errors={parse_errors} missing_fields={missing_fields} missing_src={missing_src} exceptions={exceptions}")


if __name__ == "__main__":
    main()
