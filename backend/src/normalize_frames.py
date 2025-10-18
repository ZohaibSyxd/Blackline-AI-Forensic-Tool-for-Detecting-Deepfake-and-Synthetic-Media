#!/usr/bin/env python3
"""
Step 5: Normalization & basic quality signals

What it does:
- Reads frames from sample_frames.py output
- Normalizes frames to standard size/colorspace (299x299, RGB)
- Computes quality metrics: blur, brightness, pHash
- Caches face crops for face-centric models (optional, robust Haar-based with tuning knobs)
- Outputs enhanced JSONL with quality metrics

Usage (PowerShell):
    python .\backend\src\normalize_frames.py `
        --frames .\backend\data\derived\frames.jsonl `
        --frames-root .\backend\data\derived\frames `
        --out .\backend\data\derived\frames_normalized.jsonl `
        --normalized-root .\backend\data\derived\normalized `
        --target-size 299 `
        --extract-faces `
        --min-face 48 `
        --det-scale 1.6 `
        --face-margin 0.30 `
        --max-faces 2 `
        --haar-scale-factor 1.06 `
        --haar-min-neighbors 3

Outputs:
  - Normalized frames: normalized/<sha>/<shot>/<frame>.jpg (299x299 RGB)
  - Face crops:      faces/<sha>/<shot>/<frame>_face_<idx>.jpg (optional)
  - Enhanced JSONL with quality metrics:
    {
      "asset_id": "...", "sha256": "...", "shot_index": 3, "frame_index": 12,
      "uri": "frames/<sha>/3/000012.jpg",
      "normalized_uri": "normalized/<sha>/3/000012.jpg",
      "face_uris": ["faces/<sha>/3/000012_face_0.jpg"],
      "quality_metrics": { "blur_score": 0.15, "brightness": 127.5, "phash": "a1b2..." },
      "normalization": { "original_size": [1920,1080], "normalized_size": [299,299], "colorspace": "RGB" },
      "tool_versions": { ... }
    }
"""
from __future__ import annotations

import argparse, json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import cv2
from PIL import Image, ImageOps
from PIL import __version__ as PIL_VERSION

# if your project has this; otherwise remove the audit wrapper lines below
from .audit import audit_step

# Optional imagehash (graceful fallback if not installed)
try:
    import imagehash
    from imagehash import phash as _phash
    IMAGEHASH_VERSION = getattr(imagehash, "__version__", None)
    HAS_IMAGEHASH = True
except Exception:
    IMAGEHASH_VERSION = None
    HAS_IMAGEHASH = False


# ------------------------ utilities ------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def compute_blur_score(image: np.ndarray) -> float:
    """Compute blur score using Laplacian variance (higher = sharper)."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.var(lap))


def compute_brightness(image: np.ndarray) -> float:
    """Compute average brightness in [0,255]."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    return float(np.mean(gray))


def compute_phash(image: Image.Image) -> Optional[str]:
    """Compute perceptual hash; returns None if imagehash is unavailable."""
    if not HAS_IMAGEHASH:
        return None
    return str(_phash(image))


def normalize_frame(image_path: Path, target_size: int = 299) -> Tuple[Image.Image, Dict[str, Any]]:
    """Normalize a single frame to target_size×target_size RGB, EXIF-aware."""
    img = Image.open(image_path)
    try:
        img = ImageOps.exif_transpose(img)  # fix orientation for phone footage
    except Exception:
        pass
    img = img.convert("RGB")
    original_size = img.size

    # Robust square crop+resize with high-quality resampling
    img = ImageOps.fit(img, (target_size, target_size), Image.Resampling.LANCZOS, centering=(0.5, 0.5))

    info = {
        "original_size": list(original_size),
        "normalized_size": list(img.size),
        "colorspace": img.mode,
    }
    return img, info


# ------------------------ stronger face detection ------------------------

@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int
    def as_xyxy(self):
        return self.x, self.y, self.x + self.w, self.y + self.h


def _iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a.as_xyxy()
    bx1, by1, bx2, by2 = b.as_xyxy()
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / max(1.0, area_a + area_b - inter)


def _nms(boxes: List[Box], iou_thresh: float = 0.4) -> List[Box]:
    if not boxes:
        return []
    # score by area, larger wins
    order = sorted(range(len(boxes)), key=lambda i: boxes[i].w * boxes[i].h, reverse=True)
    kept: List[Box] = []
    suppressed = [False] * len(boxes)
    for i in order:
        if suppressed[i]:
            continue
        bi = boxes[i]
        kept.append(bi)
        suppressed[i] = True
        for j in order:
            if suppressed[j] or j == i:
                continue
            if _iou(bi, boxes[j]) >= iou_thresh:
                suppressed[j] = True
    return kept


def _detect_faces_haar(
    gray: np.ndarray,
    scale_factor: float = 1.08,
    min_neighbors: int = 4,
    min_size: int = 64,
    include_profile: bool = True,
) -> List[Box]:
    """Haar frontal (and optional profile) face detection with simple NMS."""
    dets: List[Box] = []

    frontal_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    fr = cv2.CascadeClassifier(frontal_path)
    if not fr.empty():
        faces = fr.detectMultiScale(
            gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(min_size, min_size)
        )
        for (x, y, w, h) in faces:
            dets.append(Box(int(x), int(y), int(w), int(h)))

    if include_profile:
        profile_path = cv2.data.haarcascades + "haarcascade_profileface.xml"
        pr = cv2.CascadeClassifier(profile_path)
        if not pr.empty():
            faces = pr.detectMultiScale(
                gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(min_size, min_size)
            )
            for (x, y, w, h) in faces:
                dets.append(Box(int(x), int(y), int(w), int(h)))

    return _nms(dets, iou_thresh=0.4)


def extract_face_crops(
    image_rgb: np.ndarray,
    *,
    min_size: int = 64,
    det_scale: float = 1.5,
    scale_factor: float = 1.08,
    min_neighbors: int = 4,
    face_margin: float = 0.25,
    max_faces: int = 2,
    include_profile: bool = True,
) -> List[np.ndarray]:
    """
    Detect faces on an upscaled copy (det_scale) to boost recall.
    Crop with padding (face_margin fraction of max(w,h)).
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # upscale only for detection (helps tiny faces)
    if det_scale != 1.0:
        gray_det = cv2.resize(gray, None, fx=det_scale, fy=det_scale, interpolation=cv2.INTER_LINEAR)
    else:
        gray_det = gray

    dets = _detect_faces_haar(
        gray_det,
        scale_factor=scale_factor,
        min_neighbors=min_neighbors,
        min_size=int(min_size * det_scale),
        include_profile=include_profile,
    )

    H, W = gray.shape[:2]
    crops: List[np.ndarray] = []
    for b in dets[:max_faces]:
        # map back to original coordinates
        x = int(round(b.x / det_scale))
        y = int(round(b.y / det_scale))
        w = int(round(b.w / det_scale))
        h = int(round(b.h / det_scale))

        pad = int(face_margin * max(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(W, x + w + pad)
        y2 = min(H, y + h + pad)
        if (x2 - x1) <= 1 or (y2 - y1) <= 1:
            continue
        crops.append(image_rgb[y1:y2, x1:x2])

    return crops


# ------------------------ per-frame processing ------------------------

def process_single_frame(
    frame_data: Dict[str, Any],
    *,
    frames_root: Path,
    normalized_root: Path,
    target_size: int,
    extract_faces: bool,
    faces_root: Optional[Path],
    # face detection knobs
    min_face: int,
    det_scale: float,
    haar_scale_factor: float,
    haar_min_neighbors: int,
    face_margin: float,
    max_faces: int,
    frontal_only: bool,
) -> Optional[Dict[str, Any]]:
    """Process one frame: normalize, metrics, optional face crops; return enhanced row."""
    uri = frame_data.get("uri")
    if not uri:
        return None

    # Resolve on-disk path (supports "frames/..." or raw relative path)
    try:
        rel = Path(uri).relative_to("frames")
    except ValueError:
        rel = Path(uri)
    src_path = frames_root / rel
    if not src_path.exists():
        return None

    try:
        # Normalize
        pil_img, norm_info = normalize_frame(src_path, target_size)
        img_array = np.array(pil_img)  # RGB np.ndarray

        # Metrics
        blur_score = compute_blur_score(img_array)
        brightness = compute_brightness(img_array)
        phash_str = compute_phash(pil_img)

        # Save normalized image
        sha = frame_data.get("sha256")
        shot_idx = frame_data.get("shot_index")
        frame_idx = frame_data.get("frame_index")
        if not (sha and shot_idx is not None and frame_idx is not None):
            return None

        norm_dir = normalized_root / str(sha) / str(shot_idx)
        ensure_dir(norm_dir)
        norm_path = norm_dir / f"{int(frame_idx):06d}.jpg"
        pil_img.save(norm_path, "JPEG", quality=95)

        # Faces
        face_uris: List[str] = []
        if extract_faces and faces_root is not None:
            face_crops = extract_face_crops(
                img_array,
                min_size=min_face,
                det_scale=det_scale,
                scale_factor=haar_scale_factor,
                min_neighbors=haar_min_neighbors,
                face_margin=face_margin,
                max_faces=max_faces,
                include_profile=(not frontal_only),
            )
            faces_dir = faces_root / str(sha) / str(shot_idx)
            if face_crops:
                ensure_dir(faces_dir)
            for face_idx, crop in enumerate(face_crops):
                face_pil = Image.fromarray(crop)
                face_path = faces_dir / f"{int(frame_idx):06d}_face_{face_idx}.jpg"
                face_pil.save(face_path, "JPEG", quality=95)
                # Make URI relative to derived root (…/derived/<faces/...>)
                face_uris.append(str(face_path.relative_to(faces_root.parent).as_posix()))

        # Build enhanced row
        enhanced = frame_data.copy()
        enhanced.update({
            "normalized_uri": str(norm_path.relative_to(normalized_root.parent).as_posix()),
            "face_uris": face_uris,
            "quality_metrics": {
                "blur_score": blur_score,
                "brightness": brightness,
                "phash": phash_str,
            },
            "normalization": norm_info,
            "tool_versions": {
                "opencv": getattr(cv2, "__version__", None),
                "pillow": PIL_VERSION,
                "imagehash": IMAGEHASH_VERSION,
            },
        })
        return enhanced

    except Exception as e:
        print(f"Error processing frame {uri}: {e}")
        return None


# ------------------------ CLI & driver ------------------------

def main():
    ap = argparse.ArgumentParser(description="Step 5: Normalize frames and compute quality metrics")
    ap.add_argument("--frames", default="backend/data/derived/frames.jsonl", help="Input frames JSONL")
    ap.add_argument("--frames-root", default="backend/data/derived/frames", help="Root folder containing frames")
    ap.add_argument("--out", default="backend/data/derived/frames_normalized.jsonl", help="Output JSONL")
    ap.add_argument("--normalized-root", default="backend/data/derived/normalized", help="Folder for normalized frames")
    ap.add_argument("--target-size", type=int, default=299, help="Target size (square)")
    ap.add_argument("--extract-faces", action="store_true", help="Extract and save face crops")
    ap.add_argument("--faces-root", default="backend/data/derived/faces", help="Folder for face crops")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N frames")

    # New: detector knobs
    ap.add_argument("--min-face", type=int, default=64, help="Minimum face box (pixels) in detection space")
    ap.add_argument("--det-scale", type=float, default=1.5, help="Upscale factor for detection (1.0 = none)")
    ap.add_argument("--face-margin", type=float, default=0.25, help="Crop margin as fraction of max(w,h)")
    ap.add_argument("--max-faces", type=int, default=2, help="Max faces to save per frame")
    ap.add_argument("--haar-scale-factor", type=float, default=1.08, help="Haar scaleFactor (smaller => denser)")
    ap.add_argument("--haar-min-neighbors", type=int, default=4, help="Haar minNeighbors (smaller => more)")
    ap.add_argument("--frontal-only", action="store_true", help="Disable profile-face detector (frontal only)")

    args = ap.parse_args()

    frames_root = Path(args.frames_root)
    normalized_root = Path(args.normalized_root)
    faces_root = Path(args.faces_root) if args.extract_faces else None

    ensure_dir(normalized_root)
    if args.extract_faces and faces_root:
        ensure_dir(faces_root)

    processed = 0
    with audit_step("normalize_frames", params=vars(args), inputs={"frames": args.frames}) as outputs:
        with open(args.out, "w", encoding="utf-8") as fw, open(args.frames, encoding="utf-8") as fr:
            for line in fr:
                if not line.strip():
                    continue
                try:
                    frame_data = json.loads(line)
                except Exception:
                    continue

                enhanced = process_single_frame(
                    frame_data,
                    frames_root=frames_root,
                    normalized_root=normalized_root,
                    target_size=args.target_size,
                    extract_faces=args.extract_faces,
                    faces_root=faces_root,
                    min_face=args.min_face,
                    det_scale=args.det_scale,
                    haar_scale_factor=args.haar_scale_factor,
                    haar_min_neighbors=args.haar_min_neighbors,
                    face_margin=args.face_margin,
                    max_faces=args.max_faces,
                    frontal_only=args.frontal_only,
                )

                if enhanced:
                    fw.write(json.dumps(enhanced) + "\n")
                    processed += 1
                    if processed % 100 == 0:
                        print(f"Processed {processed} frames...")

                if args.limit and processed >= args.limit:
                    break

        outputs["frames_normalized"] = {"path": args.out}

    print(f"Normalization complete: processed {processed} frames → {args.out}")


if __name__ == "__main__":
    main()
