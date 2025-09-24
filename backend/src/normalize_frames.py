"""
Step 5: Normalization & basic quality signals

What it does:
- Reads frames from sample_frames.py output
- Normalizes frames to standard size/colorspace (224x224, RGB)
- Computes quality metrics: blur, brightness, pHash
- Caches face crops for face-centric models (optional)
- Outputs enhanced JSONL with quality metrics

Usage (PowerShell):
    python .\backend\src\normalize_frames.py `
        --frames .\backend\data\derived\frames.jsonl `
        --frames-root .\backend\data\derived\frames `
        --out .\backend\data\derived\frames_normalized.jsonl `
        --normalized-root .\backend\data\derived\normalized `
        --target-size 224 `
        --extract-faces

Outputs:
  - Normalized frames: normalized/<sha>/<shot>/<frame>.jpg (224x224 RGB)
  - Face crops: faces/<sha>/<shot>/<frame>_face_<idx>.jpg (optional)
  - Enhanced JSONL with quality metrics:
    {
      "asset_id": "...", "sha256": "...", "shot_index": 3, "frame_index": 12,
      "uri": "frames/<sha>/3/000012.jpg",
      "normalized_uri": "normalized/<sha>/3/000012.jpg",
      "face_uris": ["faces/<sha>/3/000012_face_0.jpg"],
      "quality_metrics": {
        "blur_score": 0.15,
        "brightness": 127.5,
        "phash": "a1b2c3d4e5f6"
      },
      "normalization": {
        "original_size": [1920, 1080],
        "normalized_size": [224, 224],
        "colorspace": "RGB"
      }
    }
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import cv2
from PIL import Image, ImageOps
from PIL import __version__ as PIL_VERSION

# Optional imagehash (graceful fallback if not installed)
try:
    import imagehash
    from imagehash import phash as _phash
    IMAGEHASH_VERSION = getattr(imagehash, "__version__", None)
    HAS_IMAGEHASH = True
except Exception:
    IMAGEHASH_VERSION = None
    HAS_IMAGEHASH = False


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


def normalize_frame(image_path: Path, target_size: int = 224) -> Tuple[Image.Image, Dict[str, Any]]:
    """Normalize a single frame to target_size×target_size RGB, EXIF-aware."""
    img = Image.open(image_path)
    # Respect EXIF orientation (common for phone footage)
    try:
        img = ImageOps.exif_transpose(img)
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


def extract_face_crops(image: np.ndarray, min_size: int = 64) -> List[np.ndarray]:
    """Extract face crops from an RGB numpy array using Haar cascades."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        return []

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(min_size, min_size),
    )

    crops: List[np.ndarray] = []
    for (x, y, w, h) in faces:
        pad = int(0.1 * max(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + w + pad)
        y2 = min(image.shape[0], y + h + pad)
        crops.append(image[y1:y2, x1:x2])
    return crops


def process_single_frame(
    frame_data: Dict[str, Any],
    frames_root: Path,
    normalized_root: Path,
    target_size: int = 224,
    extract_faces: bool = False,
    faces_root: Optional[Path] = None,
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
        img_array = np.array(pil_img)

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

        # Faces (always include field; empty list if none/not requested)
        face_uris: List[str] = []
        if extract_faces and faces_root is not None:
            face_crops = extract_face_crops(img_array)
            faces_dir = faces_root / str(sha) / str(shot_idx)
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


def main():
    ap = argparse.ArgumentParser(description="Step 5: Normalize frames and compute quality metrics")
    ap.add_argument("--frames", default="backend/data/derived/frames.jsonl", help="Input frames JSONL")
    ap.add_argument("--frames-root", default="backend/data/derived/frames", help="Root folder containing frames")
    ap.add_argument("--out", default="backend/data/derived/frames_normalized.jsonl", help="Output JSONL")
    ap.add_argument("--normalized-root", default="backend/data/derived/normalized", help="Folder for normalized frames")
    ap.add_argument("--target-size", type=int, default=224, help="Target size (square)")
    ap.add_argument("--extract-faces", action="store_true", help="Extract and save face crops")
    ap.add_argument("--faces-root", default="backend/data/derived/faces", help="Folder for face crops")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N frames")
    args = ap.parse_args()

    frames_root = Path(args.frames_root)
    normalized_root = Path(args.normalized_root)
    faces_root = Path(args.faces_root) if args.extract_faces else None

    ensure_dir(normalized_root)
    if args.extract_faces and faces_root:
        ensure_dir(faces_root)

    processed = 0
    with open(args.out, "w", encoding="utf-8") as fw, open(args.frames, encoding="utf-8") as fr:
        for line in fr:
            if not line.strip():
                continue
            try:
                frame_data = json.loads(line)
            except Exception:
                continue

            enhanced = process_single_frame(
                frame_data=frame_data,
                frames_root=frames_root,
                normalized_root=normalized_root,
                target_size=args.target_size,
                extract_faces=args.extract_faces,
                faces_root=faces_root,
            )

            if enhanced:
                fw.write(json.dumps(enhanced) + "\n")
                processed += 1
                if processed % 100 == 0:
                    print(f"Processed {processed} frames...")

            if args.limit and processed >= args.limit:
                break

    print(f"Normalization complete: processed {processed} frames → {args.out}")


if __name__ == "__main__":
    main()
