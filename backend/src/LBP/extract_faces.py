"""
src/extract_faces.py

Detect and normalize faces from sampled frames.
Each input frame (from frames.jsonl) is scanned for faces using MTCNN.
The largest detected face is cropped, resized, and saved as a normalized image.
Output is JSONL, one row per face, including bounding box and metadata.

Usage (PowerShell)
        python .\src\LBP\extract_faces.py \
                --frames .\data\derived\frames.jsonl \
                --frames-root .\data\derived\frames \
                --faces-root .\data\derived\faces \
                --out .\data\derived\faces.jsonl \
                --size 128

Notes
- frames.jsonl is written by sample_frames.py and contains asset_id, shot_index,
  frame_index, approx_t_ms, and frame URI.
- Each frame is read, face(s) are detected, and the largest bounding box is kept.
- The face is cropped, resized to the requested --size, and saved under faces-root.
- JSONL rows mirror frames.jsonl with additional face_uri, bbox, and detector fields.
- These cropped faces can be used as inputs for LBP, CNN feature extraction, etc.
"""

import argparse
import json
import os
from PIL import Image
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True, help="Path to frames.jsonl from sample_frames.py")
    parser.add_argument("--frames-root", required=True, help="Root directory of frame images")
    parser.add_argument("--faces-root", required=True, help="Root directory to save cropped faces")
    parser.add_argument("--out", required=True, help="Output JSONL file with detected faces")
    parser.add_argument("--size", type=int, default=128, help="Output face size (pixels, square)")
    args = parser.parse_args()

    # Initialize face detector
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(image_size=args.size, margin=20, device=device, post_process=True)

    with open(args.frames, "r") as f_in, open(args.out, "w") as f_out:
        for line in tqdm(f_in, desc="Extracting faces"):
            row = json.loads(line)
            frame_uri = row.get("uri")
            if not frame_uri:
                continue
            frame_path = os.path.join(args.frames_root, frame_uri.replace("frames/", "").lstrip("/\\"))
            if not os.path.exists(frame_path):
                continue

            try:
                img = Image.open(frame_path).convert("RGB")
                # detect face
                boxes, _ = mtcnn.detect(img)
                if boxes is None or len(boxes) == 0:
                    continue

                # choose largest face
                largest_box = max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
                x1, y1, x2, y2 = [int(v) for v in largest_box]
                face = img.crop((x1, y1, x2, y2)).resize((args.size, args.size))

                # save cropped face
                face_relpath = os.path.join(
                    row["sha256"], str(row["shot_index"]), f"{row['frame_index']:06d}.jpg"
                )
                face_path = os.path.join(args.faces_root, face_relpath)
                ensure_dir(os.path.dirname(face_path))
                face.save(face_path, format="JPEG", quality=95)

                # write JSONL row
                out_row = {
                    "asset_id": row.get("asset_id"),
                    "sha256": row.get("sha256"),
                    "shot_index": row.get("shot_index"),
                    "frame_index": row.get("frame_index"),
                    "approx_t_ms": row.get("approx_t_ms"),
                    "face_uri": os.path.join("faces", face_relpath).replace("\\", "/"),
                    "bbox": [x1, y1, x2, y2],
                    "detector": "MTCNN"
                }
                f_out.write(json.dumps(out_row) + "\n")

            except Exception as e:
                print(f"Error processing {frame_path}: {e}")

if __name__ == "__main__":
    main()
