"""
lbp_heatmap.py
----------------------------------------------------------
Batch forensic LBP analysis for multiple videos.

For each video:
  - Detects faces (MTCNN)
  - Computes LBP texture anomaly heatmaps (skin-filtered)
  - Saves only top-K most suspicious frames

Usage:
    python .\backend\src\LBP\lbp_heatmap.py test --top_k 5 --out .\backend\data\derived\face_heatmaps
----------------------------------------------------------
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from skimage.feature import local_binary_pattern
from facenet_pytorch import MTCNN
import argparse
import pandas as pd

# ---------------- CONFIG ----------------
BLOCK_SIZE = 32
LBP_SCALES = [(8,1), (16,2), (24,3)]
LBP_METHOD = "uniform"
FRAME_SAMPLE_RATE = 15
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
mtcnn = MTCNN(
    image_size=224, margin=0, post_process=True,
    selection_method="largest",
    device="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
)
# ----------------------------------------

def chi2_distance(histA, histB, eps=1e-10):
    return 0.5 * np.sum(((histA - histB) ** 2) / (histA + histB + eps))

def preprocess_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = CLAHE.apply(gray)
    # emphasize residual detail
    highpass = cv2.Laplacian(gray, cv2.CV_64F)
    highpass = cv2.convertScaleAbs(highpass)
    return highpass

def compute_multiscale_lbp(gray):
    lbp_maps = [local_binary_pattern(gray, P, R, LBP_METHOD) for P, R in LBP_SCALES]
    return np.stack(lbp_maps, axis=-1)

def block_histogram(lbp_block, n_bins):
    hists = []
    for i in range(lbp_block.shape[-1]):
        lbp = lbp_block[..., i]
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-9)
        hists.append(hist)
    return np.concatenate(hists)

def compute_texture_anomaly_map(gray):
    h, w = gray.shape
    lbp_maps = compute_multiscale_lbp(gray)
    n_bins = LBP_SCALES[-1][0] + 2
    H, W = h // BLOCK_SIZE, w // BLOCK_SIZE

    if H == 0 or W == 0:
        return np.zeros_like(gray, dtype=float)

    features = np.zeros((H, W, len(LBP_SCALES) * n_bins))
    for i in range(H):
        for j in range(W):
            y1, y2 = i*BLOCK_SIZE, (i+1)*BLOCK_SIZE
            x1, x2 = j*BLOCK_SIZE, (j+1)*BLOCK_SIZE
            block = lbp_maps[y1:y2, x1:x2, :]
            features[i, j] = block_histogram(block, n_bins)

    anomaly = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            hist = features[i, j]
            dists = []
            for (di, dj) in [(1,0), (-1,0), (0,1), (0,-1)]:
                ni, nj = i+di, j+dj
                if 0 <= ni < H and 0 <= nj < W:
                    d = chi2_distance(hist, features[ni, nj])
                    dists.append(d)
            if dists:
                anomaly[i, j] = np.mean(dists)

    anomaly = (anomaly - anomaly.min()) / (anomaly.max() + 1e-9)
    heatmap = cv2.resize(anomaly, (w, h), interpolation=cv2.INTER_CUBIC)
    return heatmap

# ---------------- SKIN MASK ----------------
def skin_mask_bgr(face_bgr):
    hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 40, 60], dtype=np.uint8)
    upper1 = np.array([20, 150, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower1, upper1)
    lower2 = np.array([170, 40, 60], dtype=np.uint8)
    upper2 = np.array([180, 150, 255], dtype=np.uint8)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.GaussianBlur(mask, (7,7), 0)
    return mask.astype(float) / 255.0
# -------------------------------------------

def overlay_heatmap(face, heatmap, alpha=0.6):
    heat_color = cv2.applyColorMap((heatmap*255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(face, 1-alpha, heat_color, alpha, 0)
    return overlay

def analyze_video(video_path, out_dir, top_k=5):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot open {video_path}")
        return None

    frame_idx = 0
    frame_results = []

    for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), desc=f"Analyzing {video_path.name}"):
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % FRAME_SAMPLE_RATE != 0:
            continue

        boxes, _ = mtcnn.detect(frame)
        if boxes is None:
            continue

        for f_idx, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(b) for b in box]
            x1, y1 = max(0, x1), max(0, y1)
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            gray = preprocess_gray(face)
            heatmap = compute_texture_anomaly_map(gray)
            mask = skin_mask_bgr(face)
            heatmap *= mask
            heatmap = cv2.GaussianBlur(heatmap, (7,7), 0)

            suspicion = float(np.mean(heatmap))
            frame_results.append({
                "frame": frame_idx,
                "face_idx": f_idx + 1,
                "score": suspicion,
                "face": face,
                "heatmap": heatmap
            })

    cap.release()
    if not frame_results:
        print(f"No faces detected in {video_path.name}")
        return None

    # Sort by score and keep top_k
    frame_results.sort(key=lambda x: x["score"], reverse=True)
    top_frames = frame_results[:top_k]

    # --- per-video output directory ---
    video_out = Path(out_dir) / video_path.stem
    video_out.mkdir(parents=True, exist_ok=True)

    for item in top_frames:
        overlay = overlay_heatmap(item["face"], item["heatmap"])
        out_name = f"frame{item['frame']:05d}_score{round(item['score'],3)}.jpg"
        cv2.imwrite(str(video_out / out_name), overlay)

    suspicion_mean = np.mean([r["score"] for r in frame_results])
    print(f"{video_path.name}: avg anomaly {suspicion_mean:.3f} | saved top {len(top_frames)} frames")
    return suspicion_mean

def process_folder(input_folder, top_k=5, out="forensic_faces"):
    input_folder = Path(input_folder)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    summary = []
    videos = list(input_folder.glob("*.mp4")) + list(input_folder.glob("*.mov")) + list(input_folder.glob("*.avi"))
    if not videos:
        print("No video files found in the folder.")
        return

    for video in videos:
        score = analyze_video(video, out, top_k=top_k)
        if score is not None:
            summary.append({"video": video.name, "avg_anomaly": score})

    if summary:
        df = pd.DataFrame(summary)
        print("\nSummary of average anomaly scores:")
        print(df.to_string(index=False))
        df.to_csv(out / "summary.csv", index=False)
        print(f"Saved summary to {out/'summary.csv'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to video folder")
    parser.add_argument("--top_k", type=int, default=5, help="Number of most anomalous frames to save")
    parser.add_argument("--out", type=str, default="forensic_faces", help="Output directory for results")
    args = parser.parse_args()

    process_folder(args.input_path, top_k=args.top_k, out=args.out)
