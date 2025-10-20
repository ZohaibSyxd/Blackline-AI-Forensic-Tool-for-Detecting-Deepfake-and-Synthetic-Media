# src/temporal_inconsistency.py
"""
Temporal Inconsistency Analysis (Stage 7)

Purpose
-------
Analyze short per-shot clips to extract *temporal* facial signals:
- Blink patterns via Eye-Aspect-Ratio (EAR)
- Head pose stability via solvePnP (pitch, yaw, roll)
- Landmark motion smoothness ("jitter")

We then summarize each clip into a single JSONL row so later stages
can aggregate per-asset stats (e.g., in analyze_metadata.py).

Inputs
------
1) Clip files, produced by sample_frames.py when run with --extract-clips:
   backend/data/derived/clips/<sha>/shot_<idx>.mp4

2) OR a JSONL manifest of clips (recommended):
   backend/data/derived/clips.jsonl  with rows like:
   {
     "asset_id": "...",
     "sha256": "...",
     "shot_index": 3,
     "clip_uri": "clips/<sha>/shot_003.mp4"
   }

Outputs
-------
backend/data/derived/clips_temporal.jsonl  (one row per clip)
Example row:
{
  "asset_id": "...",
  "sha256": "...",
  "shot_index": 3,
  "clip_uri": "clips/<sha>/shot_003.mp4",
  "temporal_blink_score": 0.85,        # 0-1; higher = more regular blink intervals
  "temporal_pose_stability": 0.92,     # 0-1; higher = lower angular variance
  "temporal_landmark_jitter": 0.012,   # normalized by image diagonal; lower = smoother
  "blink_rate": 0.28,                  # blinks per second
  "face_detected_ratio": 0.95,         # fraction of processed frames with a face
  "frames_processed": 123,
  "duration_s": 4.1,
  "tool_versions": {"mediapipe": "...", "opencv": "..."}
}

Key Ideas
---------
• EAR-based blinks: if EAR < threshold for N consecutive frames → count one blink.
• Pose stability: compute per-axis std dev of (pitch, yaw, roll) across frames,
  then normalize (lower std = more stable).
• Landmark jitter: mean inter-frame landmark displacement, normalized by image
  diagonal so that values are comparable across resolutions.

Dependencies
------------
- mediapipe ~= 0.10
- opencv-python
"""

from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2
import mediapipe as mp
import numpy as np


# MediaPipe FaceMesh landmark indices for EAR.
# The six points per eye are chosen so that p1 and p4 are the horizontal corners.
# Format for each eye: [p1, p2, p3, p4, p5, p6]
LEFT_EYE_INDICES  = [263, 385, 387, 362, 373, 380]  # p1 outer corner, p4 inner corner
RIGHT_EYE_INDICES = [ 33, 160, 158, 133, 153, 144]  # p1 inner corner, p4 outer corner


def eye_aspect_ratio(landmarks_xy: np.ndarray,
                     left_eye_indices: List[int],
                     right_eye_indices: List[int]) -> float:
    """
    Compute average Eye Aspect Ratio (EAR) for both eyes.

    EAR definition:
        (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Intuition:
        Smaller EAR → eyes more closed. Blinks appear as short
        dips below a threshold.

    Args:
        landmarks_xy: (468, 2) array of (x, y) landmark coords in pixels.
        left_eye_indices/right_eye_indices: landmark indices for each eye.

    Returns:
        Average EAR across both eyes.
    """
    def ear(pts: np.ndarray) -> float:
        p1, p2, p3, p4, p5, p6 = pts
        denom = 2.0 * np.linalg.norm(p1 - p4)
        if denom <= 1e-6:
            return 0.0
        return (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / denom

    L = landmarks_xy[left_eye_indices]
    R = landmarks_xy[right_eye_indices]
    return 0.5 * (ear(L) + ear(R))


def estimate_head_pose(landmarks_xy: np.ndarray,
                       image_shape: Tuple[int, int]) -> Tuple[float, float, float]:
    """
    Estimate head pose angles (pitch, yaw, roll) using solvePnP with a tiny 3D face model.

    We pick a minimal set of stable facial points (nose tip, eye inner corners,
    mouth corners) and fit a simple planar 3D template to recover orientation.

    Returns:
        (pitch, yaw, roll) in degrees. Returns (0,0,0) if solvePnP fails.
    """
    h, w = image_shape

    # 2D points from detected landmarks
    k2d = np.array([
        landmarks_xy[  1],  # nose tip
        landmarks_xy[ 33],  # left eye inner corner
        landmarks_xy[263],  # right eye inner corner
        landmarks_xy[ 61],  # left mouth corner
        landmarks_xy[291],  # right mouth corner
    ], dtype=np.float32)

    # Coarse 3D template (units are arbitrary but roughly proportional)
    k3d = np.array([
        [   0.0,   0.0,   0.0],  # nose
        [ -50.0,  50.0,   0.0],  # left eye
        [  50.0,  50.0,   0.0],  # right eye
        [ -30.0, -50.0,   0.0],  # left mouth
        [  30.0, -50.0,   0.0],  # right mouth
    ], dtype=np.float32)

    # Pin-hole intrinsics; focal length ~ image width is a good heuristic
    f = float(w)
    cx, cy = w * 0.5, h * 0.5
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
    dist = np.zeros((4, 1), dtype=np.float32)  # assume no distortion

    ok, rvec, _tvec = cv2.solvePnP(k3d, k2d, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0

    R, _ = cv2.Rodrigues(rvec)
    # OpenCV's RQDecomp3x3 returns angles in degrees
    pitch, yaw, roll = cv2.RQDecomp3x3(R)[0]
    return float(pitch), float(yaw), float(roll)


def process_clip_temporal(
    clip_path: Path,
    *,
    ear_threshold: float = 0.25,          # Lower → more sensitive blinks, Higher → fewer false positives
    consecutive_frames: int = 2,          # Require N consecutive frames below threshold to count a blink
    mediapipe_complexity: int = 0,        # 0=fast, 1=accurate, 2=very accurate (slower)
    min_detection_confidence: float = 0.5,
    skip_frames: int = 1,                 # Process every Nth frame for speed (1 = no skipping)
) -> Dict[str, Any]:
    """
    Process a single per-shot clip and compute temporal metrics.

    Notes on robustness:
    • We gracefully handle MediaPipe versions with/without `model_complexity`.
    • Jitter is computed only across *consecutive frames with detected faces*
      to avoid spikes from placeholder zeros.
    """
    if not clip_path.exists():
        return {"error": "clip not found"}

    # ---- Face Mesh init (compat across mediapipe versions) ----
    mp_face_mesh = mp.solutions.face_mesh
    try:
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            model_complexity=mediapipe_complexity,   # may not exist in some versions
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
            static_image_mode=False,
        )
    except TypeError:
        # Fallback for older/newer builds lacking `model_complexity`
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
            static_image_mode=False,
        )

    # ---- Open video ----
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        face_mesh.close()
        return {"error": "cannot open video"}

    # fps may be 0 or NaN on some malformed files; default to 30
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = (frame_count / fps) if fps > 0 else 0.0

    # Per-frame collections
    blink_states: List[bool] = []                  # EAR<threshold? (raw)
    blink_events: List[int] = []                   # frames where a *completed* blink is registered
    pose_angles: List[Tuple[float, float, float]] = []
    landmark_positions: List[np.ndarray] = []      # (468, 2) per frame (or zeros if face missing)
    face_detected: List[bool] = []

    frame_idx = 0
    blink_run = 0                                  # number of consecutive frames below EAR threshold
    last_h, last_w = None, None

    # ---- Frame loop ----
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if (frame_idx % max(1, skip_frames)) != 0:
            frame_idx += 1
            continue

        h, w = frame.shape[:2]
        last_h, last_w = h, w

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0]
            # Convert normalized [0,1] coords → pixel coordinates
            pts = np.array([(p.x * w, p.y * h) for p in lm.landmark], dtype=np.float32)

            # --- Blink detection (EAR) ---
            ear = eye_aspect_ratio(pts, LEFT_EYE_INDICES, RIGHT_EYE_INDICES)
            is_blink_frame = bool(ear < ear_threshold)  # raw classification this frame
            blink_states.append(is_blink_frame)

            # Debounce: count blink only after N consecutive low-EAR frames
            if is_blink_frame:
                blink_run += 1
            elif blink_run >= consecutive_frames:
                blink_events.append(frame_idx)
                blink_run = 0
            else:
                blink_run = 0

            # --- Head pose ---
            pose_angles.append(estimate_head_pose(pts, (h, w)))

            # --- Landmarks & face flag ---
            landmark_positions.append(pts)  # (468, 2)
            face_detected.append(True)
        else:
            # No face this frame → store placeholders
            blink_states.append(False)
            pose_angles.append((0.0, 0.0, 0.0))
            landmark_positions.append(np.zeros((468, 2), dtype=np.float32))
            face_detected.append(False)

        frame_idx += 1

    # If video ended while eyes were still "closed" long enough, count that blink
    if blink_run >= consecutive_frames:
        blink_events.append(frame_idx - 1)

    # Cleanup resources
    cap.release()
    face_mesh.close()

    if not blink_states:
        return {"error": "no frames processed"}

    # ---- Aggregate temporal metrics ----

    # 1) Blink interval regularity (consistency) ∈ [0,1]
    #    Regular intervals → high score; erratic intervals → lower score.
    blink_intervals = np.diff(blink_events) if len(blink_events) > 1 else np.array([])
    if blink_intervals.size > 0:
        mu = float(np.mean(blink_intervals))
        sd = float(np.std(blink_intervals))
        # Normalize by mean; clamp to [0,1]; invert because lower CV = better.
        blink_consistency = 1.0 - min(1.0, (sd / max(1.0, mu)))
    else:
        # If we cannot establish interval structure (0 or 1 blink), stay neutral.
        blink_consistency = 0.5

    # 2) Pose stability ∈ [0,1]
    #    Lower angular variance (pitch,yaw,roll) → higher stability.
    if pose_angles:
        poses = np.array(pose_angles, dtype=np.float32)  # shape (N, 3)
        per_axis_std = np.std(poses, axis=0)
        # Normalize by a 10° reference range and invert.
        pose_stability = 1.0 - min(1.0, float(np.mean(per_axis_std)) / 10.0)
    else:
        pose_stability = 0.0

    # 3) Landmark jitter (unitless, lower is better)
    #    Mean inter-frame displacement per landmark, normalized by image diagonal.
    #    Only compute between pairs where *both frames* had a detected face.
    if len(landmark_positions) > 1 and last_h and last_w:
        diag = float(math.hypot(last_w, last_h))
        step_means: List[float] = []
        for i in range(1, len(landmark_positions)):
            if face_detected[i] and face_detected[i - 1]:
                d = np.linalg.norm(landmark_positions[i] - landmark_positions[i - 1], axis=1)  # (468,)
                step_means.append(float(np.mean(d)))
        jitter = (float(np.mean(step_means)) / max(1e-6, diag)) if step_means else 0.0
    else:
        jitter = 0.0

    face_ratio = float(sum(face_detected)) / float(len(face_detected))
    blink_rate = (len(blink_events) / duration_s) if duration_s > 0 else 0.0

    return {
        "temporal_blink_score": float(blink_consistency),
        "temporal_pose_stability": float(pose_stability),
        "temporal_landmark_jitter": float(jitter),
        "blink_rate": float(blink_rate),
        "face_detected_ratio": float(face_ratio),
        "frames_processed": int(len(blink_states)),
        "duration_s": float(duration_s),
    }


def main():
    """
    CLI entry-point.

    Typical usage (PowerShell):
        python .\\src\\temporal_inconsistency.py `
          --clips .\\backend\\data\\derived\\clips.jsonl `
          --clips-root .\\backend\\data\\derived\\clips `
          --out .\\backend\\data\\derived\\clips_temporal.jsonl `
          --ear-threshold 0.25 --consecutive-frames 2 --mediapipe-complexity 0 --skip-frames 1
    """
    ap = argparse.ArgumentParser(description="Temporal inconsistency analysis over per-shot clips")
    ap.add_argument("--clips", default="backend/data/derived/clips.jsonl", help="JSONL manifest of clips")
    ap.add_argument("--clips-root", default="backend/data/derived/clips", help="Root folder containing clips")
    ap.add_argument("--out", default="backend/data/derived/clips_temporal.jsonl", help="Output metrics JSONL")
    ap.add_argument("--ear-threshold", type=float, default=0.25, help="EAR threshold for blink detection")
    ap.add_argument("--consecutive-frames", type=int, default=2, help="Consecutive frames to confirm a blink")
    ap.add_argument("--mediapipe-complexity", type=int, default=0, choices=[0, 1, 2], help="MediaPipe model complexity")
    ap.add_argument("--min-detection-confidence", type=float, default=0.5, help="Min face detection confidence")
    ap.add_argument("--skip-frames", type=int, default=1, help="Process every Nth frame")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N clips")
    args = ap.parse_args()

    clips_manifest = Path(args.clips)
    clips_root = Path(args.clips_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def resolve_clip_path(clip_uri: str) -> Path:
        """
        Robustly convert a clip_uri (usually 'clips/<sha>/shot_<idx>.mp4') to a filesystem path
        under --clips-root, but do not fail if 'clips/' prefix is missing.
        """
        p = Path(clip_uri)
        try:
            rel = p.relative_to("clips") if p.parts and p.parts[0] == "clips" else p
        except Exception:
            rel = p
        return clips_root / rel

    processed = 0
    with open(out_path, "w", encoding="utf-8") as fw:
        if clips_manifest.exists():
            # Preferred path: iterate an explicit manifest of clips
            with open(clips_manifest, encoding="utf-8") as fr:
                for line in fr:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        meta = json.loads(line)
                    except Exception:
                        continue

                    clip_uri = meta.get("clip_uri")
                    if not clip_uri:
                        continue

                    clip_path = resolve_clip_path(clip_uri)
                    metrics = process_clip_temporal(
                        clip_path,
                        ear_threshold=args.ear_threshold,
                        consecutive_frames=args.consecutive_frames,
                        mediapipe_complexity=args.mediapipe_complexity,
                        min_detection_confidence=args.min_detection_confidence,
                        skip_frames=args.skip_frames,
                    )

                    row = {
                        "asset_id": meta.get("asset_id"),
                        "sha256": meta.get("sha256"),
                        "shot_index": meta.get("shot_index"),
                        "clip_uri": clip_uri,
                        **metrics,
                        "tool_versions": {"mediapipe": mp.__version__, "opencv": cv2.__version__},
                    }
                    fw.write(json.dumps(row) + "\n")
                    processed += 1
                    print(f"[temporal] {clip_uri}  "
                          f"blink={metrics.get('temporal_blink_score', 0):.2f}  "
                          f"pose={metrics.get('temporal_pose_stability', 0):.2f}  "
                          f"jitter={metrics.get('temporal_landmark_jitter', 0):.4f}")

                    if args.limit and processed >= args.limit:
                        break
        else:
            # Fallback: crawl the clips folder for *.mp4 files
            for clip_path in clips_root.rglob("*.mp4"):
                rel = clip_path.relative_to(clips_root).as_posix()

                metrics = process_clip_temporal(
                    clip_path,
                    ear_threshold=args.ear_threshold,
                    consecutive_frames=args.consecutive_frames,
                    mediapipe_complexity=args.mediapipe_complexity,
                    min_detection_confidence=args.min_detection_confidence,
                    skip_frames=args.skip_frames,
                )

                row = {
                    "clip_uri": f"clips/{rel}",  # synthesize a URI consistent with manifest convention
                    **metrics,
                    "tool_versions": {"mediapipe": mp.__version__, "opencv": cv2.__version__},
                }
                fw.write(json.dumps(row) + "\n")
                processed += 1
                print(f"[temporal] {rel}  "
                      f"blink={metrics.get('temporal_blink_score', 0):.2f}  "
                      f"pose={metrics.get('temporal_pose_stability', 0):.2f}  "
                      f"jitter={metrics.get('temporal_landmark_jitter', 0):.4f}")

                if args.limit and processed >= args.limit:
                    break

    print(f"Temporal analysis processed {processed} clips → {args.out}")


if __name__ == "__main__":
    main()