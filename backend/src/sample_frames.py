"""
Per-shot frame sampling (and optional clip extraction).

Outputs:
  1) Saves frames to: backend/data/derived/frames/<sha>/<shot_index>/*.jpg
  2) Writes JSONL:    backend/data/derived/frames.jsonl
     {
       "asset_id": "...", "sha256": "...",
       "shot_index": 3, "frame_index": 12,
       "approx_t_ms": 123456,                  # approximated from shot start + idx/fps
       "uri": "frames/<sha>/3/000012.jpg",
       "fps": 8, "keyframe": false,            # true for first frame in each shot
       "stored_path": "sha/file.mp4", "store_root": "..."
     }

(OPTIONAL) Per-shot clip extraction:
  Saves to backend/data/derived/clips/<sha>/shot_<idx>.mp4
  - Copy mode for speed; re-encode if --reencode is passed.

Usage (PowerShell):
    python .\src\sample_frames.py ^
        --shots .\backend\data\derived\shots.jsonl ^
        --frames-out .\backend\data\derived\frames.jsonl ^
        --frames-root .\backend\data\derived\frames ^
        --fps 8 --jpeg-quality 90 ^
        --extract-clips --clips-root .\backend\data\derived\clips
"""
from __future__ import annotations
import argparse, json, math, os, shutil, time, subprocess
from pathlib import Path
from typing import Dict, Any

from utils import run_command

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def sample_frames_one_shot(
    video_path: Path,
    sha: str,
    shot_index: int,
    t_start_ms: int,
    t_end_ms: int,
    fps: int,
    jpeg_quality: int,
    frames_root: Path,
) -> int:
    """
    Extract frames for a single shot using FFmpeg. Returns number of frames written.
    """
    out_dir = frames_root / sha / str(shot_index)
    ensure_dir(out_dir)
    # Use integer frame numbering for simplicity (000000.jpg ...). Timestamps are approximated in JSON via fps.
    pattern = str(out_dir / "%06d.jpg")
    ss = f"{t_start_ms/1000.0:.3f}"
    to = f"{t_end_ms/1000.0:.3f}"
    vf = f"fps={fps}"
    # -y overwrite, -q:v jpeg quality (2..31; lower is better). We'll map [1..31] from given percent-like 90→2, 70→5 etc.
    qv = max(2, min(31, int(round((100 - jpeg_quality) * 0.29))))  # crude mapping
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-ss", ss, "-to", to,
        "-i", str(video_path),
        "-vf", vf,
        "-q:v", str(qv),
        "-vsync", "vfr",
        "-y", pattern
    ]
    p = run_command(cmd)
    if p.returncode != 0:
        return 0
    # Count files
    return sum(1 for _ in out_dir.glob("*.jpg"))

def extract_clip_one_shot(
    video_path: Path,
    sha: str,
    shot_index: int,
    t_start_ms: int,
    t_end_ms: int,
    clips_root: Path,
    reencode: bool = False,
) -> Path:
    """
    Extract a single shot clip. Copy by default; re-encode if requested.
    """
    out_dir = clips_root / sha
    ensure_dir(out_dir)
    out_path = out_dir / f"shot_{shot_index:03d}.mp4"
    ss = f"{t_start_ms/1000.0:.3f}"
    to = f"{t_end_ms/1000.0:.3f}"
    if reencode:
        cmd = [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-ss", ss, "-to", to, "-i", str(video_path),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-an", "-y", str(out_path)
        ]
    else:
        # Stream copy (fast, may start on nearest keyframe)
        cmd = [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-ss", ss, "-to", to, "-i", str(video_path),
            "-c", "copy", "-y", str(out_path)
        ]
    p = run_command(cmd)
    return out_path if p.returncode == 0 and out_path.exists() else Path()

def main():
    ap = argparse.ArgumentParser(description="Per-shot frame sampling and optional clip extraction.")
    ap.add_argument("--shots", default="backend/data/derived/shots.jsonl")
    ap.add_argument("--frames-out", default="backend/data/derived/frames.jsonl")
    ap.add_argument("--frames-root", default="backend/data/derived/frames")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--jpeg-quality", type=int, default=90, help="Approx JPEG quality 1..100 (mapped to q:v)")
    ap.add_argument("--extract-clips", action="store_true")
    ap.add_argument("--clips-root", default="backend/data/derived/clips")
    ap.add_argument("--reencode", action="store_true", help="Re-encode clips instead of stream copy")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    frames_root = Path(args.frames_root)
    ensure_dir(frames_root)
    clips_root = Path(args.clips_root)
    if args.extract_clips:
        ensure_dir(clips_root)

    rows_written = 0
    t0 = time.time()

    with open(args.frames_out, "w", encoding="utf-8") as fw, open(args.shots, encoding="utf-8") as rs:
        for line in rs:
            if not line.strip():
                continue
            shot = json.loads(line)
            sha = shot["sha256"]
            store_root = shot.get("store_root")
            video_path = Path(store_root, shot["stored_path"]) if store_root else Path(shot["stored_path"])
            if not video_path.exists():
                continue

            shot_idx = int(shot["shot_index"])
            t_start_ms = int(shot["t_start_ms"])
            t_end_ms = int(shot["t_end_ms"])
            if t_end_ms <= t_start_ms:
                continue

            # 1) Optional clip extraction
            if args.extract_clips:
                _ = extract_clip_one_shot(
                    video_path, sha, shot_idx, t_start_ms, t_end_ms,
                    clips_root, reencode=args.reencode
                )

            # 2) Frame sampling
            n = sample_frames_one_shot(
                video_path, sha, shot_idx, t_start_ms, t_end_ms,
                fps=args.fps, jpeg_quality=args.jpeg_quality, frames_root=frames_root
            )

            # 3) Emit JSONL rows for frames
            #    We approximate timestamps by uniform spacing from shot start.
            #    The first frame in each shot is tagged as keyframe=True (good heuristic for Step 4).
            if n > 0:
                dt_ms = int(round(1000.0 / max(1, args.fps)))
                for i in range(n):
                    approx_t = t_start_ms + i * dt_ms
                    rel_uri = f"frames/{sha}/{shot_idx}/{i:06d}.jpg"
                    row: Dict[str, Any] = {
                        "asset_id": shot.get("asset_id"),
                        "sha256": sha,
                        "stored_path": shot.get("stored_path"),
                        "store_root": store_root,
                        "shot_index": shot_idx,
                        "frame_index": i,
                        "approx_t_ms": approx_t,
                        "uri": rel_uri,
                        "fps": args.fps,
                        "keyframe": (i == 0),
                    }
                    fw.write(json.dumps(row) + "\n")
                    rows_written += 1

            print(f"[frames] {video_path.name} | shot {shot_idx:03d} → {n} frames")

            if args.limit and rows_written >= args.limit:
                break

    dt = time.time() - t0
    print(f"Wrote {rows_written} frame rows → {args.frames_out} in {dt:.1f}s")

if __name__ == "__main__":
    main()
