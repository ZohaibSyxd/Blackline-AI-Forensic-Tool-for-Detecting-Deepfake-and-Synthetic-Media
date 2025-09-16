"""
Scene (shot) detection using FFmpeg scene score.

Outputs (JSONL): backend/data/derived/shots.jsonl
One row per shot:
  {
    "asset_id": "...",
    "sha256": "...",
    "stored_path": "sha/filename.mp4",
    "store_root": "backend/data/raw",
    "shot_index": 0,
    "t_start_ms": 0,
    "t_end_ms": 12345,
    "detector": "ffmpeg_scene",
    "threshold": 0.30,
    "cut_score": null   # scene_score that created this boundary (None for first shot)
  }

Usage (PowerShell):
    python .\src\scene_detect.py --audit .\backend\data\audit\ingest_log.jsonl ^
        --out .\backend\data\derived\shots.jsonl --threshold 0.30 --min-shot-ms 400
"""
from __future__ import annotations
import argparse, json, math, re, time, shutil
from pathlib import Path
from typing import List, Tuple, Optional

from utils import read_unique_assets, ffprobe_json, run_command

SCENE_RE = re.compile(
    r"pts_time[:=]\s*([0-9]+(?:\.[0-9]+)?)|lavfi\.scene_score(?:\s*[:=]\s*|\s*value=)([0-9]+(?:\.[0-9]+)?)"
)

def parse_scene_times(ffmpeg_stderr: str) -> List[Tuple[float, float]]:
    """
    Parse (pts_time, scene_score) pairs from ffmpeg showinfo output.
    Returns list of tuples [(t_sec, score), ...] in ascending time.
    """
    out: List[Tuple[float, float]] = []
    cur_t: Optional[float] = None
    cur_s: Optional[float] = None
    for m in SCENE_RE.finditer(ffmpeg_stderr):
        t, s = m.groups()
        if t is not None:
            cur_t = float(t)
        elif s is not None:
            cur_s = float(s)
        if cur_t is not None and cur_s is not None:
            out.append((cur_t, cur_s))
            cur_t, cur_s = None, None
    # ensure strictly increasing (dedupe jitter)
    out.sort(key=lambda x: x[0])
    dedup: List[Tuple[float, float]] = []
    last_t = -1.0
    for t, s in out:
        if t <= last_t + 1e-3:
            continue
        dedup.append((t, s))
        last_t = t
    return dedup

def detect_scenes_ffmpeg(path: Path, threshold: float) -> List[Tuple[float, float]]:
    """
    Run ffmpeg scene detection (no output frames) and parse showinfo logs.
    Returns [(scene_time_sec, scene_score), ...]
    """
    if shutil.which("ffmpeg") is None:
        return []
    # -nostdin to avoid blocking; -hide_banner; -v info to capture showinfo lines
    vf = f"select='gt(scene,{threshold})',showinfo"
    p = run_command([
        "ffmpeg", "-nostdin", "-hide_banner", "-v", "info",
        "-i", str(path),
        "-filter:v", vf,
        "-f", "null", "-"
    ])
    stderr = (p.stderr or "") + (p.stdout or "")
    return parse_scene_times(stderr)

def to_ms(x: float) -> int:
    return int(round(1000.0 * x))

def main():
    ap = argparse.ArgumentParser(description="Scene detection → shots.jsonl")
    ap.add_argument("--audit", default="backend/data/audit/ingest_log.jsonl")
    ap.add_argument("--out", default="backend/data/derived/shots.jsonl")
    ap.add_argument("--threshold", type=float, default=0.30, help="FFmpeg scene threshold (0.3–0.5 typical)")
    ap.add_argument("--min-shot-ms", type=int, default=400, help="Minimum shot length to keep/merge (ms)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as w:
        for rec in read_unique_assets(Path(args.audit)):
            store_root = rec.get("store_root")
            in_path = Path(store_root, rec["stored_path"]) if store_root else Path(rec["stored_path"])
            if not in_path.exists():
                continue
            mime = (rec.get("mime") or "").lower()
            if not mime.startswith("video/"):
                continue

            probe = ffprobe_json(in_path) or {}
            fmt = probe.get("format") or {}
            try:
                duration_s = float(fmt.get("duration")) if fmt.get("duration") else None
            except Exception:
                duration_s = None
            if not duration_s or duration_s <= 0:
                # fallback: try decoding later steps; for now, skip
                continue

            cuts = detect_scenes_ffmpeg(in_path, threshold=args.threshold)  # [(t_sec, score), ...]
            # Build shot boundaries: 0, cut1, cut2, ..., duration
            b_times = [0.0] + [t for t, _ in cuts if 0.0 < t < duration_s] + [duration_s]
            b_scores = [None] + [s for _, s in cuts]  # score associated with entering shot i (None for first)

            # Enforce min-shot-ms by merging too-short shots with neighbors
            min_s = args.min_shot_ms / 1000.0
            i = 0
            while i < len(b_times) - 1:
                cur_len = b_times[i+1] - b_times[i]
                if cur_len < min_s and i+2 < len(b_times):
                    # Merge with next: drop boundary i+1 and its score
                    del b_times[i+1]
                    del b_scores[i+1]
                    # do not advance i, re-check merged segment
                else:
                    i += 1

            # Emit rows
            for idx in range(len(b_times) - 1):
                t_start = to_ms(b_times[idx])
                t_end   = to_ms(b_times[idx+1])
                row = {
                    "asset_id": rec.get("asset_id"),
                    "sha256": rec.get("sha256"),
                    "stored_path": rec.get("stored_path"),
                    "store_root": store_root,
                    "shot_index": idx,
                    "t_start_ms": t_start,
                    "t_end_ms": t_end,
                    "detector": "ffmpeg_scene",
                    "threshold": args.threshold,
                    "cut_score": b_scores[idx],  # None for first shot
                }
                w.write(json.dumps(row) + "\n")
                written += 1

            print(f"[shots] {in_path.name}: {len(b_times)-1} shots (cuts={len(cuts)})")

            if args.limit and written >= args.limit:
                break

    dt = time.time() - t0
    print(f"Wrote {written} shot rows → {out_path} in {dt:.1f}s")

if __name__ == "__main__":
    main()
