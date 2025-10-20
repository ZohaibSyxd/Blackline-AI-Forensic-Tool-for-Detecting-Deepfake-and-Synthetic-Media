"""
Run compute_noise.py in chunks with a live tracker and merged output.

Features:
- Splits frames.jsonl into N-line chunks (default 5000).
- For each chunk, runs compute_noise.py and prints per-chunk progress by
  watching the chunk's output line count vs the chunk's input lines.
- Skips chunks whose output already exists (optional).
- Produces a merged output JSONL of all chunk outputs at the end.

Example:
  ./.venv/bin/python backend/src/models/run_noise_chunked.py \
    --frames backend/data/derived/frames.jsonl \
    --root backend/data/derived/frames \
    --overlays backend/data/derived/overlays \
    --out-merged backend/data/derived/frames_noise_merged.jsonl \
    --chunk-size 5000 --skip-existing-chunks
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple, List


@dataclass
class Chunk:
    idx: int
    start_line: int  # inclusive, 1-based
    end_line: int    # inclusive
    in_path: Path
    out_path: Path


def count_lines(path: Path) -> int:
    try:
        with path.open("rb") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def iter_chunks(frames_path: Path, out_dir: Path, chunk_size: int, start_line: int = 1, max_chunks: int | None = None) -> Iterator[Chunk]:
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = 0
    current_start = max(1, start_line)
    # We'll stream the file once and write successive chunk files
    with frames_path.open("r", encoding="utf-8") as fr:
        # Skip to start_line-1
        for _ in range(current_start - 1):
            if not fr.readline():
                return
        while True:
            lines: List[str] = []
            # Read up to chunk_size lines
            for _ in range(chunk_size):
                ln = fr.readline()
                if not ln:
                    break
                lines.append(ln)
            if not lines:
                break
            idx += 1
            cur_end = current_start + len(lines) - 1
            in_path = out_dir / f"frames_chunk_{current_start}_{cur_end}.jsonl"
            with in_path.open("w", encoding="utf-8") as fw:
                fw.writelines(lines)
            out_path = out_dir / f"frames_noise_chunk_{current_start}_{cur_end}.jsonl"
            yield Chunk(idx=idx, start_line=current_start, end_line=cur_end, in_path=in_path, out_path=out_path)
            current_start = cur_end + 1
            if max_chunks is not None and idx >= max_chunks:
                break


def run_compute_for_chunk(chunk: Chunk, root: Path, overlays: Path, compute_script: Path, python_exec: str, verbose: bool = True) -> None:
    cmd = [
        python_exec,
        str(compute_script),
        "--frames", str(chunk.in_path),
        "--root", str(root),
        "--out", str(chunk.out_path),
        "--overlays", str(overlays),
    ]
    if verbose:
        print(f"[chunk {chunk.idx}] start compute: lines {chunk.start_line}-{chunk.end_line}")
        print(" ", " ".join(cmd))

    # Launch process
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, text=True)

    # Determine input size for progress
    total_in = count_lines(chunk.in_path)
    last_report = 0
    # Read process output non-blocking while also tracking progress by file size
    while True:
        # Print any available stdout lines
        if proc.stdout is not None:
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                if verbose:
                    print(f"[chunk {chunk.idx}] {line.rstrip()}")

        ret = proc.poll()
        done = count_lines(chunk.out_path)
        # Throttle progress prints
        if done != last_report:
            pct = int(done * 100 / total_in) if total_in > 0 else 0
            print(f"[chunk {chunk.idx}] progress {done}/{total_in} ({pct}%)")
            last_report = done

        if ret is not None:
            # Process ended
            if proc.stdout is not None:
                # Drain any remaining output
                for line in proc.stdout:
                    if verbose:
                        print(f"[chunk {chunk.idx}] {line.rstrip()}")
            if ret != 0:
                raise RuntimeError(f"compute_noise failed for chunk {chunk.idx} with exit code {ret}")
            break

        time.sleep(1.0)

    print(f"[chunk {chunk.idx}] complete: wrote {count_lines(chunk.out_path)} rows -> {chunk.out_path}")


def merge_outputs(chunks_dir: Path, merged_out: Path) -> int:
    # Concatenate all chunk output files sorted by start_line embedded in filename
    outs = sorted(chunks_dir.glob("frames_noise_chunk_*_*.jsonl"), key=lambda p: int(p.stem.split("_")[3]))
    merged_out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with merged_out.open("w", encoding="utf-8") as w:
        for p in outs:
            with p.open("r", encoding="utf-8") as r:
                for line in r:
                    w.write(line)
                    total += 1
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description="Run compute_noise.py in chunks with progress and merging")
    ap.add_argument("--frames", default="backend/data/derived/frames.jsonl")
    ap.add_argument("--root", default="backend/data/derived/frames")
    ap.add_argument("--overlays", default="backend/data/derived/overlays")
    ap.add_argument("--out-merged", default="backend/data/derived/frames_noise_merged.jsonl")
    ap.add_argument("--chunks-dir", default="backend/data/derived/chunks", help="Working directory for chunk files")
    ap.add_argument("--chunk-size", type=int, default=5000)
    ap.add_argument("--start-line", type=int, default=1)
    ap.add_argument("--max-chunks", type=int, default=None)
    ap.add_argument("--skip-existing-chunks", action="store_true", help="Skip compute if chunk output exists and is non-empty")
    ap.add_argument("--keep-chunks", action="store_true", help="Do not delete chunk input files after processing")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    frames = Path(args.frames)
    root = Path(args.root)
    overlays = Path(args.overlays)
    merged_out = Path(args.out_merged)
    chunks_dir = Path(args.chunks_dir)

    compute_script = Path(__file__).parent / "compute_noise.py"
    python_exec = sys.executable  # Use current interpreter (ideally venv)

    total_frames = count_lines(frames)
    print(f"Total frames: {total_frames}")

    processed_frames = 0
    for chunk in iter_chunks(frames, chunks_dir, args.chunk_size, args.start_line, args.max_chunks):
        # If skipping existing chunks and output exists with content, skip compute
        if args.skip_existing_chunks and chunk.out_path.exists() and count_lines(chunk.out_path) > 0:
            print(f"[chunk {chunk.idx}] skip existing output: {chunk.out_path}")
        else:
            run_compute_for_chunk(chunk, root, overlays, compute_script, python_exec, verbose=args.verbose)

        processed_frames += count_lines(chunk.out_path)
        pct_total = int(processed_frames * 100 / total_frames) if total_frames > 0 else 0
        print(f"[overall] processed ~{processed_frames}/{total_frames} ({pct_total}%)")

        if not args.keep_chunks and chunk.in_path.exists():
            try:
                chunk.in_path.unlink()
            except OSError:
                pass

    print("Merging chunk outputs...")
    total_merged = merge_outputs(chunks_dir, merged_out)
    print(f"Merged {total_merged} rows -> {merged_out}")


if __name__ == "__main__":
    main()
