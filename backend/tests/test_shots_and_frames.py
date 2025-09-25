from __future__ import annotations
import json
from pathlib import Path
import subprocess

from .conftest import _run


def test_scene_detect_and_sample_frames(synthetic_asset):
    d = synthetic_asset
    data_root: Path = d["data_root"]
    derived: Path = d["derived"]

    # 1) Run scene detection module via -m
    shots_path = derived / "shots.jsonl"
    cmd_shots = [
        "python", "-m", "backend.src.scene_detect",
        "--audit", str(d["audit"]),
        "--out", str(shots_path),
        "--threshold", "0.3",
        "--min-shot-ms", "200",
    ]
    p1 = _run(cmd_shots)
    assert p1.returncode == 0, f"scene_detect failed: {p1.stderr}"
    assert shots_path.exists(), "shots.jsonl not written"

    shots = [json.loads(line) for line in shots_path.read_text().splitlines() if line.strip()]
    assert len(shots) >= 1, "Expected at least 1 shot"
    # Our synthetic video has a hard cut in the middle; with threshold 0.3 and min 200ms, expect 2 shots
    # Depending on rounding it might be 2 shots exactly.
    assert len(shots) == 2, f"Expected 2 shots, got {len(shots)}"

    # 2) Run frame sampler
    frames_path = derived / "frames.jsonl"
    frames_root = derived / "frames"
    cmd_frames = [
        "python", "-m", "backend.src.sample_frames",
        "--shots", str(shots_path),
        "--frames-out", str(frames_path),
        "--frames-root", str(frames_root),
        "--fps", "4",
        "--jpeg-quality", "90",
    ]
    p2 = _run(cmd_frames)
    assert p2.returncode == 0, f"sample_frames failed: {p2.stderr}"
    assert frames_path.exists(), "frames.jsonl not written"

    frame_rows = [json.loads(line) for line in frames_path.read_text().splitlines() if line.strip()]
    assert len(frame_rows) > 0, "No frame rows written"

    # Each shot should produce at least 1 frame at 4 fps for ~1s shots
    shots_by_index = {}
    for row in frame_rows:
        shots_by_index.setdefault(row["shot_index"], 0)
        shots_by_index[row["shot_index"]] += 1
    assert set(shots_by_index.keys()) == {0, 1}, f"Unexpected shot indices: {shots_by_index.keys()}"
    assert all(n >= 3 for n in shots_by_index.values()), f"Expected >=3 frames per ~1s shot at 4fps: {shots_by_index}"

    # Verify files exist for first few frames
    first = frame_rows[0]
    first_path = (derived / first["uri"]).resolve()
    assert first_path.exists(), f"Frame file missing: {first_path}"
