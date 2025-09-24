from __future__ import annotations
import os
from pathlib import Path
import shutil
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DERIVED = DATA_DIR / "derived"
RAW = DATA_DIR / "raw"
AUDIT = DATA_DIR / "audit"


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.fixture(scope="session")
def tmp_data_dir(tmp_path_factory: pytest.TempPathFactory):
    """Create an isolated backend/data-like directory for tests."""
    base = tmp_path_factory.mktemp("bl_data")
    (base / "derived").mkdir(parents=True, exist_ok=True)
    (base / "raw").mkdir(parents=True, exist_ok=True)
    (base / "audit").mkdir(parents=True, exist_ok=True)
    return base


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    except subprocess.CalledProcessError as e:
        return e


def make_synthetic_video(out_path: Path, w=320, h=180, fps=24, dur1=1.0, dur2=1.0, color1="red", color2="blue") -> bool:
    """
    Create a tiny video with two color segments back-to-back to yield a hard cut.
    Requires ffmpeg. Returns True if successful.
    """
    if not have_ffmpeg():
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Two color sources concatenated; ensure exact duration via -t and fps
    filter_complex = (
        f"color=c={color1}:s={w}x{h}:r={fps}:d={dur1}[v0];"
        f"color=c={color2}:s={w}x{h}:r={fps}:d={dur2}[v1];"
        f"[v0][v1]concat=n=2:v=1:a=0,format=yuv420p[v]"
    )
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc",
        "-filter_complex", filter_complex,
        "-map", "[v]", "-t", str(dur1 + dur2),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    p = _run(cmd)
    return p.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0


@pytest.fixture(scope="session")
def synthetic_asset(tmp_data_dir: Path):
    """Generate a synthetic video in a content-addressed directory and an audit row."""
    if not have_ffmpeg():
        pytest.skip("ffmpeg/ffprobe not available")
    sha = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
    raw_dir = tmp_data_dir / "raw" / sha
    raw_dir.mkdir(parents=True, exist_ok=True)
    video = raw_dir / "test_two_cuts.mp4"
    ok = make_synthetic_video(video, dur1=1.0, dur2=1.0)
    assert ok, "Failed to generate synthetic video"

    # Write audit log with required fields
    audit_path = tmp_data_dir / "audit" / "ingest_log.jsonl"
    with open(audit_path, "w", encoding="utf-8") as w:
        rec = {
            "asset_id": "asset-1",
            "sha256": sha,
            "stored_path": f"{sha}/{video.name}",
            "store_root": str(tmp_data_dir / "raw"),
            "mime": "video/mp4",
        }
        w.write(__import__("json").dumps(rec) + "\n")

    return {
        "sha": sha,
        "video": video,
        "audit": audit_path,
        "data_root": tmp_data_dir,
        "derived": tmp_data_dir / "derived",
    }
