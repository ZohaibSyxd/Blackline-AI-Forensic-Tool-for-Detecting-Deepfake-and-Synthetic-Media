"""
Format validation & safe decoding

Usage (PowerShell):
  python .\src\validate_media.py --audit .\data\audit\ingest_log.jsonl --out .\data\derived\validate.jsonl

Strategy (videos only)
- Videos: ffprobe to confirm streams; ffmpeg dry-run decode to catch corrupt samples.

Outputs one JSON per asset with fields like:
  sha256, stored_path, mime, media_kind, format_valid, decode_valid, width, height, duration_s, errors[]
"""

import argparse, json, time, shutil
from pathlib import Path
from typing import Optional, Dict, Any, List

from .utils import read_unique_assets, ffprobe_json, ffmpeg_decode_ok, run_command

def classify_kind(mime: Optional[str], path: Path) -> str:
    """Classify media kind from MIME or extension. Images are marked unsupported.

    This pipeline is video-only; images/other kinds will be flagged accordingly.
    """
    m = (mime or "").lower()
    if m.startswith("image/"):
        return "image"  # will be treated as unsupported below
    if m.startswith("video/"):
        return "video"
    # fallback by extension
    ext = path.suffix.lower()
    if ext in {".jpg",".jpeg",".png",".gif",".bmp",".tiff",".tif",".webp"}:
        return "image"  # will be treated as unsupported below
    if ext in {".mp4",".avi",".mov",".mkv",".webm",".m4v",".mpg",".mpeg"}:
        return "video"
    return "other"

def summarize_probe(probe: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize key ffprobe fields used for validation and reporting."""
    out = {"width": None, "height": None, "duration_s": None, "video_streams": 0, "audio_streams": 0}
    if not probe:
        return out
    fmt = probe.get("format") or {}
    streams = probe.get("streams") or []
    vstreams = [s for s in streams if s.get("codec_type") == "video"]
    astreams = [s for s in streams if s.get("codec_type") == "audio"]
    out["video_streams"] = len(vstreams)
    out["audio_streams"] = len(astreams)
    if fmt.get("duration") is not None:
        try:
            out["duration_s"] = float(fmt["duration"])
        except Exception:
            pass
    if vstreams:
        v0 = vstreams[0]
        out["width"] = v0.get("width")
        out["height"] = v0.get("height")
    return out


def validate_asset(stored_path: str, store_root: str | None, sha256: str, mime: str | None) -> Dict[str, Any]:
    """Validate a single asset (already ingested). Returns validation row dict.
    Does not write files.
    """
    path = Path(store_root, stored_path) if store_root else Path(stored_path)
    tstart = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ffprobe_ver = None
    ffmpeg_ver = None
    if shutil.which("ffprobe"):
        p = run_command(["ffprobe", "-version"])
        ffprobe_ver = (p.stdout or p.stderr).splitlines()[0] if (p.stdout or p.stderr) else None
    if shutil.which("ffmpeg"):
        p = run_command(["ffmpeg", "-version"])
        ffmpeg_ver = (p.stdout or p.stderr).splitlines()[0] if (p.stdout or p.stderr) else None

    if not path.exists():
        return {
            "asset_id": None,
            "sha256": sha256,
            "stored_path": stored_path,
            "store_root": store_root,
            "mime": mime,
            "media_kind": None,
            "format_valid": False,
            "decode_valid": None,
            "width": None,
            "height": None,
            "duration_s": None,
            "errors": ["file_missing"],
            "when": tstart,
            "tool_versions": {"ffprobe": ffprobe_ver, "ffmpeg": ffmpeg_ver},
        }

    kind = classify_kind(mime, path)
    errors: List[str] = []
    width = height = None
    duration_s = None
    format_valid: Optional[bool] = None
    decode_valid: Optional[bool] = None

    if kind == "image":
        format_valid = False
        decode_valid = None
        errors.append("unsupported_kind:image")
    elif kind == "video":
        probe = ffprobe_json(path)
        summ = summarize_probe(probe)
        width, height = summ.get("width"), summ.get("height")
        duration_s = summ.get("duration_s")
        has_streams = (summ.get("video_streams") or 0) + (summ.get("audio_streams") or 0) > 0
        format_valid = True if (probe and has_streams) else False
        if probe is None and shutil.which("ffprobe") is None:
            errors.append("ffprobe_missing")
        decode_valid = ffmpeg_decode_ok(path)
        if decode_valid is None and shutil.which("ffmpeg") is None:
            errors.append("ffmpeg_missing")
        if decode_valid is False:
            errors.append("decode_failed")
    else:
        format_valid = False
        errors.append("unsupported_kind")

    return {
        "asset_id": None,
        "sha256": sha256,
        "stored_path": stored_path,
        "store_root": store_root,
        "mime": mime,
        "media_kind": kind,
        "format_valid": format_valid,
        "decode_valid": decode_valid,
        "width": width,
        "height": height,
        "duration_s": duration_s,
        "errors": errors,
        "when": tstart,
        "tool_versions": {"ffprobe": ffprobe_ver, "ffmpeg": ffmpeg_ver},
    }


def main():
    ap = argparse.ArgumentParser(description="Validate media containers/codecs and safe decode")
    ap.add_argument("--audit", default="backend/data/audit/ingest_log.jsonl", help="audit log path")
    ap.add_argument("--out", default="backend/data/derived/validate.jsonl", help="output JSONL file")
    ap.add_argument("--limit", type=int, default=None, help="validate at most N assets")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tstart = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ffprobe_ver = None
    ffmpeg_ver = None
    pillow_ver = None

    # Capture tool versions for traceability in outputs
    if shutil.which("ffprobe"):
        p = run_command(["ffprobe", "-version"])
        ffprobe_ver = (p.stdout or p.stderr).splitlines()[0] if (p.stdout or p.stderr) else None
    if shutil.which("ffmpeg"):
        p = run_command(["ffmpeg", "-version"])
        ffmpeg_ver = (p.stdout or p.stderr).splitlines()[0] if (p.stdout or p.stderr) else None
    try:
        import PIL  # type: ignore
        pillow_ver = getattr(PIL, "__version__", None)
    except Exception:
        pillow_ver = None

    count = 0
    with open(out_path, "w", encoding="utf-8") as w:
        # Iterate unique assets from ingest audit log
        for rec in read_unique_assets(Path(args.audit)):
            store_root = rec.get("store_root")
            path = Path(store_root, rec["stored_path"]) if store_root else Path(rec["stored_path"]) 
            mime = rec.get("mime")
            if not path.exists():
                # Record missing files explicitly to keep audit trail complete
                row = {
                    "asset_id": rec.get("asset_id"),
                    "sha256": rec.get("sha256"),
                    "stored_path": rec.get("stored_path"),
                    "store_root": store_root,
                    "mime": mime,
                    "media_kind": None,
                    "format_valid": False,
                    "decode_valid": None,
                    "width": None,
                    "height": None,
                    "duration_s": None,
                    "errors": ["file_missing"],
                    "when": tstart,
                    "tool_versions": {"ffprobe": ffprobe_ver, "ffmpeg": ffmpeg_ver, "pillow": pillow_ver},
                }
                w.write(json.dumps(row) + "\n")
                count += 1
                if args.limit and count >= args.limit:
                    break
                continue

            kind = classify_kind(mime, path)
            errors: List[str] = []

            width = height = None
            duration_s = None
            format_valid: Optional[bool] = None
            decode_valid: Optional[bool] = None

            if kind == "image":
                # For a video-only pipeline, images are not supported
                format_valid = False
                decode_valid = None
                errors.append("unsupported_kind:image")
            elif kind == "video":
                # Probe container/streams and attempt a dry-run decode to surface corruption
                probe = ffprobe_json(path)
                summ = summarize_probe(probe)
                width, height = summ.get("width"), summ.get("height")
                duration_s = summ.get("duration_s")
                # consider format_valid True if we have at least one stream and width/height when video present
                has_streams = (summ.get("video_streams") or 0) + (summ.get("audio_streams") or 0) > 0
                format_valid = True if (probe and has_streams) else False
                if probe is None and shutil.which("ffprobe") is None:
                    errors.append("ffprobe_missing")
                decode_valid = ffmpeg_decode_ok(path)
                if decode_valid is None and shutil.which("ffmpeg") is None:
                    errors.append("ffmpeg_missing")
                if decode_valid is False:
                    errors.append("decode_failed")
            else:
                # unsupported/other
                format_valid = False
                errors.append("unsupported_kind")

            # Emit one JSONL row per asset with validation results
            row = {
                "asset_id": rec.get("asset_id"),
                "sha256": rec.get("sha256"),
                "stored_path": rec.get("stored_path"),
                "store_root": store_root,
                "mime": mime,
                "media_kind": kind,
                "format_valid": format_valid,
                "decode_valid": decode_valid,
                "width": width,
                "height": height,
                "duration_s": duration_s,
                "errors": errors,
                "when": tstart,
                "tool_versions": {"ffprobe": ffprobe_ver, "ffmpeg": ffmpeg_ver, "pillow": pillow_ver},
            }
            w.write(json.dumps(row) + "\n")
            count += 1
            print(f"[validate] {path.name} kind={kind} format={format_valid} decode={decode_valid}")
            # Support quick smoke tests by honoring --limit
            if args.limit and count >= args.limit:
                break

    print(f"Wrote {count} rows -> {out_path}")


if __name__ == "__main__":
    main()
