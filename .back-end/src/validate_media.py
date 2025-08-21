"""
Format validation & safe decoding

Usage (PowerShell):
  python .\src\validate_media.py --audit .\data\audit\ingest_log.jsonl --out .\data\derived\validate.jsonl

Strategy
- Images: open with Pillow and call .verify() (no output written).
- Videos: ffprobe to confirm streams; ffmpeg dry-run decode to catch corrupt samples.

Outputs one JSON per asset with fields like:
  sha256, stored_path, mime, media_kind, format_valid, decode_valid, width, height, duration_s, errors[]
"""

import argparse, json, shutil, subprocess, time
from pathlib import Path
from typing import Optional, Dict, Any, List


def read_unique_assets(audit_path: Path):
    seen = set()
    with open(audit_path, encoding="utf-8") as r:
        for line in r:
            rec = json.loads(line)
            key = rec.get("stored_path")
            if not key or key in seen:
                continue
            seen.add(key)
            yield rec


def classify_kind(mime: Optional[str], path: Path) -> str:
    m = (mime or "").lower()
    if m.startswith("image/"):
        return "image"
    if m.startswith("video/"):
        return "video"
    # fallback by extension
    ext = path.suffix.lower()
    if ext in {".jpg",".jpeg",".png",".gif",".bmp",".tiff",".tif",".webp"}: # We dont want image
        return "image"
    if ext in {".mp4",".avi",".mov",".mkv",".webm",".m4v",".mpg",".mpeg"}:
        return "video"
    return "other"


def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    except subprocess.CalledProcessError as e:
        return e


def ffprobe_json(path: Path) -> Optional[Dict[str, Any]]:
    if shutil.which("ffprobe") is None:
        return None
    p = _run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)])
    if p.returncode != 0:
        return None
    try:
        return json.loads(p.stdout)
    except Exception:
        return None


def summarize_probe(probe: Optional[Dict[str, Any]]) -> Dict[str, Any]:
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


def ffmpeg_decode_ok(path: Path) -> Optional[bool]:
    """Return True if a decode dry-run succeeds, False if ffmpeg reports errors, None if ffmpeg missing."""
    if shutil.which("ffmpeg") is None:
        return None
    # -v error: only show errors; -xerror: exit on first error; -f null -: decode and discard output
    # -nostdin: avoid blocking on stdin in some shells
    p = _run(["ffmpeg", "-v", "error", "-xerror", "-nostdin", "-i", str(path), "-f", "null", "-"])
    return p.returncode == 0


def pillow_verify(path: Path) -> Dict[str, Any]:
    res = {"format_valid": None, "width": None, "height": None, "error": None}
    try:
        from PIL import Image  # type: ignore
    except Exception as e:
        res["format_valid"] = None
        res["error"] = f"pillow_missing: {e}"
        return res
    try:
        with Image.open(path) as im:
            im.verify()  # quick integrity check
        # reopen to get dimensions after verify()
        with Image.open(path) as im2:
            res["width"], res["height"] = im2.size
        res["format_valid"] = True
    except Exception as e:
        res["format_valid"] = False
        res["error"] = str(e)
    return res


def main():
    ap = argparse.ArgumentParser(description="Validate media containers/codecs and safe decode")
    ap.add_argument("--audit", default="data/audit/ingest_log.jsonl", help="audit log path")
    ap.add_argument("--out", default="data/derived/validate.jsonl", help="output JSONL file")
    ap.add_argument("--limit", type=int, default=None, help="validate at most N assets")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tstart = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ffprobe_ver = None
    ffmpeg_ver = None
    pillow_ver = None

    if shutil.which("ffprobe"):
        p = _run(["ffprobe", "-version"])
        ffprobe_ver = (p.stdout or p.stderr).splitlines()[0] if (p.stdout or p.stderr) else None
    if shutil.which("ffmpeg"):
        p = _run(["ffmpeg", "-version"])
        ffmpeg_ver = (p.stdout or p.stderr).splitlines()[0] if (p.stdout or p.stderr) else None
    try:
        import PIL  # type: ignore
        pillow_ver = getattr(PIL, "__version__", None)
    except Exception:
        pillow_ver = None

    count = 0
    with open(out_path, "w", encoding="utf-8") as w:
        for rec in read_unique_assets(Path(args.audit)):
            path = Path(rec["stored_path"])
            mime = rec.get("mime")
            if not path.exists():
                row = {
                    "sha256": rec.get("sha256"),
                    "stored_path": str(path),
                    "mime": mime,
                    "media_kind": None,
                    "format_valid": False,
                    "decode_valid": None,
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
                res = pillow_verify(path)
                width, height = res.get("width"), res.get("height")
                format_valid = res.get("format_valid")
                if res.get("error"):
                    errors.append(res["error"])  # include pillow error if any
                # decode_valid not applicable for images
            elif kind == "video":
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

            row = {
                "sha256": rec.get("sha256"),
                "stored_path": str(path),
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
            if args.limit and count >= args.limit:
                break

    print(f"Wrote {count} rows -> {out_path}")


if __name__ == "__main__":
    main()
