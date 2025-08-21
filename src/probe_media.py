"""
src/probe_media.py

Probe media files with ffprobe and ExifTool.

What it does
- Reads unique assets from the ingest audit log.
- Runs ffprobe to collect container/stream metadata.
- Runs ExifTool (if available) for EXIF/tags.
- Writes one JSONL line per asset containing raw tool outputs and a small summary.

Usage (PowerShell)
    python .\src\probe_media.py --audit .\data\audit\ingest_log.jsonl --out .\data\derived\probe.jsonl
"""
import argparse, json, subprocess, shutil, time
from pathlib import Path

def read_unique_assets(audit_path: Path):
    """Yield one record per stored_path from the audit log (deduplicated)."""
    seen = set()
    with open(audit_path, encoding="utf-8") as r:
        for line in r:
            rec = json.loads(line)
            key = rec.get("stored_path")
            if not key or key in seen:
                continue
            seen.add(key)
            yield rec

def _run(cmd):
    """Run a subprocess and return stdout (or stderr on failure)."""
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           check=True, text=True)
        return p.stdout
    except FileNotFoundError:
        return None
    except subprocess.CalledProcessError as e:
        return e.stdout or e.stderr

def ffprobe_json(path: Path):
    """Return parsed ffprobe JSON or None if ffprobe is missing/failed."""
    if shutil.which("ffprobe") is None:
        return None
    out = _run(["ffprobe", "-v", "error", "-print_format", "json",
                "-show_format", "-show_streams", str(path)])
    try:
        return json.loads(out) if out else None
    except Exception:
        return None

def exiftool_json(path: Path):
    """Return parsed ExifTool JSON (single-object) or None if missing/failed."""
    if shutil.which("exiftool") is None:
        return None
    out = _run(["exiftool", "-json", "-n", str(path)])
    try:
        data = json.loads(out) if out else None
        return data[0] if isinstance(data, list) and data else data
    except Exception:
        return None

def _parse_rate(s):
    """Parse a rate like '30000/1001' to a float; returns None on failure."""
    if not s: return None
    try:
        num, den = s.split("/")
        num, den = int(num), int(den)
        return num / den if den else float(num)
    except Exception:
        try:
            return float(s)
        except Exception:
            return None

def summarize_ffprobe(probe):
    """Extract quick fields for convenience (width/height/fps/codec/duration)."""
    if not probe: return {}
    streams = probe.get("streams") or []
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not v: return {}
    fmt = probe.get("format") or {}
    return {
        "width": v.get("width"),
        "height": v.get("height"),
        "fps": _parse_rate(v.get("avg_frame_rate") or v.get("r_frame_rate")),
        "codec": v.get("codec_name"),
        "duration_s": float(fmt.get("duration")) if fmt.get("duration") else None,
        "nb_streams": fmt.get("nb_streams"),
    }

def main():
    ap = argparse.ArgumentParser(description="Probe media with ffprobe + exiftool, write JSONL")
    ap.add_argument("--audit", default="data/audit/ingest_log.jsonl")
    ap.add_argument("--out", default="data/derived/probe.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="probe at most N assets")
    args = ap.parse_args()

    Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)

    count = 0
    with open(args.out, "w", encoding="utf-8") as w:
        for rec in read_unique_assets(Path(args.audit)):
            p = Path(rec["stored_path"])
            ffj = ffprobe_json(p)
            exj = exiftool_json(p)
            summary = summarize_ffprobe(ffj)
            row = {
                "sha256": rec["sha256"],
                "stored_path": rec["stored_path"],
                "mime": rec.get("mime"),
                "probe": ffj,      # raw ffprobe JSON
                "exif": exj,       # raw exif JSON (single-object)
                "summary": summary,
                "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "tool_versions": {"ingest": rec.get("tool_version")},
            }
            w.write(json.dumps(row) + "\n")
            count += 1
            print(f"[probed] {p.name} {summary.get('width')}x{summary.get('height')} @ {summary.get('fps')} fps "
                  f"| sha256={rec['sha256'][:12]}...")
            if args.limit and count >= args.limit:
                break
    print(f"Wrote {count} rows -> {args.out}")

if __name__ == "__main__":
    main()
