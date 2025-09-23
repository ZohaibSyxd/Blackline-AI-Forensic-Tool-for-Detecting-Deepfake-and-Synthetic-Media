r"""
src/probe_media.py

Probe video files with ffprobe and ExifTool (videos only).

What it does
- Reads unique assets from the ingest audit log.
- Runs ffprobe to collect container/stream metadata.
- Runs ExifTool (if available) for EXIF/tags.
- Writes one JSONL line per asset containing raw tool outputs and a small summary.

Usage (PowerShell)
    python .\src\probe_media.py --audit .\data\audit\ingest_log.jsonl --out .\data\derived\probe.jsonl
"""
import argparse, json, time
from pathlib import Path

from .utils import read_unique_assets, ffprobe_json, exiftool_json, summarize_ffprobe


def probe_asset(
    stored_path: str,
    store_root: str | None,
    sha256: str,
    mime: str | None,
    *,
    no_exif: bool = False
) -> dict | None:
    """Probe a single asset already ingested. Returns probe record structure.

    stored_path: relative path like "<sha>/<filename>" under store_root
    store_root: absolute or relative root directory where raw assets were stored
    sha256: content hash
    mime: optional mime determined at ingest
    no_exif: skip exiftool for speed
    """
    p = Path(store_root, stored_path) if store_root else Path(stored_path)
    if not p.exists():
        return None
    if mime and not str(mime).lower().startswith("video/"):
        return None
    ffj = ffprobe_json(p)
    exj = None if no_exif else exiftool_json(p)
    summary = summarize_ffprobe(ffj)
    return {
        "asset_id": None,  # caller can fill
        "sha256": sha256,
        "stored_path": stored_path,
        "store_root": store_root,
        "mime": mime,
        "probe": ffj,
        "exif": exj,
        "summary": summary,
        "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool_versions": {},
    }


def main():
    ap = argparse.ArgumentParser(description="Probe media with ffprobe + exiftool, write JSONL")
    ap.add_argument("--audit", default="backend/data/audit/ingest_log.jsonl")
    ap.add_argument("--out", default="backend/data/derived/probe.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="probe at most N assets")
    ap.add_argument("--no-exif", action="store_true", help="skip ExifTool probing for speed")
    args = ap.parse_args()

    Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)

    count = 0
    with open(args.out, "w", encoding="utf-8") as w:
        for rec in read_unique_assets(Path(args.audit)):
            # stored_path is now relative to store_root; reconstruct absolute path if store_root present
            store_root = rec.get("store_root")
            p = Path(store_root, rec["stored_path"]) if store_root else Path(rec["stored_path"])
            # Skip non-video assets in a video-only pipeline
            if rec.get("mime") and not str(rec.get("mime")).lower().startswith("video/"):
                continue
            ffj = ffprobe_json(p)
            exj = None if args.no_exif else exiftool_json(p)
            summary = summarize_ffprobe(ffj)
            row = {
                "asset_id": rec.get("asset_id"),
                "sha256": rec["sha256"],
                "stored_path": rec["stored_path"],
                "store_root": store_root,
                "mime": rec.get("mime"),
                "probe": ffj,
                "exif": exj,
                "summary": summary,
                "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "tool_versions": {"ingest": rec.get("tool_version")},
            }
            w.write(json.dumps(row) + "\n")
            count += 1
            print(
                f"[probed] {p.name} {summary.get('width')}x{summary.get('height')} @ {summary.get('fps')} fps | sha256={rec['sha256'][:12]}..."
            )
            if args.limit and count >= args.limit:
                break
    print(f"Wrote {count} rows -> {args.out}")


if __name__ == "__main__":
    main()
