"""
src/ingest.py

Immutable ingest with SHA-256 hashing and an append-only audit log.

What it does
- Walk a file or directory, filter by known video extensions.
- Compute SHA-256 for each kept file and store it under data/raw/<sha256>/<orig_name>.
- Optionally move instead of copy to save disk.
- Write one JSON line per kept asset to data/audit/ingest_log.jsonl capturing
    who/when/how and basic asset information.

Usage (PowerShell)
    python .\src\ingest.py .\datasets --store .\data\raw --move

Notes
- Chain-of-custody: the audit log and content-addressed storage provide immutability and traceability.
"""

import argparse, hashlib, json, os, shutil, sys, time, getpass, uuid, mimetypes
from pathlib import Path

# Allowed video file extensions (detection is for videos only)
ALLOWED_EXTS = {".mp4",".m4v",".avi",".mov",".mkv",".webm",".mpg",".mpeg"}

# Simple tool version tag for traceability in downstream stages
TOOL_VERSION = "ingest-2"

def sha256sum(path, chunk_size=1024*1024):
    """Compute streaming SHA-256 of a file to avoid loading it entirely into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def iter_files(path: Path):
    """Yield files under a path (single file or recursive directory walk)."""
    if path.is_file():
        yield path
    else:
        for p in path.rglob("*"):
            if p.is_file():
                yield p

def should_keep(p: Path):
    """Return True if the file extension is in the allowed media set."""
    return p.suffix.lower() in ALLOWED_EXTS


def main():
    ap = argparse.ArgumentParser(description="Immutable ingest with SHA-256 + audit log")
    ap.add_argument("src", help="file or directory to ingest")
    ap.add_argument("--store", default="backend/data/raw", help="where to store raw files")
    ap.add_argument("--audit", default="backend/data/audit/ingest_log.jsonl", help="audit log path")
    ap.add_argument("--move", action="store_true", help="move files instead of copy (saves disk)")
    args = ap.parse_args()

    src = Path(args.src)
    store_root = Path(args.store)
    audit_path = Path(args.audit)
    store_root.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    user = getpass.getuser()
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    total, kept, skipped = 0, 0, 0
    # Walk files and process only allowed media
    for f in iter_files(src):
        total += 1
        if not should_keep(f):
            skipped += 1
            continue
        # Compute content hash and derive destination folder
        sha = sha256sum(f)
        dest_dir = store_root / sha
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f.name

        action = "copy"
        if dest.exists():
            # Duplicate content (same hash + name); keep a single canonical copy
            action = "duplicate"
        else:
            if args.move:
                shutil.move(str(f), str(dest))
                action = "move"
            else:
                shutil.copy2(f, dest)

        # Stable asset identifier (separate from content hash)
        asset_id = str(uuid.uuid4())
        record = {
            "asset_id": asset_id,
            "when": now_iso,
            "who": user,
            "sha256": sha,
            "action": action,
            "tool_version": TOOL_VERSION,
            "stored_path": f"{sha}/{dest.name}",
            "store_root": str(store_root),
        }
        # Add MIME type (best-effort)
        mime, _ = mimetypes.guess_type(str(dest))
        if not mime:
            ext = dest.suffix.lower()
            ext_map = {
                ".mp4": "video/mp4",
                ".m4v": "video/x-m4v",
                ".webm": "video/webm",
                ".mkv": "video/x-matroska",
                ".mov": "video/quicktime",
                ".avi": "video/x-msvideo",
                ".mpg": "video/mpeg",
                ".mpeg": "video/mpeg",
            }
            mime = ext_map.get(ext)
        if mime:
            record["mime"] = mime
        # Append to audit log (JSON Lines)
        with open(audit_path, "a", encoding="utf-8") as out:
            out.write(json.dumps(record) + "\n")
        kept += 1
        print(f"[{action}] {f.name}  ->  {dest.name}  ({sha[:12]}...)")

    print(f"\nDone. scanned={total} kept={kept} skipped={skipped}")
    print(f"audit log → {audit_path}")

if __name__ == "__main__":
    sys.exit(main())
