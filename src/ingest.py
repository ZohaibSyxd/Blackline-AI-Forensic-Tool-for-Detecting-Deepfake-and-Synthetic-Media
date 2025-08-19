# src/ingest.py
# ingest and move, capturing split
# python .\src\ingest.py ".\tests" --store ".\data\raw" --move


import argparse, hashlib, json, os, shutil, sys, time, socket, getpass, mimetypes
from pathlib import Path

ALLOWED_EXTS = {".jpg",".jpeg",".png",".gif",".bmp",".tiff",".mp4",".avi",".mov",".mkv",".webm"}

def sha256sum(path, chunk_size=1024*1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def iter_files(path: Path):
    if path.is_file():
        yield path
    else:
        for p in path.rglob("*"):
            if p.is_file():
                yield p

def should_keep(p: Path):
    return p.suffix.lower() in ALLOWED_EXTS

def detect_split(src_root: Path, f: Path) -> str:
    # infer from relative path parts
    try:
        parts = [p.lower() for p in f.relative_to(src_root).parts]
    except ValueError:
        parts = [p.lower() for p in f.parts]
    if "train" in parts: return "train"
    if "test"  in parts: return "test"
    if any(x in parts for x in ("val","valid","validation")): return "val"
    return "unknown"

def main():
    ap = argparse.ArgumentParser(description="Immutable ingest with SHA-256 + audit log")
    ap.add_argument("src", help="file or directory to ingest")
    ap.add_argument("--store", default="data/raw", help="where to store raw files")
    ap.add_argument("--audit", default="data/audit/ingest_log.jsonl", help="audit log path")
    ap.add_argument("--move", action="store_true", help="move files instead of copy (saves disk)")
    args = ap.parse_args()

    src = Path(args.src)
    store_root = Path(args.store)
    audit_path = Path(args.audit)
    store_root.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    user = getpass.getuser()
    host = socket.gethostname()
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    total, kept, skipped = 0, 0, 0
    for f in iter_files(src):
        total += 1
        if not should_keep(f):
            skipped += 1
            continue
        sha = sha256sum(f)
        dest_dir = store_root / sha
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f.name

        action = "copy"
        if dest.exists():
            action = "duplicate"
        else:
            if args.move:
                shutil.move(str(f), str(dest))
                action = "move"
            else:
                shutil.copy2(f, dest)

        stat = (dest if dest.exists() else f).stat()  # size after move/copy, else original
        record = {
            "when": now_iso,
            "who": user,
            "host": host,
            "src_path": str(f.resolve()),
            "src_rel": str(f.relative_to(src)) if f.is_relative_to(src) else None,
            "stored_path": str(dest.resolve()),
            "sha256": sha,
            "split": detect_split(src, f),
            "size_bytes": stat.st_size,
            "mtime": int(stat.st_mtime),
            "mime": mimetypes.guess_type(f.name)[0],
            "action": action,
            "tool_version": "ingest.v2",
        }
        with open(audit_path, "a", encoding="utf-8") as out:
            out.write(json.dumps(record) + "\n")
        kept += 1
        print(f"[{action}] {f}  ->  {dest}  ({sha[:12]}...)")

    print(f"\nDone. scanned={total} kept={kept} skipped={skipped}")
    print(f"audit log → {audit_path}")

if __name__ == "__main__":
    sys.exit(main())
