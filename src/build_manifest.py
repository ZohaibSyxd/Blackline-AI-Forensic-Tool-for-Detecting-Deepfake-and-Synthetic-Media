"""
src/build_manifest.py

Build a single manifest CSV from the ingest audit log, optionally merging labels
from a metadata file.

Usage (PowerShell)
    python .\src\build_manifest.py --audit .\data\audit\ingest_log.jsonl \
                 --out .\data\derived\manifest.csv --meta .\datasets\train\metadata.json

Notes
- The audit log is JSONL written by ingest.py (one row per kept asset).
- Labels are looked up by original filename (case-insensitive) if provided.
"""

import argparse, csv, json
from pathlib import Path

def infer_split_from_paths(rec):
    """Infer dataset split from audit record fields or known path hints."""
    s = (rec.get("split") or "").lower()
    if s in {"train","test","val"}:
        return s
    # fallback from paths if split missing
    for key in ("src_rel","src_path","stored_path"):
        p = (rec.get(key) or "").lower()
        if "/train/" in p or "\\train\\" in p: return "train"
        if "/test/"  in p or "\\test\\"  in p: return "test"
        if any(x in p for x in ("/val/","\\val\\","validation")): return "val"
    return "unknown"

def norm_label(s):
    """Normalize label values as uppercase strings (REAL/FAKE) by default.
    """
    if s is None:
        return ""
        s = str(s).strip().lower()
    if s in {"real","genuine","true","authentic"}:
        return "REAL"
    if s in {"fake","manipulated","deepfake","synthetic"}:
        return "FAKE"
    return s.upper()

def load_meta(meta_path: str) -> dict[str, str]:
    if not meta_path:
        return {}
    p = Path(meta_path)
    if not p.exists():
        print(f"Meta not found: {p}")
        return {}
    data = json.load(open(p, encoding="utf-8"))
    mapping = {}
    if isinstance(data, dict):
        for k, v in data.items():
            lab = v.get("label") if isinstance(v, dict) else v
            if lab is None:
                continue
            mapping[Path(k).name.lower()] = norm_label(lab)
    elif isinstance(data, list):
        for row in data:
            fn = row.get("filename") or row.get("file") or row.get("name")
            lab = row.get("label")
            if fn and lab:
                mapping[Path(fn).name.lower()] = norm_label(lab)
    else:
        print("Unrecognized metadata.json shape; skipping labels.")
    return mapping

def main():
    ap = argparse.ArgumentParser(description="Build single manifest CSV (with labels if provided).")
    ap.add_argument("--audit", default="data/audit/ingest_log.jsonl", help="audit log path")
    ap.add_argument("--out", default="data/derived/manifest.csv", help="output CSV")
    ap.add_argument("--meta", default=None, help="path to metadata.json with labels (optional)")
    args = ap.parse_args()

    Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
    label_map = load_meta(args.meta) if args.meta else {}

    seen = set()
    rows = []
    label_hits = 0

    with open(args.audit, encoding="utf-8") as r:
        for line in r:
            rec = json.loads(line)
            key = rec.get("stored_path")
            if not key or key in seen:
                continue
            seen.add(key)
            orig_name = Path(rec["stored_path"]).name
            lab = label_map.get(orig_name.lower(), "")
            if lab:
                label_hits += 1
            rows.append({
                "sha256": rec["sha256"],
                "split": infer_split_from_paths(rec),
                "stored_path": rec["stored_path"],
                "orig_name": orig_name,
                "size_bytes": rec.get("size_bytes"),
                "mime": rec.get("mime"),
                "label": lab,  # unified manifest includes labels here
            })

    if not rows:
        print("No rows found. Did you ingest anything?")
        return

    with open(args.out, "w", newline="", encoding="utf-8") as w:
        writer = csv.DictWriter(w, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {args.out}")
    if args.meta:
        print(f"Attached labels for {label_hits}/{len(rows)} files")

if __name__ == "__main__":
    main()
