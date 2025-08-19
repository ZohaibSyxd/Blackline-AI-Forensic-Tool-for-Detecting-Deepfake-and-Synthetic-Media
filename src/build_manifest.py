# src/build_manifest.py

# build a manifest CSV from the audit log
# python .\src\build_manifest.py
import csv, json
from pathlib import Path

def infer_split_from_paths(rec):
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

def main(audit="data/audit/ingest_log.jsonl", out_csv="data/derived/manifest.csv"):
    Path("data/derived").mkdir(parents=True, exist_ok=True)
    seen = set()
    rows = []
    with open(audit, encoding="utf-8") as r:
        for line in r:
            rec = json.loads(line)
            key = rec.get("stored_path")
            if not key or key in seen:
                continue
            seen.add(key)
            rows.append({
                "sha256": rec["sha256"],
                "split": infer_split_from_paths(rec),
                "stored_path": rec["stored_path"],
                "orig_name": Path(rec["stored_path"]).name,
                "size_bytes": rec.get("size_bytes"),
                "mime": rec.get("mime"),
            })
    if not rows:
        print("No rows found. Did you ingest anything?")
        return
    with open(out_csv, "w", newline="", encoding="utf-8") as w:
        writer = csv.DictWriter(w, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {out_csv}")

if __name__ == "__main__":
    main()
