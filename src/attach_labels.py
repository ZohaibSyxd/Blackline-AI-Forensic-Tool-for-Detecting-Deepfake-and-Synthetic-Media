# src/attach_labels.py
import argparse, csv, json
from pathlib import Path

def load_meta(path):
    meta = json.load(open(path, encoding="utf-8"))
    mapping = {}
    # Common DFDC-style: { "video.mp4": { "label": "FAKE", ... }, ... }
    if isinstance(meta, dict):
        for k, v in meta.items():
            if isinstance(v, dict):
                lab = v.get("label", v.get("Label", v.get("LABEL")))
            else:
                lab = v
            if lab is None: 
                continue
            mapping[Path(k).name.lower()] = str(lab).strip().lower()
        return mapping
    # Fallback: list of {filename,label}
    if isinstance(meta, list):
        for row in meta:
            fn = row.get("filename") or row.get("file") or row.get("name")
            lab = row.get("label")
            if fn and lab:
                mapping[Path(fn).name.lower()] = str(lab).strip().lower()
        return mapping
    raise ValueError("Unrecognized metadata.json shape")

def norm_label(s):
    s = s.lower()
    if s in {"real","genuine","true","authentic"}: return "REAL"
    if s in {"fake","manipulated","deepfake","synthetic"}: return "FAKE"
    return s.upper()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/derived/manifest.csv")
    ap.add_argument("--meta", required=True, help="path to metadata.json")
    ap.add_argument("--out", default="data/derived/manifest_labeled.csv")
    args = ap.parse_args()

    meta = load_meta(args.meta)
    hits = 0
    rows = []
    with open(args.manifest, newline="", encoding="utf-8") as r:
        reader = csv.DictReader(r)
        fieldnames = reader.fieldnames + (["label"] if "label" not in reader.fieldnames else [])
        for row in reader:
            key = Path(row.get("orig_name","")).name.lower()
            lab = meta.get(key)
            row["label"] = norm_label(lab) if lab else ""
            hits += 1 if lab else 0
            rows.append(row)

    Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as w:
        writer = csv.DictWriter(w, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)

    print(f"Attached labels for {hits}/{len(rows)} files → {args.out}")

if __name__ == "__main__":
    main()
