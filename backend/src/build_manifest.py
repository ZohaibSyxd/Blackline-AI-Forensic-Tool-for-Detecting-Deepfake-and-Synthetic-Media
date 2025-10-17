"""
src/build_manifest.py

Build a single manifest CSV from the ingest audit log, merging optional labels
and splits from metadata.json and optional technical fields from
probe/validate outputs.

Usage (PowerShell)
        python .\src\build_manifest.py \
                --audit .\data\audit\ingest_log.jsonl \
                --out .\data\derived\manifest.csv \
                --meta .\datasets\train\metadata.json \
                --probe .\data\derived\probe.jsonl \
                --validate .\data\derived\validate.jsonl

Notes
- The audit log is JSONL written by ingest.py (one row per kept asset).
- Labels and splits are looked up by original filename (case-insensitive) if provided.
- If probe/validate JSONL files exist, we merge a compact summary of fields
    such as width/height/fps/codec/duration_s and format/decode flags.
- For a video-only pipeline, ensure ingest filters non-video types; this script
    will naturally include only what ingest kept.
"""

import argparse, csv, json
from pathlib import Path
from .audit import audit_step

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

def norm_split(s):
    """Normalize split values to one of train/test/val/unknown."""
    if s is None:
        return ""
    s = str(s).strip().lower()
    if s in {"train","training"}:
        return "train"
    if s in {"test","testing"}:
        return "test"
    if s in {"val","valid","validn","validation","dev","devval"}:
        return "val"
    return s

def norm_label(s):
    """Normalize label to 'REAL'/'FAKE' (accepts strings or 0/1)."""
    if s is None:
        return ""
    txt = str(s).strip().lower()
    if txt in {"real","genuine","true","1"}:
        return "REAL"
    if txt in {"fake","manipulated","deepfake","synthetic","0"}:
        return "FAKE"
    return txt.upper()

def label_to_num(label: str):
    """Map label strings to numeric codes: REAL->1, FAKE->0, else empty string."""
    if not label:
        return ""
    if label == "REAL":
        return 1
    if label == "FAKE":
        return 0
    return ""

def load_meta(meta_path: str) -> dict[str, dict]:
    if not meta_path:
        return {}
    p = Path(meta_path)
    if not p.exists():
        print(f"Meta not found: {p}")
        return {}
    data = json.load(open(p, encoding="utf-8"))
    mapping: dict[str, dict] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            if not isinstance(v, dict):
                # Assume bare label value
                mapping[Path(k).name.lower()] = {
                    "label": norm_label(v),
                    "split": "",
                }
                continue
            lab = v.get("label")
            spl = v.get("split")
            mapping[Path(k).name.lower()] = {
                "label": norm_label(lab),
                "split": norm_split(spl),
            }
    elif isinstance(data, list):
        for row in data:
            fn = row.get("filename") or row.get("file") or row.get("name")
            lab = row.get("label")
            spl = row.get("split")
            if fn:
                mapping[Path(fn).name.lower()] = {
                    "label": norm_label(lab),
                    "split": norm_split(spl),
                }
    else:
        print("Unrecognized metadata.json shape; skipping labels.")
    return mapping

def load_probe_summary(probe_path: str) -> dict[str, dict]:
    """Load probe.jsonl into mapping by sha256 -> summary fields.

    Also carry through asset_id if present (keyed by sha256 for now).
    """
    if not probe_path:
        return {}
    p = Path(probe_path)
    if not p.exists():
        return {}
    out: dict[str, dict] = {}
    with open(p, encoding="utf-8") as r:
        for line in r:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            sha = rec.get("sha256")
            if not sha:
                continue
            summ = rec.get("summary") or {}
            out[sha] = {
                "asset_id": rec.get("asset_id"),
                "width": summ.get("width"),
                "height": summ.get("height"),
                "fps": summ.get("fps"),
                "codec": summ.get("codec"),
                "duration_s": summ.get("duration_s"),
                "nb_streams": summ.get("nb_streams"),
            }
    return out

def load_validate_summary(validate_path: str) -> dict[str, dict]:
    """Load validate.jsonl into mapping by sha256 -> format/decode + dims/duration."""
    if not validate_path:
        return {}
    p = Path(validate_path)
    if not p.exists():
        return {}
    out: dict[str, dict] = {}
    with open(p, encoding="utf-8") as r:
        for line in r:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            sha = rec.get("sha256")
            if not sha:
                continue
            out[sha] = {
                "asset_id": rec.get("asset_id"),
                "format_valid": rec.get("format_valid"),
                "decode_valid": rec.get("decode_valid"),
                "val_width": rec.get("width"),
                "val_height": rec.get("height"),
                "val_duration_s": rec.get("duration_s"),
            }
    return out

def main():
    ap = argparse.ArgumentParser(description="Build single manifest CSV (labels + probe/validate summaries if provided).")
    ap.add_argument("--audit", default="backend/data/audit/ingest_log.jsonl", help="audit log path")
    ap.add_argument("--out", default="backend/data/derived/manifest.csv", help="output CSV")
    ap.add_argument("--meta", default=None, help="path to metadata.json with labels (optional)")
    ap.add_argument("--probe", default="backend/data/derived/probe.jsonl", help="path to probe.jsonl (optional)")
    ap.add_argument("--validate", default="backend/data/derived/validate.jsonl", help="path to validate.jsonl (optional)")
    args = ap.parse_args()

    Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
    meta_map = load_meta(args.meta) if args.meta else {}
    probe_map = load_probe_summary(args.probe) if args.probe else {}
    val_map = load_validate_summary(args.validate) if args.validate else {}

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
            meta = meta_map.get(orig_name.lower(), {})
            lab = meta.get("label", "") if isinstance(meta, dict) else str(meta)
            if lab:
                label_hits += 1
            split = meta.get("split") if isinstance(meta, dict) else ""
            split = split or infer_split_from_paths(rec)
            # lookups by sha256 for probe/validate summaries
            sha = rec.get("sha256")
            ps = probe_map.get(sha, {})
            vs = val_map.get(sha, {})
            asset_id = ps.get("asset_id") or vs.get("asset_id") or rec.get("asset_id")
            rows.append({
                "asset_id": asset_id,
                "sha256": sha,
                "split": split,
                "stored_path": rec["stored_path"],
                "store_root": rec.get("store_root"),
                "orig_name": orig_name,
                "size_bytes": rec.get("size_bytes"),
                "mime": rec.get("mime"),
                # labels
                "label": lab,
                "label_num": label_to_num(lab),
                # probe summary
                "width": ps.get("width"),
                "height": ps.get("height"),
                "fps": ps.get("fps"),
                "codec": ps.get("codec"),
                "duration_s": ps.get("duration_s"),
                "nb_streams": ps.get("nb_streams"),
                # validate summary
                "format_valid": vs.get("format_valid"),
                "decode_valid": vs.get("decode_valid"),
                "val_width": vs.get("val_width"),
                "val_height": vs.get("val_height"),
                "val_duration_s": vs.get("val_duration_s"),
            })

    if not rows:
        print("No rows found. Did you ingest anything?")
        return

    with audit_step("build_manifest", params=vars(args), inputs={"audit": args.audit, "meta": args.meta or "", "probe": args.probe or "", "validate": args.validate or ""}) as outputs:
        with open(args.out, "w", newline="", encoding="utf-8") as w:
            writer = csv.DictWriter(w, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        outputs["manifest"] = {"path": args.out}

    print(f"Wrote {len(rows)} rows -> {args.out}")
    if args.meta:
        print(f"Attached labels for {label_hits}/{len(rows)} files")

if __name__ == "__main__":
    main()
