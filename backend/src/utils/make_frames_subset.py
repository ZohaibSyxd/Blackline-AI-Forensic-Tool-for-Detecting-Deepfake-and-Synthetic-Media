"""
Create a subset of frames.jsonl with up to N frames per asset (sha256).

Usage:
  python backend/src/utils/make_frames_subset.py \
    --frames backend/data/derived/frames.jsonl \
    --out backend/data/derived/frames_subset.jsonl \
    --per-asset 1 \
    [--manifest backend/data/derived/manifest.csv] \
    [--assets-limit 1000]

This lets you run compute_noise.py quickly over representative frames without long waits.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set


def load_manifest_assets(path: Optional[str]) -> Optional[Set[str]]:
    if not path:
        return None
    assets: Set[str] = set()
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                sha = row.get("sha256")
                if sha:
                    assets.add(sha)
    except Exception:
        return None
    return assets


def iter_frames(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            yield row


def main() -> None:
    ap = argparse.ArgumentParser(description="Create per-asset frames subset JSONL")
    ap.add_argument("--frames", required=True, help="Path to full frames.jsonl")
    ap.add_argument("--out", required=True, help="Path to write subset frames.jsonl")
    ap.add_argument("--per-asset", type=int, default=1, help="Max frames per asset (sha256)")
    ap.add_argument("--manifest", default=None, help="Optional manifest.csv to restrict the asset set")
    ap.add_argument("--assets-limit", type=int, default=None, help="Optional cap on number of distinct assets to include")
    args = ap.parse_args()

    manifest_assets = load_manifest_assets(args.manifest)
    per_asset = max(1, int(args.per_asset))
    assets_limit = args.assets_limit

    counts: Dict[str, int] = {}
    selected_assets: Set[str] = set()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(out_path, "w", encoding="utf-8") as w:
        for row in iter_frames(args.frames):
            sha = row.get("sha256")
            if not sha:
                continue
            if manifest_assets is not None and sha not in manifest_assets:
                continue
            if assets_limit is not None and len(selected_assets) >= assets_limit and sha not in selected_assets:
                continue
            cnt = counts.get(sha, 0)
            if cnt >= per_asset:
                continue
            # accept
            counts[sha] = cnt + 1
            selected_assets.add(sha)
            w.write(json.dumps(row) + "\n")
            written += 1

    print(f"Wrote subset: {written} frames across {len(selected_assets)} assets -> {out_path}")


if __name__ == "__main__":
    main()
