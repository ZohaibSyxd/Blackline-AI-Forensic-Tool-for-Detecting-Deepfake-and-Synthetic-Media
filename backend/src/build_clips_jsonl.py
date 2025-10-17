#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from .audit import audit_step

def main():
    ap = argparse.ArgumentParser("Build clips.jsonl from shots.jsonl and a clips folder")
    ap.add_argument("--shots", default="backend/data/derived/shots.jsonl")
    ap.add_argument("--clips-root", default="backend/data/derived/clips")
    ap.add_argument("--out", default="backend/data/derived/clips.jsonl")
    args = ap.parse_args()

    shots = Path(args.shots)
    clips_root = Path(args.clips_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_in = n_written = n_missing = 0
    with audit_step("build_clips", params=vars(args), inputs={"shots": args.shots}) as outputs:
        with shots.open(encoding="utf-8") as rs, out.open("w", encoding="utf-8") as w:
            for line in rs:
                if not line.strip():
                    continue
                r = json.loads(line)
                sha = r["sha256"]
                i = int(r["shot_index"])
                rel = f"clips/{sha}/shot_{i:03d}.mp4"
                abs_p = clips_root / sha / f"shot_{i:03d}.mp4"
                n_in += 1
                if abs_p.exists():
                    w.write(json.dumps({
                        "asset_id": r.get("asset_id"),
                        "sha256": sha,
                        "shot_index": i,
                        "clip_uri": rel
                    }) + "\n")
                    n_written += 1
                else:
                    n_missing += 1

        outputs["clips"] = {"path": args.out}
    print(f"Wrote {n_written} rows → {out} (missing clips: {n_missing} / shots: {n_in})")

if __name__ == "__main__":
    main()
