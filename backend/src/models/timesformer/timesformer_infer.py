#!/usr/bin/env python3
"""
Run TimeSformer inference over per-shot clips, write JSONL scores.

Inputs:
  --clips        backend/data/derived/clips.jsonl
     rows: { asset_id, sha256, shot_index, clip_uri }
  --clips-root   backend/data/derived/clips
  --checkpoint   backend/models/timesformer_v1.pt   (optional)
  --config       backend/models/timesformer_v1.config.json (optional)
  --out          backend/data/derived/timesformer_scores.jsonl

Output rows (JSONL):
{
  "asset_id": "...",
  "sha256": "...",
  "shot_index": 3,
  "clip_uri": "clips/<sha>/shot_003.mp4",
  "deepfake_score": 0.83,                    # P(FAKE)
  "predicted_label": "FAKE",                 # "FAKE" or "REAL" based on threshold
  "decision_threshold": 0.5,                 # threshold used for prediction
  "model_name": "facebook/timesformer-base-finetuned-k400",
  "model_version": "v1.0",
  "frames": 8,
  "input_size": [224, 224]
}
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List
import warnings

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from tqdm import tqdm
from ...audit import audit_step

# Preferred: decord; fallback: OpenCV
try:
    import decord
    from decord import VideoReader, cpu as decord_cpu
    HAS_DECORD = True
except Exception:
    HAS_DECORD = False
    import cv2

from transformers import AutoImageProcessor, TimesformerModel

DEFAULT_MODEL_ID = "facebook/timesformer-base-finetuned-k400"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

warnings.filterwarnings("ignore", category=UserWarning)

class TSBinary(nn.Module):
    """Wrap TimesformerModel with a binary classification head (FAKE logit)."""
    def __init__(self, base: TimesformerModel, hidden: int):
        super().__init__()
        self.base = base
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden, 1))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.base(pixel_values=pixel_values, return_dict=True)
        # unify with training: use CLS token pooling
        tokens = out.last_hidden_state                  # (B, N, H)
        pooled = tokens[:, 0]                           # CLS token
        logit = self.head(pooled)                       # (B, 1)
        return logit.squeeze(1)                         # (B,)

def pick_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sample_frames_decord(vpath: Path, num: int) -> List[Image.Image]:
    vr = VideoReader(str(vpath), ctx=decord_cpu(0))
    idxs = np.linspace(0, len(vr)-1, num=num, dtype=int)
    frames = vr.get_batch(idxs).asnumpy()  # (num, H, W, 3) RGB
    return [Image.fromarray(fr) for fr in frames]

def sample_frames_cv2(vpath: Path, num: int) -> List[Image.Image]:
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release(); return []
    idxs = np.linspace(0, total-1, num=num, dtype=int)
    out: List[Image.Image] = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok: continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.append(Image.fromarray(frame))
    cap.release()
    return out

def load_clip_frames(vroot: Path, clip_uri: str, frames: int) -> List[Image.Image]:
    # clip_uri is usually "clips/<sha>/shot_003.mp4"
    p = Path(clip_uri)
    rel = p.relative_to("clips") if p.parts and p.parts[0] == "clips" else p
    vpath = vroot / rel
    if HAS_DECORD:
        try:
            imgs = sample_frames_decord(vpath, frames)
            if imgs: return imgs
        except Exception:
            pass
    return sample_frames_cv2(vpath, frames)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="backend/data/derived/clips.jsonl")
    ap.add_argument("--clips-root", default="backend/data/derived/clips")
    ap.add_argument("--checkpoint", default="", help="Optional .pt checkpoint to load")
    ap.add_argument("--config", default="")
    ap.add_argument("--out", default="backend/data/derived/timesformer_scores.jsonl")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    ap.add_argument("--decision-threshold", type=float, default=0.5)
    args = ap.parse_args()

    device = pick_device(args.device)
    model_name = DEFAULT_MODEL_ID
    version = "v1.0"

    # Load config if provided
    cfg = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        model_name = cfg.get("model_name", model_name)
        version = cfg.get("version", version)
        args.frames = cfg.get("frames", args.frames)
        args.size = cfg.get("size", args.size)

    # Base model + processor
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)
    base = TimesformerModel.from_pretrained(model_name, use_safetensors=True)
    hidden = base.config.hidden_size
    model = TSBinary(base, hidden)
    model.eval().to(device)

    # Optional fine-tuned checkpoint
    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ck, strict=False)

    clips_root = Path(args.clips_root)
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)

    # Stream JSONL
    rows = []
    with open(args.clips, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

    def batches(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]
    
    written = 0
    with audit_step("timesformer_infer", params=vars(args), inputs={"clips": args.clips}) as outputs:
        with open(out_path, "w", encoding="utf-8") as fw:
            for chunk in tqdm(list(batches(rows, args.batch_size)), desc="timesformer infer"):
                batch_frames = []   # list of list[ PIL.Image ]
                metas = []
                for r in chunk:
                    imgs = load_clip_frames(clips_root, r.get("clip_uri",""), frames=args.frames)
                    if len(imgs) == 0:
                        # write neutral score if unreadable
                        threshold = args.decision_threshold if args.decision_threshold is not None else cfg.get("decision_threshold", 0.5)
                        predicted_label = "FAKE" if 0.5 >= threshold else "REAL"
                        rec = {
                            "asset_id": r.get("asset_id"),
                            "sha256": r.get("sha256"),
                            "shot_index": r.get("shot_index"),
                            "clip_uri": r.get("clip_uri"),
                            "deepfake_score": 0.5,
                            "predicted_label": predicted_label,
                            "decision_threshold": threshold,
                            "model_name": model_name,
                            "model_version": version,
                            "frames": args.frames,
                            "input_size": [args.size, args.size],
                            "note": "unreadable_clip"
                        }
                        fw.write(json.dumps(rec) + "\n"); written += 1
                        continue
                    # Center-crop/resize handled by processor
                    batch_frames.append(imgs)
                    metas.append(r)

                if not batch_frames:
                    continue

            with torch.no_grad():
                inputs = processor(batch_frames, return_tensors="pt", size={"shortest_edge": args.size})
                pixel_values = inputs["pixel_values"].to(device)   # (B, T, C, H, W)
                logits = model(pixel_values)                       # (B,)
                probs = torch.sigmoid(logits).detach().cpu().numpy().tolist()  # P(FAKE)

            for r, p in zip(metas, probs):
                threshold = args.decision_threshold if args.decision_threshold is not None else cfg.get("decision_threshold", 0.5)
                predicted_label = "FAKE" if p >= threshold else "REAL"
                rec = {
                    "asset_id": r.get("asset_id"),
                    "sha256": r.get("sha256"),
                    "shot_index": r.get("shot_index"),
                    "clip_uri": r.get("clip_uri"),
                    "deepfake_score": float(p),
                    "predicted_label": predicted_label,
                    "decision_threshold": threshold,
                    "model_name": model_name,
                    "model_version": version,
                    "frames": args.frames,
                    "input_size": [args.size, args.size],
                }
                fw.write(json.dumps(rec) + "\n"); written += 1

        outputs["timesformer_scores"] = {"path": args.out}
    print(f"Wrote {written} rows → {out_path}")

if __name__ == "__main__":
    main()
