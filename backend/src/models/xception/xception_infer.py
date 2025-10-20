import argparse, json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import torch
import timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from ...audit import audit_step

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

xception_tf = transforms.Compose([
    transforms.Resize((299, 299)),   # idempotent for your 299x299 frames
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def resolve_path(rel_or_abs: str, root: Path) -> Optional[Path]:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p if p.exists() else None
    p = (root / p).resolve()
    return p if p.exists() else None

class FrameRecord:
    __slots__ = ("path","asset_id","sha256","shot_index","frame_index")
    def __init__(self, path: Path, row: Dict[str, Any]):
        self.path = path
        self.asset_id = row.get("asset_id")
        self.sha256 = row.get("sha256")
        self.shot_index = row.get("shot_index")
        self.frame_index = row.get("frame_index")

class FramesJsonlDataset(Dataset):
    def __init__(self, frames_jsonl: Path, root: Path, use_faces: bool = False, every: int = 1, limit: Optional[int] = None):
        self.items: List[FrameRecord] = []
        with open(frames_jsonl, "r", encoding="utf-8") as fr:
            for i, line in enumerate(fr):
                if every > 1 and (i % every) != 0:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                # prefer normalized_uri; fallback to uri
                rels: List[str] = []
                if use_faces and row.get("face_uris"):
                    rels.extend(row["face_uris"])
                else:
                    if row.get("normalized_uri"):
                        rels.append(row["normalized_uri"])
                    elif row.get("uri"):
                        rels.append(row["uri"])
                for rel in rels:
                    p = resolve_path(rel, root)
                    if p:
                        self.items.append(FrameRecord(p, row))
                if limit and len(self.items) >= limit:
                    break

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, FrameRecord]:
        rec = self.items[idx]
        img = Image.open(rec.path).convert("RGB")
        x = xception_tf(img)
        return x, rec

# NEW: custom collate so we can keep FrameRecord objects in a list
def collate_batch(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)  # B,C,H,W
    recs = [b[1] for b in batch]                    # list[FrameRecord]
    return xs, recs

class XceptionDeepfakeDetector:
    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # Binary head (1 logit). If a fine-tuned checkpoint is provided, avoid downloading
        # ImageNet weights by constructing the model with pretrained=False.
        pretrained_flag = False if checkpoint_path else True
        self.model = timm.create_model("legacy_xception", pretrained=pretrained_flag, num_classes=1)
        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict_batch(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x.to(self.device, non_blocking=True))
        return torch.sigmoid(logits).squeeze(1).float()  # (B,)

def aggregate_shot(scores: List[Dict[str, Any]], thr: float) -> List[Dict[str, Any]]:
    # mean per (asset_id, shot_index)
    agg: Dict[Tuple[Any, Any], List[float]] = {}
    for r in scores:
        key = (r.get("asset_id"), r.get("shot_index"))
        agg.setdefault(key, []).append(r["deepfake_score"])
    out = []
    for (asset_id, shot_idx), vals in agg.items():
        agg_score = float(sum(vals)/len(vals))
        out.append({
            "asset_id": asset_id,
            "shot_index": shot_idx,
            "model": "xception_timm",
            "deepfake_score": agg_score,
            "frames": len(vals),
            "predicted_label": "FAKE" if agg_score >= thr else "REAL",
            "decision_threshold": thr,
        })
    return out

def main():
    ap = argparse.ArgumentParser("Xception deepfake detector (timm)")
    ap.add_argument("--frames-jsonl", required=True, help="frames_normalized.jsonl or frames.jsonl")
    ap.add_argument("--root", required=True, help="Root folder for images (e.g., backend/data/derived)")
    ap.add_argument("--out", required=True, help="Output JSONL of per-frame scores")
    ap.add_argument("--checkpoint", default=None, help="Deepfake-trained Xception checkpoint (.pth)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0)  # default 0 for Windows/CPU
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--every", type=int, default=1, help="Row stride (subsample)")
    ap.add_argument("--use-faces", action="store_true", help="Score face_uris instead of full frames when available")
    ap.add_argument("--aggregate-shot-out", dest="aggregate_shot_out", default=None,
                    help="Optional shot-level aggregate JSONL")
    ap.add_argument("--device", default=None, help="cuda|cpu (auto if not set)")
    ap.add_argument("--decision-threshold", type=float, default=0.5)
    args = ap.parse_args()

    ds = FramesJsonlDataset(Path(args.frames_jsonl), Path(args.root), use_faces=args.use_faces, every=args.every, limit=args.limit)
    if len(ds) == 0:
        print("No frames found. Check --frames-jsonl and --root.")
        return

    det = XceptionDeepfakeDetector(checkpoint_path=args.checkpoint, device=args.device)

    pin_mem = det.device.type == "cuda"
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        shuffle=False,
        collate_fn=collate_batch,   # NEW
    )
    
    per_frame: List[Dict[str, Any]] = []
    with audit_step("xception_infer", params=vars(args), inputs={"frames_jsonl": args.frames_jsonl}) as outputs:
        with open(args.out, "w", encoding="utf-8") as fw:
            for xb, recs in tqdm(dl, desc="xception infer"):
                probs = det.predict_batch(xb).cpu().tolist()
                for p, rec in zip(probs, recs):
                    thr = args.decision_threshold
                    row = {
                        "asset_id": rec.asset_id,
                        "sha256": rec.sha256,
                        "shot_index": rec.shot_index,
                        "frame_index": rec.frame_index,
                        "image_path": str(rec.path),
                        "model": "xception_timm",
                        "deepfake_score": float(p),
                        "predicted_label": "FAKE" if p >= thr else "REAL",
                        "decision_threshold": thr,
                    }
                    per_frame.append(row)
                    fw.write(json.dumps(row) + "\n")

        outputs["xception_scores_frames"] = {"path": args.out}
    if args.aggregate_shot_out:
        shots = aggregate_shot(per_frame, args.decision_threshold)
        out_p = Path(args.aggregate_shot_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as fw:
            for r in shots:
                fw.write(json.dumps(r) + "\n")
    # Optional aggregate output logged separately if needed

    print(f"Scored {len(per_frame)} images → {args.out}")
    if args.aggregate_shot_out:
        print(f"Wrote shot aggregates → {args.aggregate_shot_out}")

if __name__ == "__main__":
    main()