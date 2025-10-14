#!/usr/bin/env python3
"""
Fine-tune TimeSformer on your labeled assets.

It joins:
  - clips.jsonl  (per-shot clips)
  - manifest.csv (labels: label_num, where by default 0=FAKE, 1=REAL in your pipeline)

Training target = P(FAKE). Uses BCEWithLogitsLoss with pos_weight for class imbalance.

Usage:
  python backend/src/models/timesformer_train.py \
    --clips backend/data/derived/clips.jsonl \
    --clips-root backend/data/derived/clips \
    --manifest backend/data/derived/manifest.csv \
    --out backend/models/timesformer_v1.pt \
    --config backend/models/timesformer_v1.config.json \
    --epochs 10 --batch-size 4 --frames 16 --size 224
"""
from __future__ import annotations
import argparse, csv, json, os, random
from pathlib import Path
from typing import List, Dict, Any
import warnings

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, precision_score, recall_score

# Determinism
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Suppress dynamo errors to fall back to eager mode if compilation fails
import torch._dynamo
torch._dynamo.config.suppress_errors = True

try:
    import decord
    from decord import VideoReader, cpu as decord_cpu
    HAS_DECORD = True
except Exception:
    HAS_DECORD = False
    import cv2

from transformers import AutoImageProcessor, TimesformerModel

DEFAULT_MODEL_ID = "facebook/timesformer-base-finetuned-k400"
warnings.filterwarnings("ignore", category=UserWarning)

# ---------- utils ----------
def pick_device(arg: str) -> torch.device:
    if arg == "cpu":  return torch.device("cpu")
    if arg == "cuda": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_manifest_csv(path: Path) -> Dict[str, Dict[str, Any]]:
    """Return mapping sha256 -> manifest row."""
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            sha = (row.get("sha256") or "").strip()
            if not sha: continue
            lab_raw = (row.get("label_num") or "").strip()
            lab = None
            if lab_raw != "":
                try: lab = int(lab_raw)
                except Exception: lab = None
            out[sha] = {"label_num": lab, "asset_id": row.get("asset_id") or ""}
    return out

def read_clips_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows

def split_by_sha(items, val_frac=0.2, seed=42):
    import random
    by_sha = {}
    for r in items:
        by_sha.setdefault(r["sha256"], []).append(r)
    shas = list(by_sha.keys())
    random.Random(seed).shuffle(shas)
    k = max(1, int(round(val_frac * len(shas))))
    val_shas = set(shas[:k])
    train, val = [], []
    for sha, rows in by_sha.items():
        (val if sha in val_shas else train).extend(rows)
    return train, val

def sample_frames_decord(vpath: Path, num: int) -> List[Image.Image]:
    vr = VideoReader(str(vpath), ctx=decord_cpu(0))
    idxs = np.linspace(0, len(vr)-1, num=num, dtype=int)
    frames = vr.get_batch(idxs).asnumpy()
    return [Image.fromarray(fr) for fr in frames]

def sample_frames_cv2(vpath: Path, num: int) -> List[Image.Image]:
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened(): return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0: cap.release(); return []
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

# ---------- dataset ----------
class ClipsDataset(Dataset):
    """
    Each item is one clip -> (pixel_values tensor, y_fake float)
    y_fake = 1.0 if label_num==0 (FAKE), else 0.0
    """
    def __init__(self, items: List[Dict[str, Any]], clips_root: Path, frames: int, processor):
        self.items = items
        self.clips_root = clips_root
        self.frames = frames
        self.processor = processor

    def __len__(self): return len(self.items)

    def _load_frames(self, clip_uri: str) -> List[Image.Image]:
        p = Path(clip_uri)
        rel = p.relative_to("clips") if p.parts and p.parts[0]=="clips" else p
        vpath = self.clips_root / rel
        if HAS_DECORD:
            try:
                imgs = sample_frames_decord(vpath, self.frames)
                if imgs: return imgs
            except Exception:
                pass
        return sample_frames_cv2(vpath, self.frames)

    def __getitem__(self, idx):
        r = self.items[idx]
        imgs = self._load_frames(r["clip_uri"])
        if not imgs:
            # dummy black frames if unreadable
            imgs = [Image.fromarray(np.zeros((224,224,3), dtype=np.uint8)) for _ in range(self.frames)]
        inputs = self.processor([imgs], return_tensors="pt")  # batch=1
        x = inputs["pixel_values"].squeeze(0)                 # (T, C, H, W)
        # 0=FAKE, 1=REAL  →  y_fake in {1,0}
        y_fake = float(1.0 if r["label_num"] == 0 else 0.0)   # target is FAKE probability
        return x, torch.tensor(y_fake, dtype=torch.float32)

# ---------- model wrapper ----------
class TSBinary(nn.Module):
    def __init__(self, base: TimesformerModel, hidden: int):
        super().__init__()
        self.base = base
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden, 1))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Ensure dict output
        out = self.base(pixel_values=pixel_values, return_dict=True)
        # Use CLS token pooling
        tokens = out.last_hidden_state                            # (B, T*Patches+1, Hdim)
        pooled = tokens[:, 0]                                     # CLS token
        logit = self.head(pooled)                                 # (B, 1)
        return logit.squeeze(1)                                   # (B,)

def split_train_val(items: List[Dict[str,Any]], val_frac=0.2, seed=42):
    random.Random(seed).shuffle(items)
    n = len(items); k = max(1, int(round(val_frac*n)))
    return items[k:], items[:k]

def collate_stack(batch):
    xs, ys = zip(*batch)
    X = torch.stack(xs, dim=0)   # (B, T, C, H, W)
    y = torch.stack(ys, dim=0)   # (B,)
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="backend/data/derived/clips.jsonl")
    ap.add_argument("--clips-root", default="backend/data/derived/clips")
    ap.add_argument("--manifest", default="backend/data/derived/manifest.csv")
    ap.add_argument("--out", default="backend/models/timesformer_v1.pt")
    ap.add_argument("--config", default="backend/models/timesformer_v1.config.json")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = pick_device(args.device)
    scaler = torch.amp.GradScaler(enabled=(device.type=="cuda"))

    manifest = read_manifest_csv(Path(args.manifest))
    clips = read_clips_jsonl(Path(args.clips))

    # Join labels to clips (drop unlabeled)
    items: List[Dict[str,Any]] = []
    for r in clips:
        sha = r.get("sha256")
        if not sha or sha not in manifest: continue
        labnum = manifest[sha].get("label_num")
        if labnum is None: continue
        items.append({
            "asset_id": r.get("asset_id"),
            "sha256": sha,
            "shot_index": r.get("shot_index"),
            "clip_uri": r.get("clip_uri"),
            "label_num": int(labnum),  # 0=FAKE, 1=REAL
        })

    if not items:
        raise SystemExit("No labeled clips found. Ensure manifest.csv has label_num and clips.jsonl matches sha256.")

    train_items, val_items = split_by_sha(items, val_frac=0.2, seed=args.seed)
    print(f"train_clips={len(train_items)}  val_clips={len(val_items)}  unique_sha_train={len({r['sha256'] for r in train_items})}  unique_sha_val={len({r['sha256'] for r in val_items})}")
    print("val class mix:", sum(1 for r in val_items if r['label_num']==0),"FAKE  /", sum(1 for r in val_items if r['label_num']==1),"REAL")

    # Remove quick bug check tiny subset overfit
    # train_items = train_items[:40]

    processor = AutoImageProcessor.from_pretrained(DEFAULT_MODEL_ID, use_fast=True)  # silences the "slow processor" warning
    base = TimesformerModel.from_pretrained(DEFAULT_MODEL_ID, use_safetensors=True)
    hidden = base.config.hidden_size
    model = TSBinary(base, hidden).to(device)
    USE_COMPILE = os.getenv("TORCH_COMPILE", "0") == "1"
    if torch.__version__ >= "2.0" and USE_COMPILE:
        backend = os.getenv("TORCH_BACKEND", "aot_eager")  # 'inductor' needs MSVC
        try:
            model = torch.compile(model, backend=backend)
        except Exception as e:
            print(f"[warn] torch.compile disabled: {e}")

    train_ds = ClipsDataset(train_items, Path(args.clips_root), args.frames, processor)
    val_ds   = ClipsDataset(val_items,   Path(args.clips_root), args.frames, processor)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_stack, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_stack, pin_memory=True)

    # Freeze base initially, train head only (epoch 1..N)
    for p in model.base.parameters(): p.requires_grad = False
    for p in model.head.parameters(): p.requires_grad = True
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)

    # Counts for pos_weight (y=1 is FAKE; label_num 0=FAKE, 1=REAL)
    n_pos = sum(1 for r in train_items if r["label_num"] == 0)
    n_neg = sum(1 for r in train_items if r["label_num"] == 1)
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], device=device, dtype=torch.float32) if n_pos else torch.tensor([1.0], device=device)
    print(f"pos_weight={pos_weight.item():.3f}  (pos=FAKE {n_pos}, neg=REAL {n_neg})")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Confirm trainable params
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: total={total:,} trainable={trainable:,}")

    best_val = -1.0
    for epoch in range(1, args.epochs+1):
        if epoch == 2:
            for p in model.base.parameters(): p.requires_grad = True
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        # ---- train
        model.train()
        tot, n = 0.0, 0
        for X, y in tqdm(train_dl, desc=f"train epoch {epoch}"):
            X = X.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast(
                device_type=("cuda" if device.type=="cuda" else "cpu"),
                enabled=(device.type=="cuda")
            ):
                logits = model(X)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if epoch >= 2:
                scheduler.step()
            tot += loss.item() * X.size(0); n += X.size(0)
        tr_loss = tot / max(1, n)

        # ---- val
        model.eval()
        tot, n = 0.0, 0
        corr = 0
        probs_all = []
        y_all = []
        with torch.no_grad():
            for X, y in tqdm(val_dl, desc=f"val epoch {epoch}"):
                X = X.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
                logits = model(X)
                loss = loss_fn(logits, y)
                tot += loss.item() * X.size(0); n += X.size(0)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                corr += (preds == y).sum().item()
                probs_all.extend(probs.cpu().numpy().tolist())
                y_all.extend(y.cpu().numpy().tolist())
        val_loss = tot / max(1, n)
        val_acc  = corr / max(1, n)

        probs_all = np.array(probs_all)
        y_all = np.array(y_all)

        # default threshold 0.5
        preds05 = (probs_all >= 0.5).astype(np.float32)
        acc05 = (preds05 == y_all).mean()
        try:
            auc = roc_auc_score(y_all, probs_all)
        except Exception:
            auc = float('nan')

        try:
            f1 = f1_score(y_all, preds05)
            prec = precision_score(y_all, preds05)
            rec = recall_score(y_all, preds05)
        except Exception:
            f1 = prec = rec = float('nan')

        # find best threshold for accuracy
        ths = np.linspace(0.05, 0.95, 19)
        accs = [(t, ((probs_all >= t).astype(np.float32) == y_all).mean()) for t in ths]
        best_t, best_acc = max(accs, key=lambda x: x[1])
        cm = confusion_matrix(y_all, preds05, labels=[1, 0])  # FAKE=1, REAL=0 in target space

        print(f"Epoch {epoch}: train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")
        print(f"Val: acc@0.5={acc05:.3f}  AUC={auc:.3f}  F1={f1:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  best_t={best_t:.2f} acc@best={best_acc:.3f}")
        print("Confusion@0.5:\n", cm)

        if val_acc > best_val:
            best_val = val_acc
            Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.out)
            cfg = {
                "model_name": DEFAULT_MODEL_ID,
                "version": "v1.0",
                "frames": args.frames,
                "size": args.size,
                "mean": [0.485, 0.456, 0.406],
                "std":  [0.229, 0.224, 0.225],
                "class_map": {"FAKE": 0, "REAL": 1}, # pipeline convention
                "decision_threshold": float(best_t),
            }
            Path(args.config).parent.mkdir(parents=True, exist_ok=True)
            with open(args.config, "w", encoding="utf-8") as w:
                json.dump(cfg, w, indent=2)
            print(f"Saved checkpoint → {args.out}  and config → {args.config} (val_acc={val_acc:.3f})")

    print("Training done.")

if __name__ == "__main__":
    main()
