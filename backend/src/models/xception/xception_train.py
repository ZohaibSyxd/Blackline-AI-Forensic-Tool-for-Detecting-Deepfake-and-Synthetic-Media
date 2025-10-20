#!/usr/bin/env python3
"""
Fast fine-tuning of Xception for Deepfake Detection (improved)
-------------------------------------------------------------
- Mixed Precision (AMP, modern API)
- Balanced batches (WeightedRandomSampler) + pos_weight BCE
- Two-phase fine-tune (head warmup → full model, param-group LRs)
- Cosine LR with warmup
- Forensics-friendly augmentations (JPEG jitter, light blur, mild color jitter)
- Channels-last + high matmul precision
- Early stopping on AUC (configurable)
- Validation: per-frame + per-asset (aggregated by --id-col), threshold sweep
- Saves best checkpoint with thresholds in the file

Outputs:
  checkpoint["meta"] = {
    "monitor": str, "best_metric": float,
    "best_threshold_frame": float, "best_threshold_asset": float,
    "model_name": "legacy_xception",
    "imagenet_norm": True,
    "id_col": args.id_col
  }
"""

import argparse, json, random, os, math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image, ImageFile, ImageFilter
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix

# --- Stable, reproducible-ish but fast ---
os.environ.setdefault("PYTHONUNBUFFERED", "1")
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# -------- Forensics-friendly augmentations --------
class RandomJPEGCompression:
    """Round-trip through JPEG at a random quality [min_q, max_q]."""
    def __init__(self, min_q=60, max_q=95, p=0.5):
        self.min_q, self.max_q, self.p = min_q, max_q, p
    def __call__(self, img: Image.Image):
        if random.random() > self.p:
            return img
        from io import BytesIO
        q = random.randint(self.min_q, self.max_q)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

class RandomGaussianBlur:
    def __init__(self, max_sigma=0.8, p=0.2):
        self.max_sigma, self.p = max_sigma, p
    def __call__(self, img: Image.Image):
        if random.random() > self.p:
            return img
        sigma = random.random() * self.max_sigma
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))

train_tf = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomResizedCrop(299, scale=(0.9, 1.0)),
    transforms.RandomHorizontalFlip(),
    RandomJPEGCompression(60, 95, p=0.6),
    RandomGaussianBlur(max_sigma=0.8, p=0.25),
    transforms.ColorJitter(brightness=0.08, contrast=0.08),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_tf = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# --- Helpers ---
def _norm_label(v: Any) -> Optional[int]:
    if v is None: return None
    s = str(v).strip().lower()
    if s in {"fake", "deepfake"}: return 0
    if s in {"real", "genuine"}:  return 1
    try:
        n = int(float(s))
        if n in (0, 1): return n
    except Exception:
        pass
    return None

def _resolve_path(rel_or_abs: str, root: Path) -> Optional[Path]:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p if p.exists() else None
    p = (root / p).resolve()
    return p if p.exists() else None

# --- Dataset ---
class FramesWithLabels(Dataset):
    """
    Returns (tensor, y_fake, id_str)
      - y_fake: 1 for FAKE, 0 for REAL  (we invert original labels)
    """
    def __init__(self, rows: List[Dict[str, Any]], root: Path, tf, id_col: str):
        self.items: List[Tuple[Path, int, str]] = []
        self.tf = tf
        self.id_col = id_col
        for r in rows:
            rel = r.get("normalized_uri") or r.get("uri")
            if not rel: continue
            p = _resolve_path(rel, root)
            if not p: continue
            y_orig = r.get("_label")
            if y_orig is None: continue  # 0=fake,1=real (orig)
            # invert so that y_fake=1 means FAKE (positive)
            y_fake = 1 - int(y_orig)
            self.items.append((p, y_fake, str(r[id_col])))

    def __len__(self): return len(self.items)
    def __getitem__(self, idx: int):
        path, y_fake, id_str = self.items[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (299, 299))
        x = self.tf(img)
        return x, torch.tensor(y_fake, dtype=torch.float32), id_str

def load_rows_with_labels(frames_jsonl: Path, label_by: Dict[str, int], id_col: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(frames_jsonl, "r", encoding="utf-8") as fr:
        for line in fr:
            try:
                row = json.loads(line)
            except Exception:
                continue
            key = row.get(id_col)
            if key is None: continue
            key = str(key)
            if key in label_by:
                row["_label"] = label_by[key]  # 0=fake,1=real
                rows.append(row)
    return rows

def split_by_id(rows: List[Dict[str, Any]], id_col: str, val_frac: float, seed: int = 42):
    ids = {}
    for r in rows:
        ids.setdefault(r[id_col], 0)
    id_list = list(ids.keys())
    random.Random(seed).shuffle(id_list)
    val_n = max(1, int(len(id_list)*val_frac))
    val_ids = set(id_list[:val_n])
    train, val = [], []
    for r in rows:
        (val if r[id_col] in val_ids else train).append(r)
    return train, val

def build_balanced_sampler(dataset: FramesWithLabels) -> WeightedRandomSampler:
    labels = [lbl for _, lbl, _ in dataset.items]  # y_fake (1=FAKE)
    labels_t = torch.tensor(labels, dtype=torch.long)
    counts = torch.bincount(labels_t, minlength=2)  # [REAL_count, FAKE_count]
    class_weight = torch.zeros(2, dtype=torch.float32)
    # weight inversely proportional to class frequency
    for c in (0,1):
        class_weight[c] = (counts.sum() / (2.0 * max(1, counts[c])))
    sample_weight = class_weight[labels_t]
    return WeightedRandomSampler(weights=sample_weight, num_samples=len(labels), replacement=True)

def balanced_accuracy_from_cm(cm: np.ndarray) -> float:
    # cm = [[tn, fp],[fn, tp]]
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / max(1, tp + fn)
    tnr = tn / max(1, tn + fp)
    return 0.5*(tpr + tnr)

def sweep_best_threshold(y_true: np.ndarray, probs: np.ndarray, metric="bal_acc") -> Tuple[float, Dict[str, float]]:
    # thresholds from unique probs plus edges
    unique = np.unique(probs)
    if unique.size > 1024:
        # sample to keep it fast
        unique = np.quantile(unique, np.linspace(0,1,1025))
    thresholds = np.concatenate(([0.0], unique, [1.0]))
    best_thr, best_val, best_cm = 0.5, -1.0, None
    for thr in thresholds:
        preds = (probs >= thr).astype(np.float32)
        cm = confusion_matrix(y_true, preds, labels=[0,1])  # [[tn,fp],[fn,tp]]
        if metric == "f1":
            prec = precision_score(y_true, preds, zero_division=0)
            rec  = recall_score(y_true, preds, zero_division=0)
            val = 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)
        elif metric == "acc":
            val = (cm[0,0] + cm[1,1]) / max(1, cm.sum())
        else:  # balanced accuracy
            val = balanced_accuracy_from_cm(cm)
        if val > best_val:
            best_val, best_thr, best_cm = val, thr, cm
    return float(best_thr), {
        "metric_value": float(best_val),
        "tn": int(best_cm[0,0]), "fp": int(best_cm[0,1]),
        "fn": int(best_cm[1,0]), "tp": int(best_cm[1,1]),
    }

# --- Main ---
def main():
    ap = argparse.ArgumentParser("Fast fine-tune Xception for Deepfake Detection (improved)")
    ap.add_argument("--frames-jsonl", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--id-col", default="sha256")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--out", default="backend/models/xception_v1.pth")
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--head-warmup-epochs", type=int, default=1)
    ap.add_argument("--monitor", choices=["auc","bal_acc","acc"], default="auc")
    ap.add_argument("--agg", choices=["median","mean","p95","max"], default="median",
                    help="Asset-level aggregation over frame probabilities")
    ap.add_argument("--no-balanced-sampler", action="store_true",
                    help="Disable WeightedRandomSampler; fall back to shuffle=True")
    args = ap.parse_args()

    root = Path(args.root)
    df = pd.read_csv(args.manifest)

    # Map id -> {0,1} where 0=fake, 1=real (original), we will invert later for y_fake
    label_by: Dict[str, int] = {}
    for _, row in df.iterrows():
        k = row.get(args.id_col)
        lab = _norm_label(row.get(args.label_col))
        if pd.isna(k) or lab is None: continue
        label_by[str(k)] = lab

    rows = load_rows_with_labels(Path(args.frames_jsonl), label_by, args.id_col)
    if not rows:
        print("No labeled frames found. Check --id-col/--label-col and inputs.")
        return
    train_rows, val_rows = split_by_id(rows, args.id_col, args.val_frac)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_float32_matmul_precision("high")

    train_ds = FramesWithLabels(train_rows, root, train_tf, args.id_col)
    val_ds   = FramesWithLabels(val_rows,   root, val_tf,   args.id_col)

    # Compute pos_weight for BCE (positive=FAKE=1)
    n_pos = sum(1 for _, y, _ in train_ds.items if y == 1)  # FAKE
    n_neg = sum(1 for _, y, _ in train_ds.items if y == 0)  # REAL
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], device=device, dtype=torch.float32)
    print(f"pos_weight={pos_weight.item():.3f} (pos=FAKE {n_pos}, neg=REAL {n_neg})")

    if args.no_balanced_sampler:
        sampler = None
        shuffle = True
    else:
        sampler = build_balanced_sampler(train_ds)
        shuffle = False

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers>0),
        prefetch_factor=4 if args.num_workers>0 else None
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers>0),
        prefetch_factor=4 if args.num_workers>0 else None
    )

    # --- Model ---
    model = timm.create_model("legacy_xception", pretrained=True, num_classes=1)
    model = model.to(device)
    model = model.to(memory_format=torch.channels_last)

    # Load last saved checkpoint if exists
    checkpoint_path = Path(args.out)
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded checkpoint from {checkpoint_path}")

    # Loss
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def make_optimizer(head_only: bool):
        if head_only:
            for p in model.parameters(): p.requires_grad = False
            for p in model.get_classifier().parameters(): p.requires_grad = True
            params = [{"params": model.get_classifier().parameters(), "lr": args.lr_head}]
        else:
            for p in model.parameters(): p.requires_grad = True
            # lower LR for backbone, higher for head
            head_params = list(model.get_classifier().parameters())
            head_ids = set(id(p) for p in head_params)
            backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
            params = [
                {"params": backbone_params, "lr": args.lr_backbone},
                {"params": head_params,     "lr": args.lr_head}
            ]
        optimizer = torch.optim.AdamW(params, weight_decay=1e-4)
        return optimizer

    def make_scheduler(optimizer, total_steps: int, warmup_ratio=0.05):
        warmup_steps = max(1, int(total_steps * warmup_ratio))
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            # cosine decay to zero
            progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    def eval_dl(dl):
        model.eval()
        total, correct, loss_sum = 0, 0, 0.0
        all_probs, all_labels, all_ids = [], [], []
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device.type=="cuda")):
            for xb, yb, ids in dl:
                xb = xb.to(device, non_blocking=True, memory_format=torch.channels_last)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb).squeeze(1)
                loss = criterion(logits, yb)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                correct += (preds == yb).sum().item()
                total += yb.numel()
                loss_sum += loss.item() * yb.size(0)
                all_probs.append(probs.detach().cpu())
                all_labels.append(yb.detach().cpu())
                all_ids.extend(list(ids))
        probs = torch.cat(all_probs).numpy()
        labels = torch.cat(all_labels).numpy().astype(np.float32)
        # per-frame metrics
        preds05 = (probs >= 0.5).astype(np.float32)
        prec = precision_score(labels, preds05, zero_division=0)
        rec  = recall_score(labels, preds05, zero_division=0)
        auc  = roc_auc_score(labels, probs) if (labels.min()!=labels.max()) else float('nan')
        cm05 = confusion_matrix(labels, preds05, labels=[0,1])
        acc05 = (cm05[0,0] + cm05[1,1]) / max(1, cm05.sum())
        bal05 = balanced_accuracy_from_cm(cm05)
        thr_frame, best_frame = sweep_best_threshold(labels, probs, metric="bal_acc")
        # asset-level aggregation
        buckets = defaultdict(list)
        for p, y, idv in zip(probs, labels, all_ids):
            buckets[idv].append((p, y))
        agg_probs, agg_labels = [], []
        for idv, items in buckets.items():
            ps = np.array([p for p,_ in items], dtype=np.float32)
            ys = np.array([y for _,y in items], dtype=np.float32)
            # assert same label within asset; take majority just in case
            y_lab = 1.0 if (ys.mean()>=0.5) else 0.0
            if args.agg == "mean":   p_agg = float(ps.mean())
            elif args.agg == "max":  p_agg = float(ps.max())
            elif args.agg == "p95":  p_agg = float(np.quantile(ps, 0.95))
            else:                    p_agg = float(np.median(ps))
            agg_probs.append(p_agg); agg_labels.append(y_lab)
        agg_probs = np.array(agg_probs, dtype=np.float32)
        agg_labels = np.array(agg_labels, dtype=np.float32)
        preds05_a = (agg_probs >= 0.5).astype(np.float32)
        auc_a = roc_auc_score(agg_labels, agg_probs) if (agg_labels.min()!=agg_labels.max()) else float('nan')
        cm05_a = confusion_matrix(agg_labels, preds05_a, labels=[0,1])
        acc05_a = (cm05_a[0,0] + cm05_a[1,1]) / max(1, cm05_a.sum())
        bal05_a = balanced_accuracy_from_cm(cm05_a)
        thr_asset, best_asset = sweep_best_threshold(agg_labels, agg_probs, metric="bal_acc")
        return {
            "val_loss": loss_sum / max(1,total),
            "frame": {
                "auc": float(auc), "acc@0.5": float(acc05), "bal_acc@0.5": float(bal05),
                "precision@0.5": float(prec), "recall@0.5": float(rec),
                "cm@0.5": cm05.tolist(),
                "best_thr": thr_frame, "best": best_frame
            },
            "asset": {
                "auc": float(auc_a), "acc@0.5": float(acc05_a), "bal_acc@0.5": float(bal05_a),
                "cm@0.5": cm05_a.tolist(),
                "best_thr": thr_asset, "best": best_asset,
                "n_assets": int(len(agg_labels))
            }
        }

    # ---------- Training ----------
    best_metric, best_state, best_meta = -1.0, None, None
    patience, no_improve = 2, 0

    total_epochs = max(1, args.epochs)
    head_epochs = min(args.head_warmup_epochs, total_epochs)
    full_epochs = max(0, total_epochs - head_epochs)

    # Phase 1: head-only warmup
    if head_epochs > 0:
        optimizer = make_optimizer(head_only=True)
        total_steps = head_epochs * len(train_dl)
        scheduler = make_scheduler(optimizer, total_steps)
        step_ct = 0
        for epoch in range(1, head_epochs+1):
            model.train()
            epoch_loss = 0.0
            for xb, yb, _ in tqdm(train_dl, desc=f"[Head] epoch {epoch}/{head_epochs}"):
                xb = xb.to(device, non_blocking=True, memory_format=torch.channels_last)
                yb = yb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=(device.type=="cuda")):
                    logits = model(xb).squeeze(1)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step(); step_ct += 1
                epoch_loss += loss.item() * xb.size(0)
            metrics = eval_dl(val_dl)
            print(f"Head Epoch {epoch}: "
                  f"val_loss={metrics['val_loss']:.4f}  "
                  f"frame_auc={metrics['frame']['auc']:.3f}  asset_auc={metrics['asset']['auc']:.3f}  "
                  f"asset_bal@0.5={metrics['asset']['bal_acc@0.5']:.3f}  n_assets={metrics['asset']['n_assets']}")
            # monitor
            monitor_val = metrics['asset']['auc'] if args.monitor=="auc" else \
                          (metrics['asset']['bal_acc@0.5'] if args.monitor=="bal_acc" else \
                           (metrics['frame']['acc@0.5']))
            if monitor_val > best_metric:
                best_metric = monitor_val
                best_state = {"state_dict": model.state_dict()}
                best_meta = {
                    "monitor": args.monitor,
                    "best_metric": float(best_metric),
                    "best_threshold_frame": metrics['frame']['best_thr'],
                    "best_threshold_asset": metrics['asset']['best_thr'],
                    "model_name": "legacy_xception",
                    "imagenet_norm": True,
                    "id_col": args.id_col
                }
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({**best_state, "meta": best_meta}, checkpoint_path)
                print(f"✅ Saved best checkpoint ({args.monitor}={monitor_val:.3f}) → {checkpoint_path}")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print("⏹ Early stopping during head warmup.")
                    print("Training complete.")
                    return

    # Phase 2: unfreeze all
    if full_epochs > 0:
        optimizer = make_optimizer(head_only=False)
        total_steps = full_epochs * len(train_dl)
        scheduler = make_scheduler(optimizer, total_steps)
        step_ct = 0
        for i in range(full_epochs):
            epoch = i + 1
            model.train()
            epoch_loss = 0.0
            for xb, yb, _ in tqdm(train_dl, desc=f"[Full] epoch {epoch}/{full_epochs}"):
                xb = xb.to(device, non_blocking=True, memory_format=torch.channels_last)
                yb = yb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=(device.type=="cuda")):
                    logits = model(xb).squeeze(1)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step(); step_ct += 1
                epoch_loss += loss.item() * xb.size(0)

            metrics = eval_dl(val_dl)
            print(f"Full Epoch {epoch}: "
                  f"val_loss={metrics['val_loss']:.4f}  "
                  f"frame_auc={metrics['frame']['auc']:.3f}  asset_auc={metrics['asset']['auc']:.3f}  "
                  f"asset_bal@0.5={metrics['asset']['bal_acc@0.5']:.3f}  n_assets={metrics['asset']['n_assets']}")
            # monitor
            monitor_val = metrics['asset']['auc'] if args.monitor=="auc" else \
                          (metrics['asset']['bal_acc@0.5'] if args.monitor=="bal_acc" else \
                           (metrics['frame']['acc@0.5']))
            if monitor_val > best_metric:
                best_metric = monitor_val
                best_state = {"state_dict": model.state_dict()}
                best_meta = {
                    "monitor": args.monitor,
                    "best_metric": float(best_metric),
                    "best_threshold_frame": metrics['frame']['best_thr'],
                    "best_threshold_asset": metrics['asset']['best_thr'],
                    "model_name": "legacy_xception",
                    "imagenet_norm": True,
                    "id_col": args.id_col
                }
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({**best_state, "meta": best_meta}, checkpoint_path)
                print(f"✅ Saved best checkpoint ({args.monitor}={monitor_val:.3f}) → {checkpoint_path}")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print("⏹ Early stopping (no improvement).")
                    break

    print("Training complete.")

if __name__ == "__main__":
    main()
