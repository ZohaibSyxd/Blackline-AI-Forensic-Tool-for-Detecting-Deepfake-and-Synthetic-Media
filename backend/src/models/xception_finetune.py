import argparse, json, random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import torch
import timm
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
val_tf = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def _norm_label(v: Any) -> Optional[int]:
    if v is None: return None
    s = str(v).strip().lower()
    if s in ("1","true","fake","deepfake"): return 1
    if s in ("0","false","real","genuine"): return 0
    try:
        n = int(float(s))
        if n in (0,1): return n
    except Exception:
        pass
    return None

def _resolve_path(rel_or_abs: str, root: Path) -> Optional[Path]:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p if p.exists() else None
    p = (root / p).resolve()
    return p if p.exists() else None

class FramesWithLabels(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], root: Path, tf):
        self.items: List[Tuple[Path, int]] = []
        self.tf = tf
        for r in rows:
            rel = r.get("normalized_uri") or r.get("uri")
            if not rel: continue
            p = _resolve_path(rel, root)
            if not p: continue
            y = r.get("_label")
            if y is None: continue
            self.items.append((p, int(y)))

    def __len__(self): return len(self.items)

    def __getitem__(self, idx: int):
        path, y = self.items[idx]
        img = Image.open(path).convert("RGB")
        x = self.tf(img)
        return x, torch.tensor(y, dtype=torch.float32)

def load_rows_with_labels(frames_jsonl: Path, label_by: Dict[str, int], id_col: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(frames_jsonl, "r", encoding="utf-8") as fr:
        for line in fr:
            try:
                row = json.loads(line)
            except Exception:
                continue
            key = row.get(id_col)
            if key in label_by:
                row["_label"] = label_by[key]
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

def main():
    ap = argparse.ArgumentParser("Fine-tune Xception head for deepfake detection")
    ap.add_argument("--frames-jsonl", required=True, help="frames_normalized.jsonl")
    ap.add_argument("--root", required=True, help="Root for images (e.g., backend/data/derived)")
    ap.add_argument("--manifest", required=True, help="CSV with labels")
    ap.add_argument("--id-col", default="sha256", help="Key to join labels (sha256 or asset_id)")
    ap.add_argument("--label-col", default="label", help="Column in manifest with labels (real/fake or 0/1)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--out", default="backend/data/derived/xception_finetuned.pth")
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)
    df = pd.read_csv(args.manifest)

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

    train_ds = FramesWithLabels(train_rows, root, train_tf)
    val_ds   = FramesWithLabels(val_rows, root, val_tf)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = timm.create_model("xception", pretrained=True, num_classes=1)
    for p in model.parameters(): p.requires_grad = False
    for p in model.get_classifier().parameters(): p.requires_grad = True
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.get_classifier().parameters(), lr=args.lr)

    def eval_dl(dl):
        model.eval()
        total, correct, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for xb, yb in dl:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb).squeeze(1)
                loss = criterion(logits, yb)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                correct += (preds == yb).sum().item()
                total += yb.numel()
                loss_sum += loss.item() * yb.size(0)
        return (loss_sum / max(1,total), correct / max(1,total))

    best_acc, best_state = 0.0, None
    for epoch in range(1, args.epochs+1):
        model.train()
        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb).squeeze(1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        val_loss, val_acc = eval_dl(val_dl)
        print(f"Epoch {epoch}/{args.epochs} - val_loss={val_loss:.4f} val_acc={val_acc:.3f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = { "state_dict": model.state_dict() }

    if best_state is None:
        best_state = { "state_dict": model.state_dict() }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_path)
    print(f"Saved fine-tuned checkpoint → {out_path}")

if __name__ == "__main__":
    main()