"""Train the classifier. Run on a GPU machine with the COVIDx CT images, e.g.:
    python scripts/train.py --image-dir /kaggle/input/datasets/hgunraj/covidxct/3A_images
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ctclf.data import CTDataset, eval_transform, train_transform  # noqa: E402
from ctclf.model import build_model  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--image-dir", required=True)
parser.add_argument("--epochs", type=int, default=12)
parser.add_argument("--out", default=str(ROOT / "checkpoints" / "best_exp2.pt"))
args = parser.parse_args()

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits = ROOT / "splits"
train_dl = DataLoader(CTDataset(splits / "train_balanced.csv", args.image_dir, train_transform),
                      batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(CTDataset(splits / "val.csv", args.image_dir, eval_transform),
                    batch_size=128, num_workers=4, pin_memory=True)

model = build_model(pretrained=True).to(device)
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=3e-4, total_steps=args.epochs * len(train_dl), pct_start=0.1)
use_amp = device.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)


def run_epoch(loader, train):
    model.train(train)
    total_loss, preds, labels = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train), torch.autocast(device.type, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        total_loss += loss.item() * len(y)
        preds.append(logits.argmax(1).cpu())
        labels.append(y.cpu())
    preds, labels = torch.cat(preds), torch.cat(labels)
    return (total_loss / len(labels), (preds == labels).float().mean().item(),
            f1_score(labels, preds, average="macro"))


best_f1, bad_epochs, patience = 0.0, 0, 4
for epoch in range(1, args.epochs + 1):
    t0 = time.time()
    tr_loss, tr_acc, _ = run_epoch(train_dl, train=True)
    va_loss, va_acc, va_f1 = run_epoch(val_dl, train=False)
    print(f"Epoch {epoch:2d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
          f"val loss {va_loss:.3f} acc {va_acc:.3f} F1 {va_f1:.3f} | {time.time() - t0:.0f}s")
    if va_f1 > best_f1:
        best_f1, bad_epochs = va_f1, 0
        Path(args.out).parent.mkdir(exist_ok=True)
        torch.save(model.state_dict(), args.out)
        print("   new best, saved")
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            print("Early stopping")
            break