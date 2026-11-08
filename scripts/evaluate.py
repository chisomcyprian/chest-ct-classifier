"""Evaluate a trained model on the test split. Run where the images are, e.g.:
    python scripts/evaluate.py --image-dir /kaggle/input/datasets/hgunraj/covidxct/3A_images
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ctclf.data import CTDataset, eval_transform  # noqa: E402
from ctclf.metrics import metrics, person_bootstrap  # noqa: E402
from ctclf.model import load_model  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--image-dir", required=True)
parser.add_argument("--weights", default=str(ROOT / "checkpoints" / "best_exp2.pt"))
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(args.weights, device)
ds = CTDataset(ROOT / "splits" / "test.csv", args.image_dir, eval_transform)

logits = []
with torch.no_grad():
    for x, _ in DataLoader(ds, batch_size=128, num_workers=4):
        logits.append(model(x.to(device)).float().cpu())
probs = torch.softmax(torch.cat(logits), dim=1).numpy()
pred, y, df = probs.argmax(1), ds.df["label"].values, ds.df

report = {"test": metrics(y, pred, probs),
          "test_95ci": person_bootstrap(y, pred, probs, df["person_id"].values),
          "confusion_matrix": confusion_matrix(y, pred).tolist()}

cncb = (df["source"] == "CNCB").values
report["cncb_only_macro_F1"] = float(f1_score(y[cncb], pred[cncb], average="macro"))

per_person = (df.assign(p0=probs[:, 0], p1=probs[:, 1], p2=probs[:, 2])
              .groupby("person_id").agg(label=("label", "first"), p0=("p0", "mean"),
                                        p1=("p1", "mean"), p2=("p2", "mean")))
report["per_person_accuracy"] = float(
    (per_person[["p0", "p1", "p2"]].values.argmax(1) == per_person["label"].values).mean())

(ROOT / "reports").mkdir(exist_ok=True)
with open(ROOT / "reports" / "metrics.json", "w") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))