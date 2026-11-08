import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ctclf.sampling import (check_no_leakage, load_split, sample_balanced,  # noqa: E402
                            sample_split, shared_people)

RAW = ROOT / "data" / "raw"
OUT = ROOT / "splits"

meta = pd.read_csv(RAW / "metadata.csv")
full = {name: load_split(RAW / f"{name}_COVIDx_CT-3A.txt", meta)
        for name in ["train", "val", "test"]}

leaky = shared_people(full["train"], full["test"])
print("People in both train and test:", sorted(leaky))
full["train"] = full["train"][~full["train"]["person_id"].isin(leaky)]
check_no_leakage(full)

splits = {
    "train_balanced": sample_balanced(full["train"], per_class=5333),
    "val": sample_split(full["val"], 2000),
    "test": sample_split(full["test"], 2000),
}
check_no_leakage({"train": splits["train_balanced"], "val": splits["val"], "test": splits["test"]})

OUT.mkdir(exist_ok=True)
for name, df in splits.items():
    df.to_csv(OUT / f"{name}.csv", index=False)
    print(f"{name}: {len(df)} slices, {df['person_id'].nunique()} people")