import numpy as np
import pandas as pd

from .ids import patient_id, person_id

COLUMNS = ["filename", "label", "xmin", "ymin", "xmax", "ymax"]


def load_split(path, metadata):
    """Read a COVIDx CT label file and add patient, person and source columns."""
    df = pd.read_csv(path, sep=" ", names=COLUMNS)
    df["patient_id"] = df["filename"].apply(patient_id)
    df["person_id"] = df["patient_id"].apply(person_id)
    source = metadata.set_index("patient id")["source"]
    # underscore-style radiopaedia files are missing from metadata.csv
    df["source"] = df["patient_id"].map(source).fillna("radiopaedia.org")
    return df


def shared_people(a, b):
    return set(a["person_id"]) & set(b["person_id"])


def check_no_leakage(splits):
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = shared_people(splits[a], splits[b])
            if shared:
                raise ValueError(f"{len(shared)} people in both {a} and {b}: {sorted(shared)[:5]}")


def even_take(group, k):
    """Take k slices spread evenly through one person's scan."""
    group = group.sort_values("filename")
    if len(group) <= k:
        return group
    idx = np.linspace(0, len(group) - 1, k).round().astype(int)
    return group.iloc[idx]


def sample_split(df, target, max_per_person=30, seed=42):
    """Class-balanced sample, picking whole people in random order."""
    rng = np.random.default_rng(seed)
    per_class = target // 3
    picked = []
    for label in [0, 1, 2]:
        sub = df[df["label"] == label]
        groups = dict(tuple(sub.groupby("person_id")))
        people = np.array(list(groups), dtype=object)
        rng.shuffle(people)
        taken = 0
        for pid in people:
            if taken >= per_class:
                break
            part = even_take(groups[pid], min(max_per_person, per_class - taken))
            picked.append(part)
            taken += len(part)
    return pd.concat(picked).sample(frac=1, random_state=seed).reset_index(drop=True)


def sample_balanced(df, per_class, max_per_person=15, seed=42):
    """Class- and source-balanced sample, so no hospital dominates a class."""
    rng = np.random.default_rng(seed)
    picked = []
    for label in [0, 1, 2]:
        sub = df[df["label"] == label]
        cap = {s: int(g.groupby("person_id").size().clip(upper=max_per_person).sum())
               for s, g in sub.groupby("source")}
        quota, remaining = {s: 0 for s in cap}, per_class
        while remaining > 0:
            open_s = [s for s in cap if quota[s] < cap[s]]
            if not open_s:
                break
            share = max(1, remaining // len(open_s))
            for s in open_s:
                add = min(share, cap[s] - quota[s], remaining)
                quota[s] += add
                remaining -= add
                if remaining == 0:
                    break
        for s, q in quota.items():
            groups = dict(tuple(sub[sub["source"] == s].groupby("person_id")))
            people = np.array(list(groups), dtype=object)
            rng.shuffle(people)
            taken = 0
            for pid in people:
                if taken >= q:
                    break
                part = even_take(groups[pid], min(max_per_person, q - taken))
                picked.append(part)
                taken += len(part)
    return pd.concat(picked).sample(frac=1, random_state=seed).reset_index(drop=True)