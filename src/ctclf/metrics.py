import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from . import CLASSES


def metrics(y, pred, probs):
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    out = {"accuracy": float((y == pred).mean()),
           "macro_F1": float(f1_score(y, pred, average="macro")),
           "macro_AUC": float(roc_auc_score(y, probs, multi_class="ovr"))}
    for k, name in enumerate(CLASSES):
        tp = cm[k, k]
        fn = cm[k].sum() - tp
        fp = cm[:, k].sum() - tp
        tn = cm.sum() - tp - fn - fp
        out[f"{name} sensitivity"] = float(tp / (tp + fn))
        out[f"{name} specificity"] = float(tn / (tn + fp))
    return out


def person_bootstrap(y, pred, probs, persons, n=1000, seed=0):
    """95% CIs by resampling people, not slices (slices from one person aren't independent)."""
    idx_by_person = {p: np.flatnonzero(persons == p) for p in np.unique(persons)}
    people = np.array(list(idx_by_person), dtype=object)
    rng, rows = np.random.default_rng(seed), []
    for _ in range(n):
        idx = np.concatenate([idx_by_person[p] for p in rng.choice(people, len(people))])
        if len(np.unique(y[idx])) == 3:
            rows.append(metrics(y[idx], pred[idx], probs[idx]))
    keys = rows[0].keys()
    return {k: (float(np.quantile([r[k] for r in rows], 0.025)),
                float(np.quantile([r[k] for r in rows], 0.975))) for k in keys}