"""Cross-validation: random vs scaffold-grouped, repeated, with pooled OOF metrics."""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def scaffold_group_folds(
    groups: np.ndarray, n_splits: int, seed: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Group k-fold where all molecules sharing a scaffold land in the same fold.

    Largest groups are placed first into the currently smallest fold (greedy
    balancing); ties are broken randomly so different seeds give different splits.
    """
    groups = np.asarray(groups)
    uniq, counts = np.unique(groups, return_counts=True)
    rng = np.random.default_rng(seed)
    order = sorted(rng.permutation(len(uniq)), key=lambda i: -counts[i])
    sizes = np.zeros(n_splits)
    assign = {}
    for i in order:
        fold = int(np.argmin(sizes))
        assign[uniq[i]] = fold
        sizes[fold] += counts[i]
    fold_id = np.array([assign[g] for g in groups])
    for k in range(n_splits):
        yield np.where(fold_id != k)[0], np.where(fold_id == k)[0]


def get_folds(scheme: str, groups, n: int, n_splits: int, seed: int):
    if scheme == "random":
        return list(KFold(n_splits, shuffle=True, random_state=seed).split(np.arange(n)))
    if scheme == "scaffold":
        return list(scaffold_group_folds(groups, n_splits, seed))
    raise ValueError(f"unknown scheme: {scheme}")


def run_cv(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    groups,
    scheme: str,
    n_splits: int = 10,
    n_repeats: int = 3,
    seed: int = 42,
) -> tuple[dict[str, float], np.ndarray]:
    """Repeated CV. Returns (mean/std metrics over repeats, OOF predictions of repeat 0).

    Metrics are computed on the pooled out-of-fold predictions of each repeat, so
    every molecule is predicted by a model that never saw it (or its scaffold,
    under the scaffold scheme).
    """
    per_repeat, oof0 = [], None
    for r in range(n_repeats):
        folds = get_folds(scheme, groups, len(y), n_splits, seed + r)
        oof = np.zeros(len(y))
        for train, test in folds:
            est = clone(model).fit(X.iloc[train], y[train])
            oof[test] = est.predict(X.iloc[test])
        per_repeat.append(metrics(y, oof))
        if r == 0:
            oof0 = oof
    res = pd.DataFrame(per_repeat)
    out = {}
    for m in res.columns:
        out[m] = res[m].mean()
        out[f"{m}_sd"] = res[m].std(ddof=1) if n_repeats > 1 else 0.0
    return out, oof0
