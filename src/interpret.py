"""Figures, SHAP interpretation and applicability-domain analysis."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from rdkit import DataStructs

from .config import AQUA, BLUE, GRID, INK, INK_2, ORANGE, SURFACE
from .evaluate import get_folds, metrics
from .features import morgan_bitvects


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "text.color": INK,
            "font.size": 10,
            "legend.frameon": False,
            "figure.dpi": 130,
        }
    )


_style()


def plot_benchmark(results: pd.DataFrame, feature_set: str, path: Path) -> None:
    """Grouped bars: RMSE under random vs scaffold CV for one feature set."""
    df = results[results["features"] == feature_set]
    models = list(dict.fromkeys(df["model"]))
    x = np.arange(len(models))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for off, scheme, color in [(-w / 2, "random", BLUE), (w / 2, "scaffold", ORANGE)]:
        sub = df[df["scheme"] == scheme].set_index("model").loc[models]
        bars = ax.bar(
            x + off, sub["RMSE"], w - 0.04, yerr=sub["RMSE_sd"], color=color,
            label=f"{scheme.capitalize()} CV", error_kw={"ecolor": INK_2, "lw": 1, "capsize": 2},
        )
        for b, v in zip(bars, sub["RMSE"]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.35, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=8.5, color=INK_2)
    ax.set_xticks(x, models, rotation=15, ha="right")
    ax.set_ylabel("RMSE of predicted inhibition efficiency (%)")
    ax.set_title(f"Prediction error by model and CV scheme ({feature_set})", loc="left")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(0, df["RMSE"].max() * 1.22)
    ax.legend(loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_parity(y, pred, title: str, path: Path) -> None:
    m = metrics(y, pred)
    lo, hi = min(y.min(), pred.min()) - 3, 102
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot([lo, hi], [lo, hi], color=INK_2, lw=1, ls="--", zorder=1)
    ax.scatter(y, pred, s=26, color=BLUE, alpha=0.7, edgecolor=SURFACE, linewidth=0.8, zorder=2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Experimental IE (%)")
    ax.set_ylabel("Predicted IE (%), out-of-fold")
    ax.set_title(title, loc="left")
    ax.text(0.04, 0.95, f"R$^2$ = {m['R2']:.2f}\nRMSE = {m['RMSE']:.1f}\nMAE = {m['MAE']:.1f}",
            transform=ax.transAxes, va="top", color=INK, fontsize=10)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def shap_analysis(fitted_pipeline, X: pd.DataFrame, fig_dir: Path, table_path: Path, top: int = 15):
    """SHAP values for a fitted [filter -> tree model] pipeline."""
    filt = fitted_pipeline.named_steps["filter"]
    model = fitted_pipeline.named_steps["model"]
    names = list(filt.get_feature_names_out())
    Xt = pd.DataFrame(filt.transform(X), columns=names)
    sv = shap.TreeExplainer(model).shap_values(Xt)
    imp = (
        pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(sv).mean(0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    imp.to_csv(table_path, index=False)

    # Bar chart of the top features
    t = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.barh(t["feature"], t["mean_abs_shap"], color=BLUE, height=0.62)
    ax.set_xlabel("Mean |SHAP value| (percentage points of IE)")
    ax.set_title(f"Top {top} descriptors driving predicted IE", loc="left")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(fig_dir / "shap_bar.png", bbox_inches="tight")
    plt.close(fig)

    # Beeswarm: direction of each descriptor's effect
    shap.summary_plot(sv, Xt, max_display=top, show=False, plot_size=(7.5, 5.2))
    plt.title("SHAP summary: effect of each descriptor on predicted IE", loc="left", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_summary.png", bbox_inches="tight")
    plt.close()
    return imp


def oof_max_similarity(smiles, groups, scheme: str, n_splits: int, seed: int) -> np.ndarray:
    """For each molecule, max Tanimoto (ECFP4) to the training set of its CV fold.

    Uses the same folds as repeat 0 of `run_cv`, so values align with its OOF predictions.
    """
    fps = morgan_bitvects(list(smiles))
    sim = np.zeros(len(fps))
    for train, test in get_folds(scheme, groups, len(fps), n_splits, seed):
        train_fps = [fps[i] for i in train]
        for i in test:
            sim[i] = max(DataStructs.BulkTanimotoSimilarity(fps[i], train_fps))
    return sim


def applicability_domain(y, pred, sim, path: Path, table_path: Path) -> pd.DataFrame:
    """Absolute error versus similarity to the training data."""
    err = np.abs(y - pred)
    bins = [0, 0.3, 0.4, 0.5, 0.6, 1.01]
    labels = ["<0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", ">=0.6"]
    cat = pd.cut(sim, bins=bins, labels=labels, right=False)
    tab = (
        pd.DataFrame({"similarity_bin": cat, "abs_error": err})
        .groupby("similarity_bin", observed=True)["abs_error"]
        .agg(n="size", MAE="mean")
        .reset_index()
    )
    tab.to_csv(table_path, index=False)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.25, 1]})
    a1.scatter(sim, err, s=22, color=BLUE, alpha=0.65, edgecolor=SURFACE, linewidth=0.7)
    a1.set_xlabel("Max Tanimoto similarity to training molecules (ECFP4)")
    a1.set_ylabel("Absolute error (IE %)")
    a1.set_title("Error versus novelty of the molecule", loc="left")
    a1.grid(True, color=GRID, lw=0.8)
    a1.set_axisbelow(True)
    bars = a2.bar(tab["similarity_bin"].astype(str), tab["MAE"], color=BLUE, width=0.62)
    for b, n, v in zip(bars, tab["n"], tab["MAE"]):
        a2.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}\n(n={n})",
                ha="center", va="bottom", fontsize=8.5, color=INK_2)
    a2.set_xlabel("Similarity bin")
    a2.set_ylabel("MAE (IE %)")
    a2.set_title("MAE by similarity bin", loc="left")
    a2.yaxis.grid(True, color=GRID, lw=0.8)
    a2.set_axisbelow(True)
    a2.set_ylim(0, tab["MAE"].max() * 1.35)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return tab
