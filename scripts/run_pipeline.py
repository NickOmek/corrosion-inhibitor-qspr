"""Run the full study: data -> features -> benchmark -> figures -> SHAP -> AD.

Usage:
    python scripts/run_pipeline.py            # full study (3 x 10-fold CV)
    python scripts/run_pipeline.py --quick    # fast smoke run (1 x 5-fold CV)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config as C  # noqa: E402
from src.data import build_clean_dataset  # noqa: E402
from src.evaluate import run_cv  # noqa: E402
from src.features import build_feature_sets  # noqa: E402
from src.interpret import (  # noqa: E402
    applicability_domain,
    oof_max_similarity,
    plot_benchmark,
    plot_parity,
    shap_analysis,
)
from src.models import make_models  # noqa: E402


def main(quick: bool = False) -> None:
    n_splits, n_repeats = (5, 1) if quick else (C.N_SPLITS, C.N_REPEATS)
    for d in (C.FIGURES, C.MODELS, C.RESULTS):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Data -----------------------------------------------------------------
    df = build_clean_dataset()
    y = df[C.TARGET].to_numpy(dtype=float)
    groups = df["scaffold"].to_numpy()
    print(f"Clean dataset: {len(df)} molecules, {df['scaffold'].nunique()} scaffolds, "
          f"IE {y.min():.1f}-{y.max():.1f}% (mean {y.mean():.1f}, sd {y.std():.1f})")

    # 2. Features -------------------------------------------------------------
    feats = build_feature_sets(df["smiles"].tolist())
    for k, v in feats.items():
        print(f"  feature set '{k}': {v.shape[1]} raw columns")

    # 3. Benchmark ------------------------------------------------------------
    rows, oof_store = [], {}
    models = make_models(C.SEED)
    t0 = time.time()
    for fs_name, X in feats.items():
        for m_name, model in models.items():
            for scheme in ("random", "scaffold"):
                res, oof = run_cv(model, X, y, groups, scheme, n_splits, n_repeats, C.SEED)
                rows.append({"features": fs_name, "model": m_name, "scheme": scheme, **res})
                oof_store[(fs_name, m_name, scheme)] = oof
                print(f"  [{time.time() - t0:5.0f}s] {fs_name:20s} {m_name:22s} {scheme:8s} "
                      f"R2={res['R2']:.3f} RMSE={res['RMSE']:.2f} MAE={res['MAE']:.2f}")
    results = pd.DataFrame(rows)
    results.to_csv(C.RESULTS / "benchmark.csv", index=False)

    # 4. Pick the best model under the stricter (scaffold) scheme -------------
    cand = results[(results["scheme"] == "scaffold") & (results["model"] != "Mean baseline")]
    best = cand.sort_values("RMSE").iloc[0]
    fs_best, m_best = best["features"], best["model"]
    print(f"\nBest under scaffold CV: {m_best} on '{fs_best}' "
          f"(R2={best['R2']:.3f}, RMSE={best['RMSE']:.2f})")

    plot_benchmark(results, "RDKit descriptors", C.FIGURES / "benchmark_rmse.png")
    for scheme in ("random", "scaffold"):
        plot_parity(
            y, oof_store[(fs_best, m_best, scheme)],
            f"{m_best}: {scheme} CV", C.FIGURES / f"parity_{scheme}.png",
        )

    # 5. Interpretation: XGBoost on named RDKit descriptors -------------------
    X_desc = feats["RDKit descriptors"]
    xgb = make_models(C.SEED)["XGBoost"].fit(X_desc, y)
    imp = shap_analysis(xgb, X_desc, C.FIGURES, C.RESULTS / "shap_importance.csv")
    joblib.dump(xgb, C.MODELS / "xgb_descriptors.joblib")
    print("\nTop descriptors by mean |SHAP|:")
    print(imp.head(10).to_string(index=False))

    # 6. Applicability domain (scaffold CV, best model, repeat 0) -------------
    sim = oof_max_similarity(df["smiles"], groups, "scaffold", n_splits, C.SEED)
    ad = applicability_domain(
        y, oof_store[(fs_best, m_best, "scaffold")], sim,
        C.FIGURES / "applicability_domain.png", C.RESULTS / "applicability_domain.csv",
    )
    print("\nError by similarity to training set:")
    print(ad.to_string(index=False))

    # 7. Markdown summary table ----------------------------------------------
    lines = ["| Features | Model | CV | R² | RMSE | MAE |", "|---|---|---|---|---|---|"]
    for _, r in results.sort_values(["scheme", "features", "RMSE"]).iterrows():
        lines.append(
            f"| {r['features']} | {r['model']} | {r['scheme']} | "
            f"{r['R2']:.2f} ± {r['R2_sd']:.2f} | {r['RMSE']:.1f} ± {r['RMSE_sd']:.1f} | "
            f"{r['MAE']:.1f} ± {r['MAE_sd']:.1f} |"
        )
    (C.RESULTS / "benchmark_table.md").write_text("\n".join(lines) + "\n")
    print(f"\nDone in {time.time() - t0:.0f}s. Outputs in {C.RESULTS}")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
