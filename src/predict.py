"""Predict inhibition efficiency for new molecules with the saved XGBoost model."""
from __future__ import annotations

import joblib
import pandas as pd
from rdkit import DataStructs

from .config import MODELS
from .data import load_clean, standardize_smiles
from .features import morgan_bitvects, rdkit_descriptors

MODEL_PATH = MODELS / "xgb_descriptors.joblib"


def predict_ie(smiles: list[str], model_path=MODEL_PATH) -> pd.DataFrame:
    """Return predicted IE (%) and similarity to the training set for each SMILES.

    ``max_similarity`` (Tanimoto, ECFP4) is a simple applicability-domain signal:
    in this study the error is clearly larger for molecules that are dissimilar to
    the training data, so treat low-similarity predictions as unreliable.
    Valid only for carbon steel in 1 M HCl at about 1 mmol/L inhibitor.
    """
    model = joblib.load(model_path)
    std = [standardize_smiles(s)[0] for s in smiles]
    ok = [s is not None for s in std]
    out = pd.DataFrame({"input": smiles, "standardised": std, "pred_IE": float("nan"),
                        "max_similarity": float("nan")})
    valid = [s for s in std if s is not None]
    if valid:
        out.loc[ok, "pred_IE"] = model.predict(rdkit_descriptors(valid)).clip(0, 100)
        train_fps = morgan_bitvects(load_clean()["smiles"].tolist())
        sims = [max(DataStructs.BulkTanimotoSimilarity(fp, train_fps))
                for fp in morgan_bitvects(valid)]
        out.loc[ok, "max_similarity"] = sims
    return out
