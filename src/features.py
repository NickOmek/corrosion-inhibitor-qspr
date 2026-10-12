"""Molecular featurisation and a leakage-safe descriptor filter."""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.base import BaseEstimator, TransformerMixin


def rdkit_descriptors(smiles: list[str]) -> pd.DataFrame:
    """All RDKit 2D descriptors (about 210). Failed calculations become NaN."""
    names = [n for n, _ in Descriptors._descList]
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        row = []
        for _, fn in Descriptors._descList:
            try:
                row.append(fn(mol))
            except Exception:
                row.append(np.nan)
        rows.append(row)
    return pd.DataFrame(rows, columns=names)


def _morgan_generator(radius: int, n_bits: int):
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


def morgan_bitvects(smiles: list[str], radius: int = 2, n_bits: int = 2048):
    gen = _morgan_generator(radius, n_bits)
    return [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]


def morgan_fingerprints(
    smiles: list[str], radius: int = 2, n_bits: int = 2048
) -> pd.DataFrame:
    """ECFP4-style count-free bit fingerprints as a DataFrame."""
    fps = morgan_bitvects(smiles, radius, n_bits)
    arr = np.zeros((len(fps), n_bits), dtype=np.uint8)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, arr[i])
    return pd.DataFrame(arr, columns=[f"ecfp4_{i}" for i in range(n_bits)])


def build_feature_sets(smiles: list[str]) -> dict[str, pd.DataFrame]:
    desc = rdkit_descriptors(smiles)
    ecfp = morgan_fingerprints(smiles)
    return {
        "RDKit descriptors": desc,
        "ECFP4": ecfp,
        "Descriptors + ECFP4": pd.concat([desc, ecfp], axis=1),
    }


class DescriptorFilter(BaseEstimator, TransformerMixin):
    """Clean a descriptor matrix using statistics from the training fold only.

    Steps, all fitted on training data so nothing leaks from validation molecules:
    replace +/-inf with NaN, drop columns with too many missing values, impute the
    rest with the median, drop (near-)constant columns, then drop one column of
    every pair correlated above ``corr_threshold``.
    """

    def __init__(self, max_nan_frac=0.05, min_variance=1e-8, corr_threshold=0.95):
        self.max_nan_frac = max_nan_frac
        self.min_variance = min_variance
        self.corr_threshold = corr_threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X).replace([np.inf, -np.inf], np.nan)
        keep = X.columns[X.isna().mean() <= self.max_nan_frac]
        X = X[keep]
        self.medians_ = X.median()
        X = X.fillna(self.medians_)
        keep = X.columns[X.var() > self.min_variance]
        X = X[keep]
        if self.corr_threshold < 1 and X.shape[1] > 1:
            # A column is dropped if it is highly correlated with any earlier column.
            corr = np.nan_to_num(np.abs(np.corrcoef(X.to_numpy(float), rowvar=False)))
            drop_mask = (np.triu(corr, k=1) > self.corr_threshold).any(axis=0)
            X = X.loc[:, ~drop_mask]
        self.columns_ = list(X.columns)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).replace([np.inf, -np.inf], np.nan)
        X = X[self.columns_].fillna(self.medians_[self.columns_])
        return X.to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_)
