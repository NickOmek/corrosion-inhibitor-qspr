import numpy as np
import pandas as pd
import pytest

from src.data import clean, murcko_scaffold, standardize_smiles
from src.evaluate import metrics, scaffold_group_folds
from src.features import DescriptorFilter, morgan_fingerprints, rdkit_descriptors


def test_salt_stripped_to_organic_cation():
    smi, n_frag = standardize_smiles("C[N+](C)(C)CCCCCCCCCC.[Br-]")
    assert n_frag == 2
    assert "Br" not in smi
    assert "[N+]" in smi  # charge on the retained fragment is kept


def test_invalid_smiles_returns_none():
    smi, n_frag = standardize_smiles("not_a_smiles")
    assert smi is None and n_frag == 0


def test_scaffold_labels():
    assert murcko_scaffold("CCCCO") == "acyclic"
    assert murcko_scaffold("c1ccccc1CC") == murcko_scaffold("c1ccccc1O")


def test_clean_merges_duplicates_and_drops_invalid():
    raw = pd.DataFrame(
        {
            "mole_id": [1, 2, 3, 4],
            "canonical_SMILES": ["c1ccccc1O", "Oc1ccccc1", "CCO.[Cl-]", "bad"],
            "IE_exp": [80.0, 90.0, 70.0, 50.0],
        }
    )
    out = clean(raw)
    assert len(out) == 2  # phenol duplicates merged, invalid dropped
    phenol = out[out["smiles"] == "Oc1ccccc1"].iloc[0]
    assert phenol["IE"] == pytest.approx(85.0)
    assert phenol["n_records"] == 2


def test_scaffold_folds_never_split_a_scaffold():
    groups = np.array(["a"] * 6 + ["b"] * 5 + ["c"] * 4 + ["d"] * 3 + ["e"] * 2)
    seen = np.zeros(len(groups), dtype=int)
    for train, test in scaffold_group_folds(groups, 3, seed=0):
        assert set(train).isdisjoint(test)
        assert set(groups[train]).isdisjoint(groups[test])
        seen[test] += 1
    assert (seen == 1).all()  # every molecule is tested exactly once


def test_descriptor_filter_uses_train_statistics_only():
    rng = np.random.default_rng(0)
    train = pd.DataFrame(
        {
            "a": rng.normal(size=50),
            "const": 1.0,
            "mostly_nan": [np.nan] * 40 + list(range(10)),
            "ok_nan": [np.nan] + list(rng.normal(size=49)),
        }
    )
    train["a_copy"] = train["a"] * 2 + 1e-3  # perfectly correlated with 'a'
    filt = DescriptorFilter().fit(train)
    assert "const" not in filt.columns_
    assert "mostly_nan" not in filt.columns_
    assert "a_copy" not in filt.columns_ and "a" in filt.columns_
    test = pd.DataFrame({c: [np.nan, np.inf] for c in train.columns})
    out = filt.transform(test)
    assert np.isfinite(out).all()  # NaN/inf filled from training medians


def test_descriptor_shapes_and_values():
    smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
    desc = rdkit_descriptors(smiles)
    assert desc.shape[0] == 3 and desc.shape[1] > 150
    assert desc.loc[1, "NumAromaticRings"] == 1
    fp = morgan_fingerprints(smiles)
    assert fp.shape == (3, 2048) and fp.to_numpy().sum() > 0


def test_metrics_perfect_and_mean_predictor():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert metrics(y, y)["R2"] == pytest.approx(1.0)
    assert metrics(y, np.full(4, y.mean()))["R2"] == pytest.approx(0.0)
