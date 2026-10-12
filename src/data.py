"""Load, standardise and de-duplicate the Fe-HCl-317 corrosion inhibitor dataset.

Source: Pham, T. H., Le, P. K., Son, D. N. (2024), RSC Advances,
"A data-driven QSPR model for screening organic corrosion inhibitors for carbon
steel using machine learning techniques". 317 organic inhibitors, carbon steel,
1 M HCl, 25-30 C, 1 mmol/L inhibitor, EIS-measured inhibition efficiency (IE, %).
"""
from __future__ import annotations

import urllib.request

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

from .config import CLEAN_FILE, DATA_RAW, RAW_FILE, RAW_URL

RDLogger.DisableLog("rdApp.*")
_CHOOSER = rdMolStandardize.LargestFragmentChooser()


def download_raw(force: bool = False) -> None:
    """Download the raw dataset from its original GitHub repository."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    if RAW_FILE.exists() and not force:
        print(f"Already present: {RAW_FILE}")
        return
    print(f"Downloading {RAW_URL}")
    urllib.request.urlretrieve(RAW_URL, RAW_FILE)
    print(f"Saved to {RAW_FILE}")


def load_raw() -> pd.DataFrame:
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"{RAW_FILE} not found. Run `python scripts/download_data.py` first."
        )
    return pd.read_excel(RAW_FILE)


def standardize_smiles(smiles: str) -> tuple[str | None, int]:
    """Return (canonical SMILES of the largest fragment, number of fragments).

    Salts such as quaternary ammonium bromides are reduced to the organic cation
    so that counter-ions do not dominate the descriptors. Charges on the retained
    fragment are preserved, because they are chemically meaningful.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, 0
    n_frag = len(Chem.GetMolFrags(mol))
    main = _CHOOSER.choose(mol)
    return Chem.MolToSmiles(main), n_frag


def murcko_scaffold(smiles: str) -> str:
    """Bemis-Murcko scaffold; acyclic molecules share the label 'acyclic'."""
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    return scaffold if scaffold else "acyclic"


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Standardise structures, drop invalid rows, merge duplicate molecules."""
    df = raw.rename(columns={"canonical_SMILES": "smiles_raw", "IE_exp": "IE"})
    df = df[["mole_id", "smiles_raw", "IE"]].copy()

    std = df["smiles_raw"].map(standardize_smiles)
    df["smiles"] = std.map(lambda t: t[0])
    df["n_fragments"] = std.map(lambda t: t[1])
    df = df.dropna(subset=["smiles", "IE"])

    # After salt stripping, identical molecules can appear more than once.
    grouped = (
        df.groupby("smiles", as_index=False)
        .agg(
            mole_id=("mole_id", "first"),
            smiles_raw=("smiles_raw", "first"),
            n_fragments=("n_fragments", "max"),
            IE=("IE", "mean"),
            n_records=("IE", "size"),
        )
        .sort_values("mole_id")
        .reset_index(drop=True)
    )
    grouped["scaffold"] = grouped["smiles"].map(murcko_scaffold)
    return grouped


def build_clean_dataset(save: bool = True) -> pd.DataFrame:
    df = clean(load_raw())
    if save:
        CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CLEAN_FILE, index=False)
    return df


def load_clean() -> pd.DataFrame:
    if not CLEAN_FILE.exists():
        return build_clean_dataset()
    return pd.read_csv(CLEAN_FILE)
