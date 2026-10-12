"""Generate notebooks/01_walkthrough.ipynb (run once; the notebook is committed)."""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("# Corrosion inhibitor QSPR: step-by-step walkthrough\n\n"
       "Predicting the inhibition efficiency (IE) of organic corrosion inhibitors for carbon steel in "
       "1 M HCl from molecular structure. Run the cells top to bottom; each step reuses the code in `src/`. "
       "The full benchmark is `python scripts/run_pipeline.py`."),
    md("## 1. Setup"),
    code("import sys\nfrom pathlib import Path\n\n"
         "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
         "sys.path.insert(0, str(ROOT))\n\n"
         "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\n"
         "from src import config as C\n"
         "from src.data import download_raw, load_raw, build_clean_dataset\n"
         "from src.features import rdkit_descriptors\n"
         "from src.models import make_models\n"
         "from src.evaluate import run_cv"),
    md("## 2. Get and inspect the data\n\n"
       "The Fe-HCl-317 dataset (Pham et al., *RSC Advances*, 2024) is downloaded from its original repository."),
    code("download_raw()\nraw = load_raw()\nprint(raw.shape)\nraw.drop(columns='structure').head()"),
    code("raw['IE_exp'].describe()"),
    md("## 3. Standardise structures\n\n"
       "Counter-ions are stripped (largest fragment kept), invalid SMILES are dropped, and molecules that "
       "become identical after standardisation are merged. Each molecule also gets a Bemis-Murcko scaffold "
       "label, used later to build splits that test generalisation to new chemical families."),
    code("df = build_clean_dataset()\n"
         "print(len(df), 'molecules,', df['scaffold'].nunique(), 'scaffolds')\n"
         "df.head()"),
    code("fig, ax = plt.subplots(figsize=(6, 3.5))\n"
         "ax.hist(df['IE'], bins=25, color='#2a78d6', edgecolor='white')\n"
         "ax.set_xlabel('Experimental IE (%)'); ax.set_ylabel('Molecules')\n"
         "ax.set_title('Distribution of inhibition efficiency')\n"
         "plt.show()"),
    md("## 4. Featurise\n\nRDKit computes about 210 two-dimensional descriptors per molecule."),
    code("X = rdkit_descriptors(df['smiles'].tolist())\ny = df['IE'].to_numpy()\nprint(X.shape)\nX.iloc[:3, :8]"),
    md("## 5. Why the split matters\n\n"
       "A random split lets close analogues of a test molecule sit in the training set. "
       "A scaffold split keeps whole chemical families together, which is closer to screening genuinely new compounds. "
       "Expect the scaffold numbers to be worse; that gap is the honest estimate."),
    code("model = make_models(C.SEED)['XGBoost']\n"
         "groups = df['scaffold'].to_numpy()\n"
         "for scheme in ['random', 'scaffold']:\n"
         "    res, oof = run_cv(model, X, y, groups, scheme, n_splits=5, n_repeats=1, seed=C.SEED)\n"
         "    print(f\"{scheme:9s} R2={res['R2']:.2f}  RMSE={res['RMSE']:.1f}  MAE={res['MAE']:.1f}\")"),
    md("## 6. Fit a final model and look inside it with SHAP"),
    code("import shap\n"
         "final = make_models(C.SEED)['XGBoost'].fit(X, y)\n"
         "Xt = pd.DataFrame(final['filter'].transform(X), columns=final['filter'].get_feature_names_out())\n"
         "sv = shap.TreeExplainer(final['model']).shap_values(Xt)\n"
         "shap.summary_plot(sv, Xt, max_display=15)"),
    md("## 7. Predict new molecules\n\n"
       "Needs the saved model from `python scripts/run_pipeline.py`. The similarity column flags molecules "
       "far from the training data, where predictions are least reliable."),
    code("from src.predict import predict_ie\n\n"
         "predict_ie(['c1ccc2[nH]c(N)nc2c1', 'CCCCCCCCCCCCCCCC[N+](C)(C)C.[Br-]', 'OC(=O)CCN'])"),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
out = Path(__file__).resolve().parents[1] / "notebooks" / "01_walkthrough.ipynb"
nbf.write(nb, out)
print("wrote", out)
