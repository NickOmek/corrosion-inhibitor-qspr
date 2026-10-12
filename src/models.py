"""Model zoo. Every model is a Pipeline so preprocessing is refitted per CV fold."""
from __future__ import annotations

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from xgboost import XGBRegressor

from .config import SEED
from .features import DescriptorFilter

import numpy as np


def _scaled(model) -> Pipeline:
    """Linear/NN models need comparable feature scales; descriptors are heavy-tailed,
    so a quantile transform is more robust than standard scaling."""
    return Pipeline(
        [
            ("filter", DescriptorFilter()),
            ("scale", QuantileTransformer(n_quantiles=100, output_distribution="normal")),
            ("model", model),
        ]
    )


def _tree(model) -> Pipeline:
    return Pipeline([("filter", DescriptorFilter()), ("model", model)])


def make_models(seed: int = SEED) -> dict[str, Pipeline]:
    return {
        "Mean baseline": Pipeline([("model", DummyRegressor(strategy="mean"))]),
        "Ridge": _scaled(RidgeCV(alphas=np.logspace(-1, 4, 25))),
        "Random forest": _tree(
            RandomForestRegressor(
                n_estimators=500, max_features=0.33, n_jobs=-1, random_state=seed
            )
        ),
        "XGBoost": _tree(
            XGBRegressor(
                n_estimators=400,
                learning_rate=0.03,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.5,
                reg_lambda=2.0,
                n_jobs=-1,
                random_state=seed,
                verbosity=0,
            )
        ),
        "Neural network (MLP)": _scaled(
            MLPRegressor(
                hidden_layer_sizes=(128, 64),
                alpha=1e-2,
                early_stopping=True,
                validation_fraction=0.15,
                max_iter=1000,
                random_state=seed,
            )
        ),
    }
