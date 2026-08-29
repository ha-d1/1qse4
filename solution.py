"""Model-editable candidate scoring entry point."""
import numpy as np

from baseline import fit_fm


def score(splits, data_access, target_split: str, seed: int, config: dict) -> np.ndarray:
    """Fit the baseline FM and return row-aligned logits for ``target_split``."""
    del data_access
    params = dict(config)
    params["seed"] = seed
    params.setdefault("verbose", False)
    model, encoded = fit_fm(splits, **params)
    X, _, _ = encoded[target_split]
    return np.asarray(model.predict(X))
