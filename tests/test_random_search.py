import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure local package is importable
sys.path.append(str(Path(__file__).resolve().parents[1]))

from tank_model.calibration import random_search
from tank_model.model import TankModel, ModelConfig


def model_factory(params):
    cfg = ModelConfig(area_km2=10.0)
    return TankModel(params, cfg)


def test_random_search_returns_finite_score():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "P_mm": rng.random(30),
        "PET_mm": rng.random(30)
    })
    q_obs = rng.random(30)

    best_params, best_score = random_search(model_factory, df, q_obs, n_iter=5, seed=0)
    assert best_params is not None
    assert np.isfinite(best_score)
