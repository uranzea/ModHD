
import numpy as np
from .parameters import Parameters
from .metrics import nse

def random_search(model_factory, df, q_obs, n_iter=200, seed=42, bounds=None):
    """Búsqueda aleatoria simple maximizando NSE.

    model_factory: función que crea un modelo dado un Parameters

    bounds: dict con (min, max) por parámetro a calibrar

    Retorna (best_params, best_score)
    """
    rng = np.random.default_rng(seed)
    if bounds is None:
        bounds = {
            "k_qs": (0.01, 0.5),
            "alpha": (1.0, 2.5),
            "S0_max": (5, 150),
            "k_inf": (0.01, 0.5),
            "k_perc": (0.005, 0.2),
            "beta": (0.8, 1.5),
            "phi": (0.2, 0.9),
            "k_qf": (0.05, 0.8),
            "k_bf": (0.001, 0.2),
            "n_r": (1, 4),
            "k_r": (1.0, 72.0),
            "f_et0": (0.0, 0.2),
            "f_et1": (0.0, 0.1),
        }
    best_score = -np.inf
    best_p = None

    keys = list(bounds.keys())
    for _ in range(n_iter):
        p = Parameters()
        for k in keys:
            lo, hi = bounds[k]
            if k == "n_r":
                setattr(p, k, int(rng.integers(lo, hi+1)))
            else:
                setattr(p, k, rng.uniform(lo, hi))
        m = model_factory(p)
        sim = m.run(df)["Q_m3s"].values
        score = nse(q_obs, sim)
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_p = p
    return best_p, best_score
