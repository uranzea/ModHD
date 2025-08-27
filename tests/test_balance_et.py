import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure local package is importable
sys.path.append(str(Path(__file__).resolve().parents[1]))

from tank_model.model import TankModel, ModelConfig
from tank_model.parameters import Parameters
from tank_model.states import States


def test_balance_et_closure():
    # Simple series where ET = PET and Q = 0 is known analytically
    df = pd.DataFrame({"P_mm": [4.0, 10.0, 10.0], "PET_mm": [4.0, 4.0, 4.0]})

    params = Parameters(
        S0_max=10000.0,
        k_qs=0.0,
        k_inf=0.0,
        k_perc=0.0,
        phi=0.0,
        k_qf=0.0,
        k_bf=0.0,
        f_et0=1.0,
        f_et1=0.0,
        alpha=1.0,
        beta=1.0,
    )
    cfg = ModelConfig(area_km2=1.5, route=False, debug_balance=True)
    init_states = States(S0=1000.0)
    model = TankModel(params, cfg, init_states)

    out = model.run(df)

    # Expected ET and Q
    assert np.allclose(out["ET"].values, [4.0, 4.0, 4.0])
    assert np.allclose(out["Qout_mm"].values, 0.0)

    # Residual of cumulative balance should be ~0
    assert abs(out["resid_cum_mm"].iloc[-1]) <= 1e-6

    # Summation checks for mass balance closure
    sumP = df["P_mm"].sum()
    sumET = out["ET"].sum()
    sumQ = out["Qout_mm"].sum()
    initial_storage = init_states.S0 + init_states.S1 + init_states.S2 + init_states.S3
    final_storage = out[["S0", "S1", "S2", "S3"]].iloc[-1].sum()
    deltaS = final_storage - initial_storage

    assert np.isclose(sumP, 24.0)
    assert np.isclose(sumET, 12.0)
    assert np.isclose(sumQ, 0.0)
    assert np.isclose(deltaS, 12.0)
    assert np.isclose(sumP - sumET - sumQ - deltaS, 0.0, atol=1e-6)
