import sys
from pathlib import Path

import numpy as np

# Asegurar que el paquete local esté en el path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from tank_model.model import TankModel, ModelConfig
from tank_model.parameters import Parameters
from tank_model.states import States


def test_mass_conservation_when_s0_insufficient():
    params = Parameters(
        S0_max=1000.0,
        k_qs=0.8,
        k_inf=0.5,
        k_perc=0.0,
        phi=0.0,
        k_qf=0.0,
        k_bf=0.0,
        f_et0=0.0,
        f_et1=0.0,
        alpha=1.0,
        beta=1.0,
    )
    cfg = ModelConfig(area_km2=1.5, route=False)
    init_states = States(S0=10.0)
    model = TankModel(params, cfg, init_states)

    res = model.step(P=0.0, PET=0.0)

    assert np.isclose(res["Qs"] + res["I"], 10.0)
    assert res["S0"] == 0.0
