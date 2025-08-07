
import pandas as pd
import numpy as np
from tank_model import TankModel, ModelConfig
from tank_model.parameters import Parameters
from tank_model.calibration import random_search

# 1) Carga forcing (ejemplo sintético diario 1 año)
try:
    df = pd.read_csv("../data/example_forcing.csv", parse_dates=["date"], index_col="date")
except FileNotFoundError:
    idx = pd.date_range("2020-01-01", periods=365, freq="D")
    rng = np.random.default_rng(123)
    P = np.maximum(0, rng.gamma(2.0, 5.0, size=len(idx)) - 3)  # mm/día
    PET = np.full(len(idx), 3.0)  # mm/día
    df = pd.DataFrame({"P_mm": P, "PET_mm": PET}, index=idx)

# 2) Configuración del modelo
cfg = ModelConfig(dt_hours=24.0, area_km2=5.0, route=True)
params = Parameters()

# 3) Fábrica de modelos (para calibración)
def make_model(p):
    return TankModel(params=p, config=cfg)

# 4) Simulación base
m = make_model(params)
sim = m.run(df)
print(sim.head())

# 5) Ejemplo de calibración con datos observados (aquí, falso obs = sim + ruido)
q_obs = sim["Q_m3s"].values * (1 + np.random.normal(0, 0.1, size=len(sim)))
best_p, best_score = random_search(make_model, df, q_obs, n_iter=50, seed=7)
print("Mejor NSE:", best_score)
print("Parámetros calibrados:", best_p)

# 6) Re-simular con parámetros calibrados
m2 = make_model(best_p)
sim2 = m2.run(df)
sim2.to_csv("../data/simulation_output.csv")
print("Guardado: ../data/simulation_output.csv")
