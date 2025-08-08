
from pathlib import Path
import pandas as pd
import numpy as np
import sys

# Ensure the package is importable when running this script directly
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from tank_model import TankModel, ModelConfig
from tank_model.parameters import Parameters
from tank_model.calibration import random_search

# 1) Carga forcing (ejemplo sintético diario 1 año)
data_dir = ROOT / "data"
forcing_path = data_dir / "example_forcing.csv"
try:
    df = pd.read_csv(forcing_path, parse_dates=["date"], index_col="date")

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

# 5) Calibración con datos observados
try:
    df_q = pd.read_csv(data_dir / "example_discharge.csv", parse_dates=["date"], index_col="date")
    q_obs = df_q.reindex(sim.index)["Qobs_m3s"].values
except FileNotFoundError:
    q_obs = sim["Q_m3s"].values * (1 + np.random.normal(0, 0.1, size=len(sim)))

best_p, best_score = random_search(
    make_model,
    df,
    q_obs,
    n_iter=50,
    seed=7,
    catchment_name="Rio_concepto",
    log_path="logs/calibraciones.csv",
)

print("Mejor NSE:", best_score)
print("Parámetros calibrados:", best_p)

# 6) Re-simular con parámetros calibrados
m2 = make_model(best_p)
sim2 = m2.run(df)

output_path = data_dir / "simulation_output.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
sim2.to_csv(output_path)
print(f"Guardado: {output_path}")

