
from pathlib import Path
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

# Ensure the package is importable when running this script directly
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from tank_model import TankModel, ModelConfig
from tank_model.parameters import Parameters
from tank_model.calibration import random_search
from tank_model.metrics import plot_error_metrics_heatmap


# 1) Carga forcing (ejemplo sintético diario 1 año)
data_dir = ROOT / "data"
forcing_path = data_dir / "Datos_prueba_2.csv"
try:
    df = pd.read_csv(forcing_path, parse_dates=["Fecha"], index_col="Fecha")

except FileNotFoundError:
    idx = pd.date_range("2020-01-01", periods=365, freq="D")
    rng = np.random.default_rng(123)
    P = np.maximum(0, rng.gamma(2.0, 5.0, size=len(idx)) - 3)  # mm/día
    PET = np.full(len(idx), 3.0)  # mm/día
    df = pd.DataFrame({"P_mm": P, "PET_mm": PET}, index=idx)

# 2) Configuración del modelo
cfg = ModelConfig(dt_hours=24.0, area_km2=2.0, route=True)
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
    df_q = pd.read_csv(data_dir / "Datos_prueba_2.csv", parse_dates=["Fecha"], index_col="Fecha")
    q_obs = df_q.reindex(sim.index)["Qobs_m3s"].values
except FileNotFoundError:
    q_obs = sim["Q_m3s"].values * (1 + np.random.normal(0, 0.1, size=len(sim)))

best_p, best_score = random_search(
    make_model,
    df,
    q_obs,
    n_iter=50,
    seed=7,
    catchment_name="Rio_ejemplo",
    log_path="logs/calibraciones.csv",
)

print("Mejor NSE:", best_score)
print("Parámetros calibrados:", best_p)

# 6) Re-simular con parámetros calibrados
m2 = make_model(best_p)
sim2 = m2.run(df)

# Combinar series simuladas, observadas y precipitación
calib_df = pd.DataFrame(
    {
        "Q_sim_m3s": sim2["Q_m3s"],
        "Q_obs_m3s": q_obs,
        "P_mm": df["P_mm"],
    },
    index=sim2.index,
)

# Gráfica de series de tiempo
fig_st, ax1 = plt.subplots(figsize=(10, 5))

# Graficar caudal simulado (rojo) y observado (negro) en el eje principal
calib_df["Q_sim_m3s"].plot(ax=ax1, color="red", label="Simulado")
calib_df["Q_obs_m3s"].plot(ax=ax1, color="black", label="Observado")

ax1.set_ylabel("Q (m³/s)")
ax2 = ax1.twinx()
ax2.bar(calib_df.index, calib_df["P_mm"], color="blue", alpha=0.3, label="P")
ax2.set_ylabel("P (mm)")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
fig_st.tight_layout()
ts_path = data_dir / "calibration_timeseries.png"
fig_st.savefig(ts_path)
plt.show()

# Curva de duración de caudales (FDC)

q_sim_sorted = np.sort(calib_df["Q_sim_m3s"])[::-1]
q_obs_sorted = np.sort(calib_df["Q_obs_m3s"])[::-1]
exceed_prob = np.arange(1, len(q_sim_sorted) + 1) / (len(q_sim_sorted) + 1)

fig_fdc, ax_fdc = plt.subplots(figsize=(8, 5))
ax_fdc.plot(exceed_prob, q_sim_sorted, label="Simulado", color="red")
ax_fdc.plot(exceed_prob, q_obs_sorted, label="Observado", color="black")
ax_fdc.set_xlabel("Probabilidad de excedencia")
ax_fdc.set_ylabel("Q (m³/s)")
ax_fdc.legend()
fig_fdc.tight_layout()
fdc_path = data_dir / "calibration_fdc.png"
fig_fdc.savefig(fdc_path)
plt.show()

# Crear las figuras con Matplotlib
fig, ax1 = plt.subplots(figsize=(9, 3))
calib_df["Q_sim_m3s"].plot(ax=ax1, color="red", label="Simulado")
calib_df["Q_obs_m3s"].plot(ax=ax1, color="black", label="Observado")
ax1.set_ylabel("Caudal (m³/s)")
ax1.legend(loc="upper left")

ax2 = ax1.twinx()
calib_df["P_mm"].plot(kind="bar", ax=ax2, color="blue", alpha=0.3, width=1.0, label="Precipitación")
ax2.set_ylabel("Precipitación (mm)")
ax2.legend(loc="upper right")

ax1.set_title("Caudal simulado vs. observado y precipitación")
fig.tight_layout()
Qfig_path = data_dir / "Qsim vs Qobs vs Pr.png"
fig.savefig(Qfig_path)
plt.show()

plot_error_metrics_heatmap(calib_df["Q_obs_m3s"], calib_df["Q_sim_m3s"])
plt.show()

output_path = data_dir / "simulation_output.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
sim2.to_csv(output_path)
print(f"Guardado: {output_path}")

