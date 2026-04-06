
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
    df = pd.read_csv(forcing_path, parse_dates=["date"], index_col="date")

except FileNotFoundError:
    idx = pd.date_range("2020-01-01", periods=365, freq="D")
    rng = np.random.default_rng(123)
    P = np.maximum(0, rng.gamma(2.0, 5.0, size=len(idx)) - 3)  # mm/día
    PET = np.full(len(idx), 3.0)  # mm/día
    df = pd.DataFrame({"P_mm": P, "PET_mm": PET}, index=idx)

# 2) Configuración del modelo
cfg = ModelConfig(dt_hours=24.0, area_km2=0.173, route=True)
params = Parameters()

# 3) Fábrica de modelos (para calibración)
def make_model(p):
    return TankModel(params=p, config=cfg)

# 4) Simulación base
m = make_model(params)
sim = m.run(df)
print(sim.head())

# ==============================
# 4bis) Split Calibración/Validación
# ==============================
calib_frac = 0.7      # proporción para calibración
split_date = None     # ej: '2020-09-30' para dividir por fecha (si no, usa índice)

if split_date:
    df_calib = df.loc[:split_date]
    df_valid = df.loc[split_date:]
else:
    split_idx = int(len(df) * calib_frac)
    df_calib = df.iloc[:split_idx]
    df_valid = df.iloc[split_idx:]

# Útil más adelante
split_idx = len(df_calib)


# ==============================
# 5) Observaciones (alineadas al índice de forcings)
# ==============================
try:
    df_q = pd.read_csv(data_dir / "Datos_prueba_2.csv", parse_dates=["date"], index_col="date")
    # Alinea observaciones al índice completo de forcings
    q_obs_full = df_q.reindex(df.index)["Qobs_m3s"].astype(float).values
except FileNotFoundError:
    # Si no hay observaciones, sintetiza a partir de una corrida base
    m_tmp = make_model(params)
    sim_tmp = m_tmp.run(df)
    q_obs_full = sim_tmp["Qout_mm"].values * (1 + np.random.normal(0, 0.1, size=len(sim_tmp)))

# Particiones para calibración/validación
q_obs_calib = q_obs_full[:split_idx]
q_obs_valid = q_obs_full[split_idx:]

# ==============================
# 6) Calibración SOLO con df_calib
# ==============================
best_p, best_score = random_search(
    make_model,
    df_calib,          # <- solo calibración
    q_obs_calib,
    n_iter=50,
    seed=7,
    catchment_name="Test_catchment",
    log_path="logs/calibration_log.csv",
)
print("Mejor NSE (calibración):", best_score)
print("Parámetros calibrados:", best_p)

# ==============================
# 7) Re-simular con best_p en calib y validación
# ==============================
m2 = make_model(best_p)
sim_calib = m2.run(df_calib)
sim_valid = m2.run(df_valid)

# Reconstruir una sola serie continua (opcional)
sim2 = pd.concat([sim_calib, sim_valid], axis=0)

# ==============================
# 8) DataFrame combinado para análisis
# ==============================
calib_df = pd.DataFrame({
    "Q_sim_m3s": sim2["Qout_mm"].astype(float),
    "Q_obs_m3s": q_obs_full.astype(float),
    "P_mm": df.loc[sim2.index, "P_mm"].astype(float),
}, index=sim2.index)

# ==============================
# 9) Gráfica de series con línea de split
# ==============================
fig_st, ax1 = plt.subplots(figsize=(10, 5))
calib_df["Q_sim_m3s"].plot(ax=ax1, color="red", label="Simulado")
calib_df["Q_obs_m3s"].plot(ax=ax1, color="black", label="Observado")
ax1.set_ylabel("Q (m³/s)")

ax2 = ax1.twinx()
ax2.bar(calib_df.index, calib_df["P_mm"], color="blue", alpha=0.3, label="P")
ax2.set_ylabel("P (mm)")

# Línea vertical que marca el inicio de la validación
ax1.axvline(calib_df.index[split_idx], color="gray", linestyle="--", linewidth=1)
ax1.text(calib_df.index[split_idx], ax1.get_ylim()[1]*0.95, "Inicio Validación",
         rotation=90, va="top", ha="right", fontsize=9, color="gray")

ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
fig_st.tight_layout()
(ts_path := data_dir / "calibration_timeseries_split.png")
fig_st.savefig(ts_path)
plt.show()

# ==============================
# 10) FDC separada (calib vs valid)
# ==============================
q_sim_calib_sorted = np.sort(calib_df["Q_sim_m3s"].iloc[:split_idx])[::-1]
q_obs_calib_sorted = np.sort(calib_df["Q_obs_m3s"].iloc[:split_idx])[::-1]
ex_calib = np.arange(1, len(q_sim_calib_sorted) + 1) / (len(q_sim_calib_sorted) + 1)

q_sim_valid_sorted = np.sort(calib_df["Q_sim_m3s"].iloc[split_idx:])[::-1]
q_obs_valid_sorted = np.sort(calib_df["Q_obs_m3s"].iloc[split_idx:])[::-1]
ex_valid = np.arange(1, len(q_sim_valid_sorted) + 1) / (len(q_sim_valid_sorted) + 1)

fig_fdc, ax_fdc = plt.subplots(figsize=(8, 5))
ax_fdc.plot(ex_calib, q_sim_calib_sorted, label="Simulado - Calib", color="red")
ax_fdc.plot(ex_calib, q_obs_calib_sorted, label="Observado - Calib", color="black")
ax_fdc.plot(ex_valid, q_sim_valid_sorted, label="Simulado - Valid", color="tomato")
ax_fdc.plot(ex_valid, q_obs_valid_sorted, label="Observado - Valid", color="gray")
ax_fdc.set_xlabel("Probabilidad de excedencia")
ax_fdc.set_ylabel("Q (m³/s)")
ax_fdc.legend()
fig_fdc.tight_layout()
(fdc_path := data_dir / "calibration_fdc_split.png")
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

