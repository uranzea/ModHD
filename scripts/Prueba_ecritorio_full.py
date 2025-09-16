from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Configuración de rutas para importar el paquete local
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from tank_model.model import TankModel, ModelConfig
from tank_model.parameters import Parameters
from tank_model.metrics import nse, bias_pct, rmse, plot_error_metrics_heatmap

# ======================================================
# 1) Lectura de forcings (precipitación y evapotranspiración)
# ======================================================
data_dir = ROOT / "data"
forcing_path = data_dir / "Datos_prueba_2.csv"

try:
    df = pd.read_csv(forcing_path, parse_dates=["date"], index_col="date")
except FileNotFoundError:
    idx = pd.date_range("2020-01-01", periods=365, freq="D")
    rng = np.random.default_rng(123)
    P = np.maximum(0, rng.gamma(2.0, 5.0, size=len(idx)) - 3)
    PET = np.full(len(idx), 3.0)
    df = pd.DataFrame({"P_mm": P, "PET_mm": PET}, index=idx)

# ======================================================
# 2) Lectura de observaciones de caudal (o generación sintética)
# ======================================================
try:
    df_q = pd.read_csv(data_dir / "Datos_prueba_2.csv", parse_dates=["date"], index_col="date")
    q_obs_full = df_q.reindex(df.index)["Qobs_m3s"].astype(float).values
except FileNotFoundError:
    cfg_tmp = ModelConfig(dt_hours=24.0, area_km2=0.173, route=True)
    params_tmp = Parameters()
    model_tmp = TankModel(params_tmp, cfg_tmp)
    sim_tmp = model_tmp.run(df)
    q_obs_full = sim_tmp["Qout_mm"].values * (1 + np.random.normal(0, 0.1, size=len(sim_tmp)))

# ======================================================
# 3) División de datos en calibración y validación
# ======================================================
calib_frac = 0.7
split_idx = int(len(df) * calib_frac)
df_calib = df.iloc[:split_idx].copy()
df_valid = df.iloc[split_idx:].copy()
q_obs_calib = q_obs_full[:split_idx]
q_obs_valid = q_obs_full[split_idx:]

# ======================================================
# 4) Configuración del modelo y parámetros iniciales
#    (modificar manualmente según necesidad)
# ======================================================
cfg = ModelConfig(dt_hours=24.0, area_km2=0.173, route=True)
params = Parameters()
# Ejemplo de cambios manuales:
# params.S0_max = 60
# params.k_qs = 0.2

# ======================================================
# 5) Calibración mediante búsqueda aleatoria (sin funciones)
# ======================================================
bounds = {
    "S0_max": (25, 80),
    "alpha":  (1.30, 2.00),
    "k_qs":   (0.12, 0.30),
    "k_inf":  (0.15, 0.35),
    "k_perc": (0.03, 0.10),
    "phi":    (0.55, 0.80),
    "k_qf":   (0.15, 0.35),
    "k_bf":   (0.04, 0.15),
    "f_et0":  (0.08, 0.16),
    "f_et1":  (0.03, 0.08),
    "n_r":    (1, 2),
    "k_r":    (6.0, 18.0)
}

n_iter = 50
rng = np.random.default_rng(7)
best_score = -np.inf
best_params = None

for i in range(n_iter):
    p = Parameters()
    for name, (lo, hi) in bounds.items():
        if name == "n_r":
            setattr(p, name, int(rng.integers(lo, hi + 1)))
        else:
            setattr(p, name, rng.uniform(lo, hi))
    model = TankModel(params=p, config=cfg)
    sim_df = model.run(df_calib)
    q_sim = sim_df["Qout_mm"].values
    nse_lin = nse(q_obs_calib, q_sim)
    nse_log = nse(np.log1p(q_obs_calib), np.log1p(q_sim))
    thr = np.nanpercentile(q_obs_calib, 99)
    peak_bias = (np.nansum(q_sim[q_obs_calib >= thr]) - np.nansum(q_obs_calib[q_obs_calib >= thr])) / (np.nansum(q_obs_calib[q_obs_calib >= thr]) + 1e-9)
    score = 0.45 * nse_lin + 0.35 * nse_log + 0.20 * (-abs(peak_bias))
    if np.isfinite(score) and score > best_score:
        best_score = score
        best_params = p

print("Mejor puntuación de búsqueda aleatoria:", best_score)
print("Parámetros calibrados:", best_params)

# ======================================================
# 5b) Optimización determinística con SciPy
# ======================================================
param_names = [
    "S0_max", "alpha", "k_qs", "k_inf", "k_perc",
    "phi", "k_qf", "k_bf", "f_et0", "f_et1", "k_r"
]
x0 = np.array([getattr(best_params, name) for name in param_names])
bounds_opt = [bounds[name] for name in param_names]

error_history = []

def objective(x):
    p = Parameters()
    p.n_r = best_params.n_r
    for name, val in zip(param_names, x):
        setattr(p, name, val)
    model = TankModel(params=p, config=cfg)
    q_sim = model.run(df_calib)["Qout_mm"].values
    return 1.0 - nse(q_obs_calib, q_sim)

error_history.append(objective(x0))

def callback(xk):
    error_history.append(objective(xk))

res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds_opt, callback=callback)

best_params_opt = Parameters()
best_params_opt.n_r = best_params.n_r
for name, val in zip(param_names, res.x):
    setattr(best_params_opt, name, val)

best_params = best_params_opt
print("Error final (1 - NSE) SciPy:", res.fun)
print("Parámetros optimizados SciPy:", best_params)

# ======================================================
# 6) Simulación completa con parámetros calibrados
# ======================================================
model_final = TankModel(params=best_params, config=cfg)
sim_complete = model_final.run(df)

# ======================================================
# 7) Cálculo de métricas de desempeño
# ======================================================
q_sim_calib = sim_complete["Qout_mm"].iloc[:split_idx].values
q_sim_valid = sim_complete["Qout_mm"].iloc[split_idx:].values

nse_calib = nse(q_obs_calib, q_sim_calib)
bias_calib = bias_pct(q_obs_calib, q_sim_calib)
rmse_calib = rmse(q_obs_calib, q_sim_calib)

nse_valid = nse(q_obs_valid, q_sim_valid)
bias_valid = bias_pct(q_obs_valid, q_sim_valid)
rmse_valid = rmse(q_obs_valid, q_sim_valid)

nse_global = nse(q_obs_full, sim_complete["Qout_mm"].values)
bias_global = bias_pct(q_obs_full, sim_complete["Qout_mm"].values)
rmse_global = rmse(q_obs_full, sim_complete["Qout_mm"].values)

print("Métricas calibración - NSE:{:.3f} BIAS:{:.2f}% RMSE:{:.3f}".format(nse_calib, bias_calib, rmse_calib))
print("Métricas validación  - NSE:{:.3f} BIAS:{:.2f}% RMSE:{:.3f}".format(nse_valid, bias_valid, rmse_valid))
print("Métricas globales    - NSE:{:.3f} BIAS:{:.2f}% RMSE:{:.3f}".format(nse_global, bias_global, rmse_global))

# ======================================================
# 8) DataFrame combinado para análisis
# ======================================================
results_df = pd.DataFrame({
    "Q_sim_m3s": sim_complete["Qout_mm"].astype(float),
    "Q_obs_m3s": q_obs_full.astype(float),
    "P_mm": df["P_mm"].astype(float)
}, index=sim_complete.index)

# ======================================================
# 9) Gráficas
# ======================================================
fig_ts, ax1 = plt.subplots(figsize=(12, 6))
results_df["Q_sim_m3s"].plot(ax=ax1, color="red", label="Simulado")
results_df["Q_obs_m3s"].plot(ax=ax1, color="black", label="Observado")
ax1.set_ylabel("Caudal (m³/s)")
ax1.legend(loc="upper left")

ax2 = ax1.twinx()
ax2.bar(results_df.index, results_df["P_mm"], color="blue", alpha=0.3, width=1.0, label="Precipitación")
ax2.set_ylabel("Precipitación (mm)")
ax2.legend(loc="upper right")

ax1.axvline(results_df.index[split_idx], color="gray", linestyle="--", linewidth=1)
ax1.text(results_df.index[split_idx], ax1.get_ylim()[1]*0.9, "Inicio Validación", rotation=90, va="top", ha="right")
fig_ts.tight_layout()
(fig_ts_path := data_dir / "prueba_timeseries.png")
fig_ts.savefig(fig_ts_path)

q_sim_calib_sorted = np.sort(q_sim_calib)[::-1]
q_obs_calib_sorted = np.sort(q_obs_calib)[::-1]
ex_calib = np.arange(1, len(q_sim_calib_sorted) + 1) / (len(q_sim_calib_sorted) + 1)

q_sim_valid_sorted = np.sort(q_sim_valid)[::-1]
q_obs_valid_sorted = np.sort(q_obs_valid)[::-1]
ex_valid = np.arange(1, len(q_sim_valid_sorted) + 1) / (len(q_sim_valid_sorted) + 1)

fig_fdc, ax_fdc = plt.subplots(figsize=(8,5))
ax_fdc.plot(ex_calib, q_sim_calib_sorted, label="Sim Calib", color="red")
ax_fdc.plot(ex_calib, q_obs_calib_sorted, label="Obs Calib", color="black")
ax_fdc.plot(ex_valid, q_sim_valid_sorted, label="Sim Valid", color="tomato")
ax_fdc.plot(ex_valid, q_obs_valid_sorted, label="Obs Valid", color="gray")
ax_fdc.set_xlabel("Probabilidad de excedencia")
ax_fdc.set_ylabel("Q (m³/s)")
ax_fdc.legend()
fig_fdc.tight_layout()
(fig_fdc_path := data_dir / "prueba_fdc.png")
fig_fdc.savefig(fig_fdc_path)

fig_err, ax_err = plt.subplots(figsize=(6,4))
ax_err.plot(error_history, marker="o")
ax_err.set_xlabel("Iteración")
ax_err.set_ylabel("1 - NSE")
ax_err.set_title("Evolución del error")
fig_err.tight_layout()
(fig_err_path := data_dir / "prueba_error_evolucion.png")
fig_err.savefig(fig_err_path)

plot_error_metrics_heatmap(results_df["Q_obs_m3s"], results_df["Q_sim_m3s"])
(fig_heatmap_path := data_dir / "prueba_heatmap.png")
plt.savefig(fig_heatmap_path)
plt.show()

# ======================================================
# 10) Pruebas básicas de balance hídrico
# ======================================================
initial_storage = 0.0
final_storage = sim_complete[["S0", "S1", "S2", "S3"]].iloc[-1].sum()

a = df["P_mm"].sum()
b = sim_complete["ET_mm"].sum()
c = sim_complete["Qraw_mm"].sum()
residual = a - b - c - (final_storage - initial_storage)
print("Residual de balance hídrico:", residual)
assert np.isclose(residual, 0.0, atol=1e-6)

output_path = data_dir / "prueba_simulacion.csv"
results_df.to_csv(output_path)
print("Resultados guardados en:", output_path)
