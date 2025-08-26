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

# ==============================
# 1) Carga de datos completos
# ==============================
data_dir = ROOT / "data"
forcing_path = data_dir / "Datos_prueba_2.csv"

try:
    df = pd.read_csv(forcing_path, parse_dates=["date"], index_col="date")
    print(f"Datos cargados: {len(df)} registros desde {df.index[0]} hasta {df.index[-1]}")
except FileNotFoundError:
    print("Archivo no encontrado, generando datos sintéticos...")
    idx = pd.date_range("2020-01-01", periods=365, freq="D")
    rng = np.random.default_rng(123)
    P = np.maximum(0, rng.gamma(2.0, 5.0, size=len(idx)) - 3)  # mm/día
    PET = np.full(len(idx), 3.0)  # mm/día
    df = pd.DataFrame({"P_mm": P, "PET_mm": PET}, index=idx)

# ==============================
# 2) Carga de observaciones de caudal
# ==============================
try:
    df_q = pd.read_csv(data_dir / "Datos_prueba_2.csv", parse_dates=["date"], index_col="date")
    # Alinea observaciones al índice completo de forcings
    q_obs_full = df_q.reindex(df.index)["Qobs_m3s"].astype(float).values
    print(f"Observaciones cargadas: {np.sum(~np.isnan(q_obs_full))} valores válidos")
except FileNotFoundError:
    print("Observaciones no encontradas, generando sintéticas...")
    # Configuración temporal del modelo
    cfg = ModelConfig(dt_hours=24.0, area_km2=0.173, route=True)
    params = Parameters()
    m_tmp = TankModel(params=params, config=cfg)
    sim_tmp = m_tmp.run(df)
    q_obs_full = sim_tmp["Qout_mm"].values * (1 + np.random.normal(0, 0.1, size=len(sim_tmp)))

# ==============================
# 3) División de datos para calibración/validación
# ==============================
calib_frac = 0.7      # proporción para calibración
split_date = None     # ej: '2020-09-30' para dividir por fecha específica

if split_date:
    df_calib = df.loc[:split_date].copy()
    df_valid = df.loc[split_date:].copy()
    split_idx = len(df_calib)
else:
    split_idx = int(len(df) * calib_frac)
    df_calib = df.iloc[:split_idx].copy()
    df_valid = df.iloc[split_idx:].copy()

# Particiones de observaciones
q_obs_calib = q_obs_full[:split_idx]
q_obs_valid = q_obs_full[split_idx:]

print(f"Período calibración: {len(df_calib)} días ({df_calib.index[0]} a {df_calib.index[-1]})")
print(f"Período validación: {len(df_valid)} días ({df_valid.index[0]} a {df_valid.index[-1]})")

# ==============================
# 4) Configuración del modelo y función factory
# ==============================
cfg = ModelConfig(dt_hours=24.0, area_km2=0.173, route=True)
params_initial = Parameters()

def make_model(p):
    return TankModel(params=p, config=cfg)

# ==============================
# 5) CALIBRACIÓN - Solo con período de calibración
# ==============================
print("\n=== INICIANDO CALIBRACIÓN ===")
best_params, best_score_calib = random_search(
    make_model,
    df_calib,          # <- SOLO datos de calibración
    q_obs_calib,       # <- SOLO observaciones de calibración
    n_iter=50,
    seed=7,
    catchment_name="Test_catchment",
    log_path="logs/calibration_log.csv",
)

print(f"Mejor NSE (calibración): {best_score_calib:.4f}")
print("Parámetros calibrados:", best_params)

# ==============================
# 6) SIMULACIÓN COMPLETA con parámetros calibrados
# ==============================
print("\n=== SIMULACIÓN CON PARÁMETROS CALIBRADOS ===")
model_calibrated = make_model(best_params)

# Simular toda la serie de una vez para mantener continuidad de estados
sim_complete = model_calibrated.run(df)
print(f"Simulación completa: {len(sim_complete)} registros")

# ==============================
# 7) VALIDACIÓN - Cálculo de métricas por períodos
# ==============================
from tank_model.metrics import nse, bias_pct, rmse

# Extraer simulaciones por período
q_sim_calib = sim_complete["Qout_mm"].iloc[:split_idx].values
q_sim_valid = sim_complete["Qout_mm"].iloc[split_idx:].values

# Métricas para calibración
nse_calib = nse(q_obs_calib, q_sim_calib)
bias_calib = bias_pct(q_obs_calib, q_sim_calib)
rmse_calib = rmse(q_obs_calib, q_sim_calib)

# Métricas para validación
nse_valid = nse(q_obs_valid, q_sim_valid)
bias_valid = bias_pct(q_obs_valid, q_sim_valid)
rmse_valid = rmse(q_obs_valid, q_sim_valid)

# Métricas globales
nse_global = nse(q_obs_full, sim_complete["Qout_mm"].values)
bias_global = bias_pct(q_obs_full, sim_complete["Qout_mm"].values)
rmse_global = rmse(q_obs_full, sim_complete["Qout_mm"].values)

print(f"\n=== RESULTADOS DE VALIDACIÓN ===")
print(f"CALIBRACIÓN - NSE: {nse_calib:.4f}, BIAS: {bias_calib:.2f}%, RMSE: {rmse_calib:.4f}")
print(f"VALIDACIÓN  - NSE: {nse_valid:.4f}, BIAS: {bias_valid:.2f}%, RMSE: {rmse_valid:.4f}")
print(f"GLOBAL      - NSE: {nse_global:.4f}, BIAS: {bias_global:.2f}%, RMSE: {rmse_global:.4f}")

# ==============================
# 8) DataFrame para análisis y visualización
# ==============================
results_df = pd.DataFrame({
    "Q_sim_m3s": sim_complete["Qout_mm"].astype(float),
    "Q_obs_m3s": q_obs_full.astype(float),
    "P_mm": df["P_mm"].astype(float),
    "Period": ["Calibration"] * split_idx + ["Validation"] * len(df_valid)
}, index=sim_complete.index)

# ==============================
# 9) VISUALIZACIONES
# ==============================

# 9a) Serie de tiempo completa con división
fig_ts, ax1 = plt.subplots(figsize=(12, 6))
results_df["Q_sim_m3s"].plot(ax=ax1, color="red", label="Simulado", linewidth=1.5)
results_df["Q_obs_m3s"].plot(ax=ax1, color="black", label="Observado", linewidth=1)
ax1.set_ylabel("Caudal (m³/s)")

ax2 = ax1.twinx()
ax2.bar(results_df.index, results_df["P_mm"], color="blue", alpha=0.3, width=1.0, label="Precipitación")
ax2.set_ylabel("Precipitación (mm)")
ax2.invert_yaxis()  # Invertir eje de precipitación

# Línea de división y anotación
ax1.axvline(results_df.index[split_idx], color="gray", linestyle="--", linewidth=2, alpha=0.8)
ax1.text(results_df.index[split_idx], ax1.get_ylim()[1]*0.95, 
         f"Inicio Validación\nNSE Cal: {nse_calib:.3f}\nNSE Val: {nse_valid:.3f}",
         rotation=0, va="top", ha="left", fontsize=10, 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
ax1.set_title("Calibración y Validación del Modelo - Series de Tiempo")
fig_ts.tight_layout()

# Guardar figura
ts_path = data_dir / "calibration_validation_timeseries.png"
fig_ts.savefig(ts_path, dpi=300)
plt.show()

# 9b) Curvas de duración por período
fig_fdc, (ax_fdc1, ax_fdc2) = plt.subplots(1, 2, figsize=(15, 6))

# Calibración
q_sim_calib_sorted = np.sort(q_sim_calib)[::-1]
q_obs_calib_sorted = np.sort(q_obs_calib)[::-1]
ex_calib = np.arange(1, len(q_sim_calib_sorted) + 1) / (len(q_sim_calib_sorted) + 1) * 100

ax_fdc1.plot(ex_calib, q_sim_calib_sorted, color="red", label="Simulado", linewidth=2)
ax_fdc1.plot(ex_calib, q_obs_calib_sorted, color="black", label="Observado", linewidth=1.5)
ax_fdc1.set_xlabel("Probabilidad de excedencia (%)")
ax_fdc1.set_ylabel("Caudal (m³/s)")
ax_fdc1.set_title(f"Curva de Duración - Calibración\nNSE: {nse_calib:.4f}")
ax_fdc1.legend()
ax_fdc1.grid(True, alpha=0.3)

# Validación
q_sim_valid_sorted = np.sort(q_sim_valid)[::-1]
q_obs_valid_sorted = np.sort(q_obs_valid)[::-1]
ex_valid = np.arange(1, len(q_sim_valid_sorted) + 1) / (len(q_sim_valid_sorted) + 1) * 100

ax_fdc2.plot(ex_valid, q_sim_valid_sorted, color="tomato", label="Simulado", linewidth=2)
ax_fdc2.plot(ex_valid, q_obs_valid_sorted, color="gray", label="Observado", linewidth=1.5)
ax_fdc2.set_xlabel("Probabilidad de excedencia (%)")
ax_fdc2.set_ylabel("Caudal (m³/s)")
ax_fdc2.set_title(f"Curva de Duración - Validación\nNSE: {nse_valid:.4f}")
ax_fdc2.legend()
ax_fdc2.grid(True, alpha=0.3)

fig_fdc.tight_layout()
fdc_path = data_dir / "flow_duration_curves_comparison.png"
fig_fdc.savefig(fdc_path, dpi=300)
plt.show()

# 9c) Scatter plots de correlación
fig_scatter, (ax_s1, ax_s2) = plt.subplots(1, 2, figsize=(15, 6))

# Calibración
ax_s1.scatter(q_obs_calib, q_sim_calib, alpha=0.6, color="blue", s=20)
min_q, max_q = min(np.min(q_obs_calib), np.min(q_sim_calib)), max(np.max(q_obs_calib), np.max(q_sim_calib))
ax_s1.plot([min_q, max_q], [min_q, max_q], 'r--', linewidth=2, label='1:1')
ax_s1.set_xlabel("Caudal Observado (m³/s)")
ax_s1.set_ylabel("Caudal Simulado (m³/s)")
ax_s1.set_title(f"Calibración\nNSE: {nse_calib:.4f}, R²: {nse_calib:.4f}")
ax_s1.legend()
ax_s1.grid(True, alpha=0.3)

# Validación
ax_s2.scatter(q_obs_valid, q_sim_valid, alpha=0.6, color="orange", s=20)
min_q, max_q = min(np.min(q_obs_valid), np.min(q_sim_valid)), max(np.max(q_obs_valid), np.max(q_sim_valid))
ax_s2.plot([min_q, max_q], [min_q, max_q], 'r--', linewidth=2, label='1:1')
ax_s2.set_xlabel("Caudal Observado (m³/s)")
ax_s2.set_ylabel("Caudal Simulado (m³/s)")
ax_s2.set_title(f"Validación\nNSE: {nse_valid:.4f}, R²: {nse_valid:.4f}")
ax_s2.legend()
ax_s2.grid(True, alpha=0.3)

fig_scatter.tight_layout()
scatter_path = data_dir / "scatter_plots_comparison.png"
fig_scatter.savefig(scatter_path, dpi=300)
plt.show()

# ==============================
# 10) Mapa de calor de métricas de error
# ==============================
print("\n=== GENERANDO MAPA DE CALOR DE MÉTRICAS ===")
plot_error_metrics_heatmap(results_df["Q_obs_m3s"], results_df["Q_sim_m3s"])
plt.show()

# ==============================
# 11) Guardar resultados
# ==============================
# Datos de simulación
output_path = data_dir / "simulation_results_complete.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(output_path)
print(f"Resultados guardados en: {output_path}")

# Resumen de métricas
metrics_summary = pd.DataFrame({
    'NSE': [nse_calib, nse_valid, nse_global],
    'BIAS (%)': [bias_calib, bias_valid, bias_global], 
    'RMSE': [rmse_calib, rmse_valid, rmse_global]
}, index=['Calibration', 'Validation', 'Global'])

metrics_path = data_dir / "performance_metrics_summary.csv"
metrics_summary.to_csv(metrics_path)
print(f"Métricas guardadas en: {metrics_path}")
print("\nResumen de Métricas:")
print(metrics_summary)