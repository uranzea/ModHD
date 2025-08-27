# -*- coding: utf-8 -*-
"""
Desk-check de calibración/validación (SIN funciones).
Objetivo: detectar rápidamente fuentes de error (unidades, dt, área, balances, sesgos) con impresiones detalladas.
Requisitos:
- CSV con columnas: date, P_mm, PET_mm, Qobs_m3s (si Qobs no está, se genera sintético).
- Paquete/modelo disponible como `tank_model` o, en su defecto, archivos locales (model.py, parameters.py, metrics.py).
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure the package is importable when running this script directly
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# ==============================
# 0) Imports del modelo (intenta paquete y luego local)
# ==============================
from tank_model import TankModel, ModelConfig
from tank_model.parameters import Parameters
from tank_model.calibration import random_search
from tank_model.metrics import nse, kge, rmse, bias_pct


# ---------------------------------------------------------------------
# 1) Paths y lectura de datos
# ---------------------------------------------------------------------
data_dir = ROOT / "data"
data_dir.mkdir(parents=True, exist_ok=True)

csv_path = data_dir / "Datos_prueba_2.csv"  # Ajusta si tu archivo tiene otro nombre

print("\n=== LECTURA DE DATOS ===")
if csv_path.exists():
    df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    print(f"Archivo encontrado: {csv_path.name}")
else:
    print(f"Archivo NO encontrado en {csv_path}. Generando datos sintéticos de ejemplo.")
    idx = pd.date_range("2020-01-01", periods=365, freq="D")
    rng = np.random.default_rng(42)
    P = np.maximum(0, rng.gamma(2.0, 5.0, size=len(idx)) - 3)  # mm/∆t
    PET = np.full(len(idx), 3.0)  # mm/∆t
    Qobs = np.nan * np.zeros(len(idx))  # si no hay observados, se deja NaN
    df = pd.DataFrame({"P_mm": P, "PET_mm": PET, "Qobs_m3s": Qobs}, index=idx)

# Forzar orden y limpieza básica
df = df.sort_index()
for c in ("P_mm", "PET_mm"):
    if c not in df.columns:
        raise ValueError(f"Falta columna requerida '{c}' en el CSV.")

if "Qobs_m3s" not in df.columns:
    df["Qobs_m3s"] = np.nan

print(f"Registros: {len(df)}   Rango: {df.index.min()} → {df.index.max()}")
print(f"Columnas disponibles: {list(df.columns)}")

# ---------------------------------------------------------------------
# 2) Diagnóstico de resolución temporal y dt
# ---------------------------------------------------------------------
print("\n=== DIAGNÓSTICO TEMPORAL (dt) ===")
if len(df.index) > 1:
    dt_series = pd.Series(df.index).diff().iloc[1:]
    med_dt = pd.to_timedelta(dt_series.median())
    med_dt_s = med_dt.total_seconds()
    med_dt_h = med_dt_s / 3600.0
else:
    med_dt_s = 24 * 3600.0
    med_dt_h = 24.0

print(f"Δt mediano inferido: {med_dt_h:.2f} h ({med_dt_s:.0f} s)")
# Usa el Δt inferido para el modelo (ajústalo si deseas forzarlo)
dt_hours = med_dt_h if med_dt_h > 0 else 24.0

# ---------------------------------------------------------------------
# 3) Área y conversiones de unidad
# ---------------------------------------------------------------------
print("\n=== ÁREA Y CONVERSIÓN DE UNIDADES ===")
area_km2 = 40  # AJUSTA AQUÍ tu área (km²)
seconds = dt_hours * 3600.0
area_m2 = area_km2 * 1e6
mm_to_m3s = (area_m2 * 1e-3) / seconds  # 1 mm/∆t → ? m³/s

print(f"Área declarada: {area_km2} km² ({area_m2:.0f} m²)")
print(f"dt_hours usado: {dt_hours:.2f} h → segundos/∆t = {seconds:.0f} s")
print(f"Conversión: 1.0 mm/∆t equivale a {mm_to_m3s:.6e} m³/s")

# ---------------------------------------------------------------------
# 4) Revisión de calidad de P/PET/Q
# ---------------------------------------------------------------------
print("\n=== CALIDAD DE P, PET y Qobs ===")
def _count_neg_nan(x):
    x = pd.Series(x, dtype=float)
    return (int(np.sum(x.isna())), int(np.sum(x < 0)))

P_nan, P_neg = _count_neg_nan(df["P_mm"])
E_nan, E_neg = _count_neg_nan(df["PET_mm"])
Q_nan, Q_neg = _count_neg_nan(df["Qobs_m3s"])  # Qobs puede ser NaN

print(f"P_mm: NaN={P_nan}, Negativos={P_neg}, Suma={np.nansum(df['P_mm']):.2f} mm")
print(f"PET_mm: NaN={E_nan}, Negativos={E_neg}, Suma={np.nansum(df['PET_mm']):.2f} mm")
print(f"Qobs_m3s: NaN={Q_nan}, Negativos={Q_neg}")

# ---------------------------------------------------------------------
# 5) Split Calibración/Validación
# ---------------------------------------------------------------------
print("\n=== SPLIT CALIBRACIÓN/VALIDACIÓN ===")
calib_frac = 0.7
split_idx = int(len(df) * calib_frac)
df_calib = df.iloc[:split_idx].copy()
df_valid = df.iloc[split_idx:].copy()

q_obs_full = df["Qobs_m3s"].astype(float).values
q_obs_calib = q_obs_full[:split_idx]
q_obs_valid = q_obs_full[split_idx:]

print(f"Período calibración: {len(df_calib)} registros ({df_calib.index[0]} → {df_calib.index[-1]})")
print(f"Período validación: {len(df_valid)} registros ({df_valid.index[0]} → {df_valid.index[-1]})")
print(f"Obs válidos en calibración: {np.sum(~np.isnan(q_obs_calib))}, en validación: {np.sum(~np.isnan(q_obs_valid))}")

# ---------------------------------------------------------------------
# 6) Configuración inicial del modelo
# ---------------------------------------------------------------------
print("\n=== CONFIGURACIÓN DEL MODELO ===")
cfg = ModelConfig(dt_hours=24.0, area_km2=40.0,
                  route=True,
                  debug_balance=True,
                  debug_csv_path="../data/diag_balance_run.csv")
print(cfg)

# ---------------------------------------------------------------------
# 7) Búsqueda aleatoria inline (SIN funciones)
# ---------------------------------------------------------------------
print("\n=== CALIBRACIÓN (BÚSQUEDA ALEATORIA) ===")
rng = np.random.default_rng(123)
n_iter = 150  # ajusta si quieres explorar más/menos

# Límites de parámetros (puedes ajustarlos)
bounds = {
    "S0_max": (10.0, 150.0),
    "alpha":  (1.0,  2.5),
    "beta":   (0.8,  1.5),
    "k_qs":   (0.01, 0.40),
    "k_inf":  (0.01, 0.50),
    "k_perc": (0.005,0.20),
    "phi":    (0.20, 0.90),
    "k_qf":   (0.05, 0.60),
    "k_bf":   (0.001,0.20),
    "n_r":    (1,    4),     # entero
    "k_r":    (1.0,  72.0),  # horas si dt_hours está en horas
    "f_et0":  (0.00, 0.20),
    "f_et1":  (0.00, 0.10),
}

best_score = -np.inf
best_params = None

# Avance simple sin funciones:
for it in range(1, n_iter+1):
    p = Parameters()
    # muestreo de parámetros
    p.S0_max = rng.uniform(*bounds["S0_max"])
    p.alpha  = rng.uniform(*bounds["alpha"])
    p.beta   = rng.uniform(*bounds["beta"])
    p.k_qs   = rng.uniform(*bounds["k_qs"])
    p.k_inf  = rng.uniform(*bounds["k_inf"])
    p.k_perc = rng.uniform(*bounds["k_perc"])
    p.phi    = rng.uniform(*bounds["phi"])
    p.k_qf   = rng.uniform(*bounds["k_qf"])
    p.k_bf   = rng.uniform(*bounds["k_bf"])
    p.n_r    = int(rng.integers(bounds["n_r"][0], bounds["n_r"][1]+1))
    p.k_r    = rng.uniform(*bounds["k_r"])
    p.f_et0  = rng.uniform(*bounds["f_et0"])
    p.f_et1  = rng.uniform(*bounds["f_et1"])

    # Instanciar y correr sobre calibración
    m = TankModel(params=p, config=cfg)
    sim_calib = m.run(df_calib)

    # Alinear por índice (por si el ruteo agrega "cola")
    sim_calib = sim_calib.reindex(df_calib.index).iloc[:len(df_calib)]

    q_sim_calib = sim_calib["Q_m3s"].astype(float).values  # SIEMPRE comparar en m³/s
    # Nota: usa KGE para calibración (más robusto en sesgo y variabilidad)
    score = kge(q_obs_calib, q_sim_calib)

    if score > best_score:
        best_score = score
        best_params = p

    if it % 25 == 0 or it == 1:
        # Impresión de diagnóstico periódico
        nse_now  = nse(q_obs_calib, q_sim_calib)
        bias_now = bias_pct(q_obs_calib, q_sim_calib)
        print(f"Iter {it:3d}/{n_iter}  KGE={score: .4f}  NSE={nse_now: .4f}  Bias={bias_now: .2f}%"
              f"  (n_r={p.n_r}, k_r={p.k_r:.1f}, S0_max={p.S0_max:.1f}, k_qs={p.k_qs:.3f}, alpha={p.alpha:.2f})")

# Resumen de calibración
print("\n=== MEJORES PARÁMETROS (KGE) ===")
print(best_params)
print(f"Mejor KGE (calibración): {best_score:.4f}")

# ---------------------------------------------------------------------
# 8) Simulación completa con mejores parámetros
# ---------------------------------------------------------------------
print("\n=== SIMULACIÓN COMPLETA CON PARÁMETROS ÓPTIMOS ===")
m_best = TankModel(params=best_params, config=cfg)
sim_full = m_best.run(df)

# Alinear simulación y observaciones (si la serie simulada es más larga por la cola)
sim_full = sim_full.reindex(df.index).iloc[:len(df)]

# Asegurar columnas esperadas
for c in ("Q_m3s", "Qout_mm"):
    if c not in sim_full.columns:
        raise RuntimeError(f"Salida del modelo no contiene la columna '{c}'. Columnas: {list(sim_full.columns)}")

# Construir resultados alineados
results = pd.DataFrame({
    "Q_sim_m3s": sim_full["Q_m3s"].astype(float),
    "Q_sim_mm":  sim_full["Qout_mm"].astype(float),
    "P_mm":      df["P_mm"].astype(float),
    "PET_mm":    df["PET_mm"].astype(float),
    "Q_obs_m3s": df["Qobs_m3s"].astype(float)
}, index=df.index)

# ---------------------------------------------------------------------
# 9) Métricas Cal/Val/Global (m³/s)
# ---------------------------------------------------------------------
print("\n=== MÉTRICAS (m³/s) ===")
q_sim_cal = results["Q_sim_m3s"].iloc[:split_idx].values
q_sim_val = results["Q_sim_m3s"].iloc[split_idx:].values

nse_cal  = nse(q_obs_calib, q_sim_cal)
kge_cal  = kge(q_obs_calib, q_sim_cal)
rmse_cal = rmse(q_obs_calib, q_sim_cal)
bias_cal = bias_pct(q_obs_calib, q_sim_cal)

nse_val  = nse(q_obs_valid, q_sim_val)
kge_val  = kge(q_obs_valid, q_sim_val)
rmse_val = rmse(q_obs_valid, q_sim_val)
bias_val = bias_pct(q_obs_valid, q_sim_val)

nse_glb  = nse(q_obs_full, results["Q_sim_m3s"].values)
kge_glb  = kge(q_obs_full, results["Q_sim_m3s"].values)
rmse_glb = rmse(q_obs_full, results["Q_sim_m3s"].values)
bias_glb = bias_pct(q_obs_full, results["Q_sim_m3s"].values)

print(f"CALIBRACIÓN:  NSE={nse_cal:.4f}  KGE={kge_cal:.4f}  RMSE={rmse_cal:.4f}  BIAS={bias_cal:.2f}%")
print(f"VALIDACIÓN:   NSE={nse_val:.4f}  KGE={kge_val:.4f}  RMSE={rmse_val:.4f}  BIAS={bias_val:.2f}%")
print(f"GLOBAL:       NSE={nse_glb:.4f}  KGE={kge_glb:.4f}  RMSE={rmse_glb:.4f}  BIAS={bias_glb:.2f}%")

# ---------------------------------------------------------------------
# 10) Balance hídrico y chequeos de consistencia
# ---------------------------------------------------------------------
print("\n=== BALANCE HÍDRICO (mm por ∆t) ===")
# Usamos Q_sim_mm (ya en mm/∆t). Comparamos totales.
sum_P   = float(np.nansum(results["P_mm"].values))
sum_PET = float(np.nansum(results["PET_mm"].values))
sum_Qmm = float(np.nansum(results["Q_sim_mm"].values))

print(f"ΣP =   {sum_P:10.2f} mm")
print(f"ΣPET = {sum_PET:10.2f} mm")
print(f"ΣQsim = {sum_Qmm:10.2f} mm")

# Estimación de almacenamiento (si están en la salida)
stor_cols = [c for c in ("S0","S1","S2","S3") if c in sim_full.columns]
if len(stor_cols) == 4:
    S_start = float(sim_full[stor_cols].iloc[0].sum())
    S_end   = float(sim_full[stor_cols].iloc[-1].sum())
    dS = S_end - S_start
    print(f"ΔS = {dS:10.2f} mm (Sini={S_start:.2f} → Sfin={S_end:.2f})")
else:
    dS = np.nan
    print("No se encontraron columnas de almacenamiento (S0..S3) en la salida para cierre de balance.")

# Cierre simple del balance (aprox): P - PET - Q - ΔS
resid = sum_P - sum_PET - sum_Qmm - (dS if np.isfinite(dS) else 0.0)
print(f"Residuo (P - PET - Q - ΔS) = {resid:.2f} mm  → (ideal ≈ 0, diferencias por discretización/redondeo)")

# Chequeo de unidades: reconstruir Q_sim_m3s desde mm/∆t y comparar con Q_sim_m3s reportado
q_m3s_rebuild = results["Q_sim_mm"].values * (area_m2 * 1e-3) / (dt_hours * 3600.0)
diff_m3s = float(np.nanmean(np.abs(q_m3s_rebuild - results["Q_sim_m3s"].values)))
print(f"Chequeo unidades: |Q_m3s(rebuild) - Q_m3s| promedio = {diff_m3s:.6e} m³/s (debe ser ~0)")

# ---------------------------------------------------------------------
# 11) Percentiles y Q95 (m³/s) útiles para análisis
# ---------------------------------------------------------------------
def _p(series, pctl):
    x = pd.Series(series, dtype=float).dropna().values
    if x.size == 0:
        return np.nan
    return float(np.percentile(x, pctl))

print("\n=== PERCENTILES DE CAUDAL (m³/s) ===")
pairs = [
    ("Obs-Cal", q_obs_calib),
    ("Sim-Cal", q_sim_cal),
    ("Obs-Val", q_obs_valid),
    ("Sim-Val", q_sim_val),
    ("Obs-Global", q_obs_full),
    ("Sim-Global", results["Q_sim_m3s"].values),
]
for label, arr in pairs:
    q95 = _p(arr, 5)   # 5º percentil ~ Q95 (flujo bajo)
    q50 = _p(arr, 50)
    q05 = _p(arr, 95)  # 95º percentil ~ picos
    print(f"{label:11s}  Q95={q95:.4f}  Q50={q50:.4f}  Q05={q05:.4f}")

# ---------------------------------------------------------------------
# 12) Guardado de resultados
# ---------------------------------------------------------------------
out_csv = data_dir / "desk_check_results.csv"
results.to_csv(out_csv)
print(f"\nResultados de simulación guardados en: {out_csv}")

summary = pd.DataFrame({
    "subset": ["Calibration", "Validation", "Global"],
    "NSE": [nse_cal, nse_val, nse_glb],
    "KGE": [kge_cal, kge_val, kge_glb],
    "RMSE": [rmse_cal, rmse_val, rmse_glb],
    "BIAS_%": [bias_cal, bias_val, bias_glb],
    "sumP_mm": [float(np.nansum(df_calib['P_mm'])), float(np.nansum(df_valid['P_mm'])), sum_P],
    "sumPET_mm": [float(np.nansum(df_calib['PET_mm'])), float(np.nansum(df_valid['PET_mm'])), sum_PET],
})
sum_path = data_dir / "desk_check_metrics_summary.csv"
summary.to_csv(sum_path, index=False)
print(f"Resumen de métricas guardado en: {sum_path}")

print("\n=== FIN DESK-CHECK ===")

# ==============================
# 13) GRÁFICAS (añadidas al final)
# ==============================
import matplotlib.pyplot as plt
from tank_model.metrics import plot_error_metrics_heatmap

# --- Preparación de series en m³/s ---
q_sim_full  = results["Q_sim_m3s"].astype(float).values
q_obs_full  = results["Q_obs_m3s"].astype(float).values
q_sim_calib = q_sim_full[:split_idx]
q_sim_valid = q_sim_full[split_idx:]
# (ya tienes q_obs_calib y q_obs_valid del bloque 5)

# -------------------------
# 13a) Serie de tiempo
# -------------------------
fig_ts, ax1 = plt.subplots(figsize=(12, 6))
results["Q_sim_m3s"].plot(ax=ax1, color="red", label="Simulado", linewidth=1.5)
results["Q_obs_m3s"].plot(ax=ax1, color="black", label="Observado", linewidth=1)
ax1.set_ylabel("Caudal (m³/s)")

ax2 = ax1.twinx()
ax2.bar(results.index, results["P_mm"], alpha=0.3, width=1.0, label="Precipitación")
ax2.set_ylabel("Precipitación (mm)")
ax2.invert_yaxis()

ax1.axvline(results.index[split_idx], color="gray", linestyle="--", linewidth=2, alpha=0.8)
ax1.text(results.index[split_idx], ax1.get_ylim()[1]*0.95, 
         f"Inicio Validación\nNSE Cal: {nse_cal:.3f}\nNSE Val: {nse_val:.3f}",
         rotation=0, va="top", ha="left", fontsize=10, 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
ax1.set_title("Calibración y Validación del Modelo - Series de Tiempo")
fig_ts.tight_layout()

ts_path = data_dir / "calibration_validation_timeseries.png"
fig_ts.savefig(ts_path, dpi=300)
print(f"[Plot] Serie de tiempo guardada en: {ts_path}")
plt.show()

# -------------------------
# 13b) Curvas de duración (FDC)
# -------------------------
def _plot_fdc(ax, q_obs, q_sim, title):
    import numpy as _np
    q_sim_sorted = _np.sort(q_sim)[::-1]
    q_obs_sorted = _np.sort(q_obs)[::-1]
    ex = _np.arange(1, len(q_sim_sorted) + 1) / (len(q_sim_sorted) + 1) * 100
    ax.plot(ex, q_sim_sorted, label="Simulado", linewidth=2)
    ax.plot(ex, q_obs_sorted, label="Observado", linewidth=1.5)
    ax.set_xlabel("Probabilidad de excedencia (%)")
    ax.set_ylabel("Caudal (m³/s)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

fig_fdc, (ax_fdc1, ax_fdc2) = plt.subplots(1, 2, figsize=(15, 6))
_plot_fdc(ax_fdc1, q_obs_calib, q_sim_calib, f"FDC - Calibración (NSE={nse_cal:.3f})")
_plot_fdc(ax_fdc2, q_obs_valid, q_sim_valid, f"FDC - Validación (NSE={nse_val:.3f})")
fig_fdc.tight_layout()

fdc_path = data_dir / "flow_duration_curves_comparison.png"
fig_fdc.savefig(fdc_path, dpi=300)
print(f"[Plot] FDC guardada en: {fdc_path}")
plt.show()

# -------------------------
# 13c) Dispersión 1:1
# -------------------------
def _scatter_11(ax, x_obs, y_sim, title):
    import numpy as _np
    ax.scatter(x_obs, y_sim, alpha=0.6, s=20)
    min_q = min(_np.min(x_obs), _np.min(y_sim))
    max_q = max(_np.max(x_obs), _np.max(y_sim))
    ax.plot([min_q, max_q], [min_q, max_q], 'r--', linewidth=2, label='1:1')
    ax.set_xlabel("Caudal Observado (m³/s)")
    ax.set_ylabel("Caudal Simulado (m³/s)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

fig_scatter, (ax_s1, ax_s2) = plt.subplots(1, 2, figsize=(15, 6))
_scatter_11(ax_s1, q_obs_calib, q_sim_calib, f"Calibración (NSE={nse_cal:.3f})")
_scatter_11(ax_s2, q_obs_valid, q_sim_valid, f"Validación (NSE={nse_val:.3f})")
fig_scatter.tight_layout()

scatter_path = data_dir / "scatter_plots_comparison.png"
fig_scatter.savefig(scatter_path, dpi=300)
print(f"[Plot] Scatter 1:1 guardado en: {scatter_path}")
plt.show()

# -------------------------
# 13d) Heatmap de métricas (global)
# -------------------------
print("\n=== MAPA DE CALOR DE MÉTRICAS (GLOBAL) ===")
plot_error_metrics_heatmap(results["Q_obs_m3s"], results["Q_sim_m3s"])
heatmap_path = data_dir / "metrics_heatmap.png"
plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
print(f"[Plot] Heatmap de métricas guardado en: {heatmap_path}")
plt.show()
