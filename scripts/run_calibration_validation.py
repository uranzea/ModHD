from __future__ import annotations

import os
import sys

from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common import (
    DEFAULT_FORCING_PATH,
    INPUT_DIR,
    LOGS_DIR,
    RESULTS_DIR,
    configure_matplotlib_backend,
    ensure_output_dirs,
    infer_dt_hours,
    load_default_forcing,
    save_figure,
)

configure_matplotlib_backend()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tank_model import TankModel, ModelConfig
from tank_model.calibration import random_search
from tank_model.metrics import bias_pct, nse, plot_error_metrics_heatmap, rmse, r_squared
from tank_model.parameters import Parameters


CALIBRATION_FRACTION = 0.7
AREA_KM2 = float(os.environ.get("MODHD_AREA_KM2", "40.0"))
CALIBRATION_LOG_PATH = LOGS_DIR / "calibration_log.csv"
SIMULATION_RESULTS_PATH = RESULTS_DIR / "simulacion_completa.csv"
METRICS_RESULTS_PATH = RESULTS_DIR / "metricas_desempeno.csv"
PARAMETERS_RESULTS_PATH = RESULTS_DIR / "parametros_optimos.csv"
BALANCE_DEBUG_PATH = RESULTS_DIR / "diag_balance_run.csv"


def load_observed_discharge(df: pd.DataFrame, config: ModelConfig) -> np.ndarray:
    if "Qobs_m3s" in df.columns:
        q_obs = df["Qobs_m3s"].astype(float).values
        print(f"Observaciones cargadas: {np.sum(~np.isnan(q_obs))} valores válidos")
        return q_obs

    print("Observaciones no encontradas, generando sintéticas a partir de Q_m3s...")
    sim_tmp = TankModel(params=Parameters(), config=config).run(df)
    noise = 1 + np.random.normal(0, 0.1, size=len(sim_tmp))
    return sim_tmp["Q_m3s"].values * noise


def split_dataset(df: pd.DataFrame, q_obs_full: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, int]:
    split_idx = int(len(df) * CALIBRATION_FRACTION)
    df_calib = df.iloc[:split_idx].copy()
    df_valid = df.iloc[split_idx:].copy()
    q_obs_calib = q_obs_full[:split_idx]
    q_obs_valid = q_obs_full[split_idx:]
    print(
        f"Período calibración: {len(df_calib)} pasos ({df_calib.index[0]} a {df_calib.index[-1]})"
    )
    print(
        f"Período validación: {len(df_valid)} pasos ({df_valid.index[0]} a {df_valid.index[-1]})"
    )
    return df_calib, df_valid, q_obs_calib, q_obs_valid, split_idx


def build_results_frame(df: pd.DataFrame, sim_complete: pd.DataFrame, q_obs_full: np.ndarray, split_idx: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Q_sim_m3s": sim_complete["Q_m3s"].astype(float),
            "Q_sim_mm": sim_complete["Qout_mm"].astype(float),
            "Q_obs_m3s": q_obs_full.astype(float),
            "P_mm": df["P_mm"].astype(float),
            "PET_mm": df["PET_mm"].astype(float),
            "ET_mm": sim_complete["ET_mm"].astype(float),
            "Period": ["Calibration"] * split_idx + ["Validation"] * (len(df) - split_idx),
        },
        index=sim_complete.index,
    )


def compute_metric_summary(results_df: pd.DataFrame, q_obs_calib: np.ndarray, q_obs_valid: np.ndarray, split_idx: int) -> pd.DataFrame:
    q_sim_calib = results_df["Q_sim_m3s"].iloc[:split_idx].values
    q_sim_valid = results_df["Q_sim_m3s"].iloc[split_idx:].values
    q_obs_full = results_df["Q_obs_m3s"].values
    q_sim_full = results_df["Q_sim_m3s"].values
    return pd.DataFrame(
        {
            "NSE": [nse(q_obs_calib, q_sim_calib), nse(q_obs_valid, q_sim_valid), nse(q_obs_full, q_sim_full)],
            "BIAS_pct": [bias_pct(q_obs_calib, q_sim_calib), bias_pct(q_obs_valid, q_sim_valid), bias_pct(q_obs_full, q_sim_full)],
            "RMSE_m3s": [rmse(q_obs_calib, q_sim_calib), rmse(q_obs_valid, q_sim_valid), rmse(q_obs_full, q_sim_full)],
            "R2": [r_squared(q_obs_calib, q_sim_calib), r_squared(q_obs_valid, q_sim_valid), r_squared(q_obs_full, q_sim_full)],
        },
        index=["Calibration", "Validation", "Global"],
    )


def save_outputs(results_df: pd.DataFrame, metrics_summary: pd.DataFrame, best_params: Parameters) -> None:
    results_df.to_csv(SIMULATION_RESULTS_PATH, index_label="date")
    metrics_summary.to_csv(METRICS_RESULTS_PATH, index_label="subset")
    pd.DataFrame([vars(best_params)]).to_csv(PARAMETERS_RESULTS_PATH, index=False)
    print(f"Resultados guardados en: {SIMULATION_RESULTS_PATH}")
    print(f"Métricas guardadas en: {METRICS_RESULTS_PATH}")
    print(f"Parámetros guardados en: {PARAMETERS_RESULTS_PATH}")


def plot_timeseries(results_df: pd.DataFrame, metrics_summary: pd.DataFrame, split_idx: int) -> None:
    fig, ax1 = plt.subplots(figsize=(12, 6))
    results_df["Q_sim_m3s"].plot(ax=ax1, color="red", label="Simulado", linewidth=1.5)
    results_df["Q_obs_m3s"].plot(ax=ax1, color="black", label="Observado", linewidth=1.0)
    ax1.set_ylabel("Caudal (m³/s)")
    ax2 = ax1.twinx()
    ax2.bar(results_df.index, results_df["P_mm"], color="blue", alpha=0.3, width=1.0, label="Precipitación")
    ax2.set_ylabel("Precipitación (mm)")
    ax2.invert_yaxis()
    split_marker = results_df.index[min(split_idx, len(results_df) - 1)]
    ax1.axvline(split_marker, color="gray", linestyle="--", linewidth=2, alpha=0.8)
    ax1.text(
        split_marker,
        ax1.get_ylim()[1] * 0.95,
        f"Inicio Validación\nNSE Cal: {metrics_summary.loc['Calibration', 'NSE']:.3f}\nNSE Val: {metrics_summary.loc['Validation', 'NSE']:.3f}",
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
    )
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.set_title("Calibración y Validación del Modelo - Series de Tiempo")
    fig.tight_layout()
    save_figure(fig, RESULTS_DIR / "calibration_validation_timeseries.png")


def plot_fdc(results_df: pd.DataFrame, split_idx: int, metrics_summary: pd.DataFrame) -> None:
    fig, (ax_cal, ax_val) = plt.subplots(1, 2, figsize=(15, 6))
    q_obs_cal = np.sort(results_df["Q_obs_m3s"].iloc[:split_idx].values)[::-1]
    q_sim_cal = np.sort(results_df["Q_sim_m3s"].iloc[:split_idx].values)[::-1]
    q_obs_val = np.sort(results_df["Q_obs_m3s"].iloc[split_idx:].values)[::-1]
    q_sim_val = np.sort(results_df["Q_sim_m3s"].iloc[split_idx:].values)[::-1]
    ex_cal = np.arange(1, len(q_sim_cal) + 1) / (len(q_sim_cal) + 1) * 100
    ex_val = np.arange(1, len(q_sim_val) + 1) / (len(q_sim_val) + 1) * 100
    ax_cal.plot(ex_cal, q_sim_cal, color="red", label="Simulado", linewidth=2)
    ax_cal.plot(ex_cal, q_obs_cal, color="black", label="Observado", linewidth=1.5)
    ax_cal.set_title(f"Curva de Duración - Calibración\nNSE: {metrics_summary.loc['Calibration', 'NSE']:.4f}")
    ax_val.plot(ex_val, q_sim_val, color="tomato", label="Simulado", linewidth=2)
    ax_val.plot(ex_val, q_obs_val, color="gray", label="Observado", linewidth=1.5)
    ax_val.set_title(f"Curva de Duración - Validación\nNSE: {metrics_summary.loc['Validation', 'NSE']:.4f}")
    for ax in (ax_cal, ax_val):
        ax.set_xlabel("Probabilidad de excedencia (%)")
        ax.set_ylabel("Caudal (m³/s)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, RESULTS_DIR / "flow_duration_curves_comparison.png")


def plot_scatter(results_df: pd.DataFrame, split_idx: int, metrics_summary: pd.DataFrame) -> None:
    fig, (ax_cal, ax_val) = plt.subplots(1, 2, figsize=(15, 6))
    subsets = [
        (ax_cal, results_df["Q_obs_m3s"].iloc[:split_idx].values, results_df["Q_sim_m3s"].iloc[:split_idx].values, "Calibration"),
        (ax_val, results_df["Q_obs_m3s"].iloc[split_idx:].values, results_df["Q_sim_m3s"].iloc[split_idx:].values, "Validation"),
    ]
    for ax, q_obs, q_sim, name in subsets:
        ax.scatter(q_obs, q_sim, alpha=0.6, s=20)
        min_q = min(np.min(q_obs), np.min(q_sim))
        max_q = max(np.max(q_obs), np.max(q_sim))
        ax.plot([min_q, max_q], [min_q, max_q], "r--", linewidth=2, label="1:1")
        ax.set_xlabel("Caudal Observado (m³/s)")
        ax.set_ylabel("Caudal Simulado (m³/s)")
        ax.set_title(f"{name}\nNSE: {metrics_summary.loc[name, 'NSE']:.4f}, R²: {metrics_summary.loc[name, 'R2']:.4f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, RESULTS_DIR / "scatter_plots_comparison.png")


def plot_metrics_heatmap(results_df: pd.DataFrame) -> None:
    print("\n=== GENERANDO MAPA DE CALOR DE MÉTRICAS ===")
    ax = plot_error_metrics_heatmap(results_df["Q_obs_m3s"], results_df["Q_sim_m3s"])
    save_figure(ax.figure, RESULTS_DIR / "metrics_heatmap.png")


def main() -> None:
    ensure_output_dirs()
    df = load_default_forcing(DEFAULT_FORCING_PATH)
    dt_hours = infer_dt_hours(df.index)
    print(f"Datos cargados: {len(df)} registros desde {df.index[0]} hasta {df.index[-1]}")
    print(f"Fuente de forcing: {DEFAULT_FORCING_PATH}")
    print(f"Resolución temporal inferida: {dt_hours:.2f} h")

    calibration_cfg = ModelConfig(
        dt_hours=dt_hours,
        area_km2=AREA_KM2,
        route=True,
        debug_balance=False,
    )
    report_cfg = ModelConfig(
        dt_hours=dt_hours,
        area_km2=AREA_KM2,
        route=True,
        debug_balance=True,
        debug_csv_path=str(BALANCE_DEBUG_PATH),
    )
    q_obs_full = load_observed_discharge(df, report_cfg)
    df_calib, df_valid, q_obs_calib, q_obs_valid, split_idx = split_dataset(df, q_obs_full)

    def make_model(params: Parameters) -> TankModel:
        return TankModel(params=params, config=calibration_cfg)

    print("\n=== INICIANDO CALIBRACIÓN ===")
    best_params, best_score = random_search(
        make_model,
        df_calib,
        q_obs_calib,
        n_iter=50,
        seed=7,
        catchment_name="Datos_prueba_2",
        log_path=str(CALIBRATION_LOG_PATH),
    )
    print(f"Mejor score compuesto (calibración): {best_score:.4f}")
    print(f"Parámetros calibrados: {best_params}")

    print("\n=== SIMULACIÓN CON PARÁMETROS CALIBRADOS ===")
    sim_complete = TankModel(params=best_params, config=report_cfg).run(df)
    print(f"Simulación completa: {len(sim_complete)} registros")

    results_df = build_results_frame(df, sim_complete, q_obs_full, split_idx)
    metrics_summary = compute_metric_summary(results_df, q_obs_calib, q_obs_valid, split_idx)
    print("\n=== RESULTADOS DE VALIDACIÓN ===")
    print(metrics_summary)

    save_outputs(results_df, metrics_summary, best_params)
    plot_timeseries(results_df, metrics_summary, split_idx)
    plot_fdc(results_df, split_idx, metrics_summary)
    plot_scatter(results_df, split_idx, metrics_summary)
    plot_metrics_heatmap(results_df)


if __name__ == "__main__":
    main()
