#!/usr/bin/env python3
"""
Modelo Hidrológico Tipo Tank - Versión Integrada
Incluye: modelo, parámetros, métricas, calibración y validación
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union, Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# ===============================================================================
# CLASES DE CONFIGURACIÓN Y ESTADOS
# ===============================================================================

@dataclass
class Parameters:
    """Parámetros del modelo Tank"""
    # Capacidades/forma
    S0_max: float = 50.0   # mm, capacidad del tanque superficial
    alpha: float = 1.2     # no-dim, no linealidad escorrentía
    beta: float = 1.0      # no-dim, no linealidad percolación

    # Coeficientes de flujo (por dt)
    k_qs: float = 0.15     # escorrentía directa
    k_inf: float = 0.10    # infiltración S0->S1
    k_perc: float = 0.05   # percolación S1->(S2,S3)
    phi: float = 0.6       # fracción hacia flujo rápido (S2)
    k_qf: float = 0.20     # descarga rápida S2
    k_bf: float = 0.02     # baseflow S3

    # Evapotranspiración
    f_et0: float = 0.05    # contribución de S0
    f_et1: float = 0.02    # contribución de S1

    # Enrutamiento Nash
    n_r: int = 2           # número de reservorios en cascada
    k_r: float = 12.0      # escala del Nash (horas si dt_hours es horas)

    def copy(self):
        """Crear una copia del objeto Parameters"""
        return Parameters(
            S0_max=self.S0_max, alpha=self.alpha, beta=self.beta,
            k_qs=self.k_qs, k_inf=self.k_inf, k_perc=self.k_perc,
            phi=self.phi, k_qf=self.k_qf, k_bf=self.k_bf,
            f_et0=self.f_et0, f_et1=self.f_et1,
            n_r=self.n_r, k_r=self.k_r
        )

@dataclass
class States:
    """Estados de los tanques"""
    S0: float = 0.0  # almacenamiento superficial
    S1: float = 0.0  # zona no saturada
    S2: float = 0.0  # flujo rápido
    S3: float = 0.0  # flujo base

@dataclass
class ModelConfig:
    """Configuración del modelo"""
    area_km2: float            # área de la cuenca en km²
    dt_hours: float = 24.0     # paso temporal en horas
    route: bool = True         # aplicar enrutamiento Nash
    debug_balance: bool = False
    debug_csv_path: Optional[str] = None

    def __post_init__(self):
        if self.area_km2 <= 0:
            raise ValueError("El área debe ser mayor que 0 km²")

# ===============================================================================
# FUNCIONES DE MÉTRICAS DE DESEMPEÑO
# ===============================================================================

def nse(obs, sim):
    """Nash-Sutcliffe Efficiency"""
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    
    # Máscara para valores válidos
    valid_mask = ~(np.isnan(obs) | np.isnan(sim))
    if np.sum(valid_mask) < 2:
        return np.nan
    
    obs_valid = obs[valid_mask]
    sim_valid = sim[valid_mask]
    
    obs_mean = np.mean(obs_valid)
    num = np.sum((obs_valid - sim_valid)**2)
    den = np.sum((obs_valid - obs_mean)**2)
    
    return 1 - num/den if den > 0 else np.nan

def kge(obs, sim):
    """Kling-Gupta Efficiency"""
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    
    valid_mask = ~(np.isnan(obs) | np.isnan(sim))
    if np.sum(valid_mask) < 2:
        return np.nan
    
    obs_valid = obs[valid_mask]
    sim_valid = sim[valid_mask]
    
    r = np.corrcoef(obs_valid, sim_valid)[0,1]
    alpha = np.std(sim_valid)/np.std(obs_valid) if np.std(obs_valid) > 0 else np.nan
    beta = np.mean(sim_valid)/np.mean(obs_valid) if np.mean(obs_valid) > 0 else np.nan
    
    if np.isnan(r) or np.isnan(alpha) or np.isnan(beta):
        return np.nan
    
    return 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)

def bias_pct(obs, sim):
    """Percentage Bias"""
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    
    valid_mask = ~(np.isnan(obs) | np.isnan(sim))
    if np.sum(valid_mask) == 0:
        return np.nan
    
    obs_valid = obs[valid_mask]
    sim_valid = sim[valid_mask]
    
    obs_sum = np.sum(obs_valid)
    if obs_sum == 0:
        return np.nan
    
    return 100.0 * (np.sum(sim_valid) - obs_sum) / obs_sum

def rmse(obs, sim):
    """Root Mean Square Error"""
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    
    valid_mask = ~(np.isnan(obs) | np.isnan(sim))
    if np.sum(valid_mask) == 0:
        return np.nan
    
    obs_valid = obs[valid_mask]
    sim_valid = sim[valid_mask]
    
    return np.sqrt(np.mean((obs_valid - sim_valid)**2))

def mae(obs, sim):
    """Mean Absolute Error"""
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    
    valid_mask = ~(np.isnan(obs) | np.isnan(sim))
    if np.sum(valid_mask) == 0:
        return np.nan
    
    return np.mean(np.abs(obs[valid_mask] - sim[valid_mask]))

def r_squared(obs, sim):
    """Coefficient of Determination (R²)"""
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    
    valid_mask = ~(np.isnan(obs) | np.isnan(sim))
    if np.sum(valid_mask) < 2:
        return np.nan
    
    obs_valid = obs[valid_mask]
    sim_valid = sim[valid_mask]
    
    correlation = np.corrcoef(obs_valid, sim_valid)[0,1]
    return correlation**2 if not np.isnan(correlation) else np.nan

# ===============================================================================
# FUNCIONES DE ENRUTAMIENTO
# ===============================================================================

# def nash_cascade(inflow, n, k, dt):
#     """
#     Enrutamiento Nash en cascada
    
#     Parameters:
#     -----------
#     inflow : array-like
#         Serie de entrada (mm/dt)
#     n : int
#         Número de reservorios
#     k : float  
#         Parámetro de escala temporal (horas)
#     dt : float
#         Paso temporal (horas)
    
#     Returns:
#     --------
#     np.ndarray
#         Serie enrutada
#     """
#     if n <= 0 or k <= 0:
#         return np.array(inflow)
    
#     inflow = np.asarray(inflow, dtype=float)
#     outflow = np.zeros_like(inflow)
    
#     # Estados de los reservorios
#     states = np.zeros(n)
    
#     # Coeficiente de descarga
#     alpha = dt / k
    
#     for i, q_in in enumerate(inflow):
#         if np.isnan(q_in):
#             q_in = 0.0
            
#         # Primer reservorio recibe el caudal de entrada
#         states[0] += q_in
        
#         # Propagar a través de la cascada
#         for j in range(n):
#             q_out = alpha * states[j]
#             states[j] -= q_out
            
#             # El caudal de salida va al siguiente reservorio
#             if j < n - 1:
#                 states[j + 1] += q_out
#             else:
#                 # Último reservorio: caudal de salida
#                 outflow[i] = q_out
    
#     return outflow
"""
Implementación corregida del enrutamiento Nash cascade para modelos hidrológicos

Autor: Análisis basado en literatura científica sobre Nash cascade routing
Fecha: Agosto 2025

Referencias principales:
- Nash, J.E. (1957) "The form of instantaneous unit hydrograph"
- Dooge, J.C.J. (1959) "A general theory of the unit hydrograph"
- Szollosi-Nagy (1982) "The discretization of the continuous linear cascade"

PROBLEMAS IDENTIFICADOS EN EL CÓDIGO ORIGINAL:
1. Inestabilidad numérica cuando dt > k
2. Propagación instantánea en lugar de retardo gradual
3. Posibles valores negativos por vaciado excesivo
4. No conserva masa adecuadamente
"""

def nash_cascade(inflow: np.ndarray, n: int, k: float, dt: float) -> np.ndarray:
    """
    Enrutamiento Nash en cascada - Implementación corregida
    
    Implementa correctamente una cascada de n reservorios lineales idénticos
    usando un esquema numérico estable que conserva masa.
    
    Parameters:
    -----------
    inflow : np.ndarray
        Serie de entrada (mm/dt o m³/s)
    n : int
        Número de reservorios en la cascada (típicamente 1-4)
    k : float
        Constante de tiempo de cada reservorio (horas)
        Debe ser >= dt/2 para estabilidad numérica
    dt : float
        Paso temporal (horas)
        
    Returns:
    --------
    np.ndarray
        Serie de salida enrutada
        
    Notes:
    ------
    - Cada reservorio sigue: S = K * Q (reservorio lineal)
    - Ecuación diferencial: dS/dt + S/K = I
    - Solución estable usando esquema implícito
    - Tiempo al pico ≈ (n-1)*k
    - Tiempo medio = n*k
    """
    
    # Validaciones
    if n <= 0:
        raise ValueError("n debe ser >= 1")
    if k <= 0:
        raise ValueError("k debe ser > 0")
    if dt <= 0:
        raise ValueError("dt debe ser > 0")
        
    # Advertencia sobre estabilidad numérica
    ratio = dt / k
    if ratio > 2.0:
        print(f"ADVERTENCIA: dt/k = {ratio:.2f} > 2.0. Esto puede causar inestabilidad.")
        print(f"Recomendado: dt <= {k/2:.1f}h para estabilidad óptima")
    
    inflow = np.asarray(inflow, dtype=float)
    if len(inflow) == 0:
        return np.array([])
        
    n_steps = len(inflow)
    outflow = np.zeros(n_steps)
    
    # Estados: almacenamiento normalizado (S_i/K = Q_i)
    # Esto simplifica cálculos porque la salida = almacenamiento normalizado
    state = np.zeros(n)
    
    # Coeficientes para esquema implícito estable
    # Solución de dS/dt + S/K = I con esquema backward Euler
    alpha = dt / k
    recession_factor = 1.0 / (1.0 + alpha)
    input_factor = alpha / (1.0 + alpha)
    
    # Procesar cada paso temporal
    for i in range(n_steps):
        q_in = inflow[i] if not np.isnan(inflow[i]) else 0.0
        
        # Propagar a través de la cascada de reservorios
        current_input = q_in
        
        for j in range(n):
            # Actualizar estado del reservorio j usando esquema implícito
            # S_new = (S_old + dt*I) / (1 + dt/K)
            state[j] = recession_factor * (state[j] + input_factor * current_input / recession_factor)
            
            # La salida del reservorio j es su estado normalizado
            current_output = state[j]
            
            # Esta salida se convierte en entrada al siguiente reservorio
            current_input = current_output
            
        # La salida del último reservorio es la salida total
        outflow[i] = current_output
    
    return outflow

def validar_nash_routing(inflow: np.ndarray, outflow: np.ndarray, 
                        n: int, k: float, dt: float) -> dict:
    """
    Validar la implementación del enrutamiento Nash
    
    Returns:
    --------
    dict: Métricas de validación
    """
    # Conservación de masa
    mass_input = np.sum(inflow)
    mass_output = np.sum(outflow)
    mass_error = abs(mass_input - mass_output) / mass_input if mass_input > 0 else 0
    
    # Tiempo al pico
    if np.max(outflow) > 0:
        peak_time = np.argmax(outflow) * dt
        theoretical_peak = (n - 1) * k
    else:
        peak_time = np.nan
        theoretical_peak = (n - 1) * k
    
    # Momentos
    t = np.arange(len(outflow)) * dt
    total_output = np.sum(outflow)
    
    if total_output > 0:
        # Primer momento (tiempo medio)
        mean_time = np.sum(t * outflow) / total_output
        theoretical_mean = n * k
        
        # Verificar valores negativos
        has_negative = np.any(outflow < 0)
    else:
        mean_time = np.nan
        theoretical_mean = n * k
        has_negative = False
    
    return {
        'mass_conservation_error': mass_error,
        'peak_time_hours': peak_time,
        'theoretical_peak_time': theoretical_peak,
        'mean_time_hours': mean_time,
        'theoretical_mean_time': theoretical_mean,
        'has_negative_values': has_negative,
        'max_output': np.max(outflow),
        'stability_ratio_dt_k': dt / k
    }

def recomendar_parametros_nash(dt_hours: float) -> dict:
    """
    Recomendar parámetros Nash apropiados según el paso temporal
    
    Parameters:
    -----------
    dt_hours : float
        Paso temporal en horas
        
    Returns:
    --------
    dict: Recomendaciones de parámetros
    """
    # Para estabilidad: k >= 2*dt (preferiblemente k >= 5*dt)
    k_min_estable = 2 * dt_hours
    k_recomendado = 5 * dt_hours
    
    # Rangos típicos de n
    n_min = 1
    n_max = 4
    n_recomendado = 2  # Balance entre suavizado y simplicidad
    
    return {
        'dt_hours': dt_hours,
        'k_min_for_stability': k_min_estable,
        'k_recommended': k_recomendado,
        'n_range': (n_min, n_max),
        'n_recommended': n_recomendado,
        'notes': [
            f"Para dt={dt_hours}h, usar k >= {k_min_estable}h (mínimo para estabilidad)",
            f"Recomendado: k >= {k_recomendado}h para mayor estabilidad",
            f"n=1: Sin retardo, n=2-3: Comportamiento típico, n=4+: Muy suavizado",
            "Calibrar k para ajustar tiempo de respuesta de la cuenca"
        ]
    }

# Ejemplo de uso y prueba
if __name__ == "__main__":
    # Crear serie de prueba
    np.random.seed(42)
    test_series = np.random.exponential(2.0, 50)
    
    # Parámetros de prueba
    n_test = 3
    k_test = 24.0  # Cambiar de 12 a 24 para mayor estabilidad
    dt_test = 24.0
    
    print("="*60)
    print("PRUEBA DE ENRUTAMIENTO NASH CORREGIDO")
    print("="*60)
    
    # Aplicar enrutamiento
    resultado = nash_cascade(test_series, n_test, k_test, dt_test)
    
    # Validar
    validacion = validar_nash_routing(test_series, resultado, n_test, k_test, dt_test)
    
    print(f"Parámetros: n={n_test}, k={k_test}h, dt={dt_test}h")
    print(f"Error conservación masa: {validacion['mass_conservation_error']:.6f}")
    print(f"Tiempo al pico: {validacion['peak_time_hours']:.1f}h (teórico: {validacion['theoretical_peak_time']:.1f}h)")
    print(f"Tiempo medio: {validacion['mean_time_hours']:.1f}h (teórico: {validacion['theoretical_mean_time']:.1f}h)")
    print(f"¿Valores negativos?: {validacion['has_negative_values']}")
    print(f"Ratio estabilidad dt/k: {validacion['stability_ratio_dt_k']:.2f}")
    
    # Mostrar recomendaciones
    print(f"\n{'='*60}")
    print("RECOMENDACIONES DE PARÁMETROS")
    print("="*60)
    
    recomendaciones = recomendar_parametros_nash(dt_test)
    for nota in recomendaciones['notes']:
        print(f"• {nota}")

# ===============================================================================
# MODELO HIDROLÓGICO TANK
# ===============================================================================

class TankModel:
    """Modelo hidrológico tipo Tank con 4 tanques"""
    
    def __init__(self, params: Parameters, config: ModelConfig, init_states: States = None):
        self.p = params
        self.c = config
        self.s = init_states if init_states is not None else States()
        
    def step(self, P, PET):
        """Un paso temporal del modelo"""
        p, s = self.p, self.s
        
        # Validar entradas
        P = max(0.0, float(P) if not np.isnan(P) else 0.0)
        PET = max(0.0, float(PET) if not np.isnan(PET) else 0.0)
        
        # === TANQUE S0 (SUPERFICIE) ===
        
        # Evapotranspiración potencial
        ET_pot = PET
        ET_cap = p.f_et0 * s.S0 + p.f_et1 * s.S1
        ET = min(ET_pot, ET_cap)
        
        # Balance en S0
        S0 = s.S0 + P - ET
        S0 = max(0.0, S0)  # No permitir negativos
        
        # Flujos desde S0
        Qs = p.k_qs * (S0 ** p.alpha) if S0 > 0 else 0.0
        I = p.k_inf * S0 if S0 > 0 else 0.0
        
        # Exceso sobre capacidad máxima
        if S0 > p.S0_max:
            excess = S0 - p.S0_max
            Qs += excess
            S0 = p.S0_max
        
        # Verificar disponibilidad de agua
        total_out = Qs + I
        if total_out > S0:
            if total_out > 0:
                factor = S0 / total_out
                Qs *= factor
                I *= factor
            S0 = 0.0
        else:
            S0 -= total_out
        
        S0 = max(0.0, S0)
        
        # === TANQUE S1 (ZONA NO SATURADA) ===
        
        S1 = s.S1 + I
        Perc = p.k_perc * (S1 ** p.beta) if S1 > 0 else 0.0
        
        if Perc > S1:
            Perc = S1
            S1 = 0.0
        else:
            S1 -= Perc
        
        S1 = max(0.0, S1)
        
        # Partición de percolación
        to_S2 = p.phi * Perc
        to_S3 = (1 - p.phi) * Perc
        
        # === TANQUE S2 (FLUJO RÁPIDO) ===
        
        S2 = s.S2 + to_S2
        Qf = p.k_qf * S2 if S2 > 0 else 0.0
        
        if Qf > S2:
            Qf = S2
            S2 = 0.0
        else:
            S2 -= Qf
        
        S2 = max(0.0, S2)
        
        # === TANQUE S3 (FLUJO BASE) ===
        
        S3 = s.S3 + to_S3
        Qb = p.k_bf * S3 if S3 > 0 else 0.0
        
        if Qb > S3:
            Qb = S3
            S3 = 0.0
        else:
            S3 -= Qb
        
        S3 = max(0.0, S3)
        
        # === ACTUALIZAR ESTADOS ===
        
        self.s = States(S0=S0, S1=S1, S2=S2, S3=S3)
        
        # === CAUDAL TOTAL ===
        
        Qin = Qs + Qf + Qb
        
        return {
            "ET_mm": ET, "I_mm": I, "Perc_mm": Perc,
            "Qs_mm": Qs, "Qf_mm": Qf, "Qb_mm": Qb,
            "Qin": Qin, "S0": S0, "S1": S1, "S2": S2, "S3": S3
        }
    
    def run(self, df):
        """
        Ejecutar el modelo para toda la serie temporal
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame con columnas P_mm y PET_mm
            
        Returns:
        --------
        pd.DataFrame
            Resultados de la simulación
        """
        # Verificar columnas requeridas
        required_cols = ["P_mm", "PET_mm"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Faltan columnas: {missing_cols}")
        
        # Limpiar datos de entrada
        df_clean = df.copy()
        for col in required_cols:
            # Reemplazar valores inválidos
            invalid_mask = df_clean[col].isna() | (df_clean[col] < 0)
            if invalid_mask.sum() > 0:
                df_clean[col] = df_clean[col].mask(invalid_mask).interpolate(
                    method='linear', limit_direction='both'
                ).fillna(0.0)
        
        # Reinicializar estados
        self.s = States()
        
        # Ejecutar simulación paso a paso
        results = []
        for i in range(len(df_clean)):
            step_result = self.step(
                df_clean.iloc[i]["P_mm"], 
                df_clean.iloc[i]["PET_mm"]
            )
            results.append(step_result)
        
        # Convertir a DataFrame
        out_df = pd.DataFrame(results, index=df_clean.index)
        
        # === ENRUTAMIENTO NASH ===
        
        if self.c.route and self.p.n_r > 0 and self.p.k_r > 0:
            routed = nash_cascade(
                out_df["Qin"].values, 
                n=int(self.p.n_r), 
                k=self.p.k_r, 
                dt=self.c.dt_hours
            )
            out_df["Qout_mm"] = routed
            out_df["Qraw_mm"] = out_df["Qin"].copy()
        else:
            out_df["Qout_mm"] = out_df["Qin"].copy()
            out_df["Qraw_mm"] = out_df["Qin"].copy()
        
        # Conversión a m³/s
        seconds_per_dt = self.c.dt_hours * 3600.0
        mm_to_m3s_factor = self.c.area_km2 * 1e6 * 1e-3 / seconds_per_dt
        out_df["Q_m3s"] = out_df["Qout_mm"] * mm_to_m3s_factor
        
        return out_df

# ===============================================================================
# FUNCIONES DE CALIBRACIÓN
# ===============================================================================

# def generate_random_parameters(bounds: Dict, rng: np.random.Generator) -> Parameters:
#     """Generar parámetros aleatorios dentro de los límites especificados"""
#     params = Parameters()
    
#     for name, (lo, hi) in bounds.items():
#         if name == "n_r":
#             # Parámetro entero
#             setattr(params, name, int(rng.integers(lo, hi + 1)))
#         else:
#             # Parámetro continuo
#             setattr(params, name, rng.uniform(lo, hi))
    
#     return params

def generate_random_parameters(bounds: Dict, rng: np.random.Generator) -> Parameters:
    params = Parameters()
    
    for name, (lo, hi) in bounds.items():
        if name == "k_r":
            # FORZAR k_r mínimo para estabilidad
            lo_corrected = max(lo, 60.0)  # Mínimo absoluto 60h
            hi_corrected = max(hi, 120.0) # Mínimo rango hasta 120h
            setattr(params, name, rng.uniform(lo_corrected, hi_corrected))
        elif name == "n_r":
            setattr(params, name, int(rng.integers(lo, hi + 1)))
        else:
            setattr(params, name, rng.uniform(lo, hi))
    
    return params

def compute_multi_objective_score(obs, sim, weights=None):
    """
    Función objetivo multi-criterio
    
    Parameters:
    -----------
    obs, sim : array-like
        Series observada y simulada
    weights : dict, optional
        Pesos para cada métrica
    
    Returns:
    --------
    float
        Puntuación combinada (mayor es mejor)
    """
    if weights is None:
        weights = {
            'nse_linear': 0.2,
            'nse_log': 0.15,
            'peak_bias': 0.45,
            'kge': 0.2
        }
    
    # NSE lineal
    nse_lin = nse(obs, sim)
    if np.isnan(nse_lin):
        return -np.inf
    
    # NSE logarítmico (para flujos bajos)
    obs_log = np.log1p(np.maximum(0, obs))
    sim_log = np.log1p(np.maximum(0, sim))
    nse_log = nse(obs_log, sim_log)
    if np.isnan(nse_log):
        nse_log = -1.0
    
    # Sesgo en picos (percentil 95)
    threshold = np.nanpercentile(obs, 95)
    peak_mask = obs >= threshold
    if np.sum(peak_mask) > 0:
        peak_bias = abs(bias_pct(obs[peak_mask], sim[peak_mask])) / 100.0
        peak_score = np.exp(-peak_bias)  # Convertir a score (1 = perfecto, 0 = muy mal)
    else:
        peak_score = 1.0
    
    # KGE
    kge_score = kge(obs, sim)
    if np.isnan(kge_score):
        kge_score = -1.0
    
    # Puntuación combinada
    score = (
        weights['nse_linear'] * nse_lin +
        weights['nse_log'] * nse_log +
        weights['peak_bias'] * peak_score +
        weights['kge'] * kge_score
    )
    
    return score

def random_search_calibration(
    df_calib: pd.DataFrame,
    q_obs_calib: np.ndarray,
    config: ModelConfig,
    bounds: Dict,
    n_iterations: int = 100,
    seed: int = 42,
    verbose: bool = True
) -> Tuple[Parameters, List[float], List[Parameters]]:
    """
    Calibración por búsqueda aleatoria
    
    Returns:
    --------
    best_params : Parameters
        Mejores parámetros encontrados
    scores : List[float]
        Historia de puntuaciones
    param_history : List[Parameters]
        Historia de parámetros probados
    """
    rng = np.random.default_rng(seed)
    
    best_score = -np.inf
    best_params = None
    scores = []
    param_history = []
    
    for i in range(n_iterations):
        # Generar parámetros aleatorios
        params = generate_random_parameters(bounds, rng)
        param_history.append(params.copy())
        
        try:
            # Simular
            model = TankModel(params=params, config=config)
            sim_df = model.run(df_calib)
            q_sim = sim_df["Qout_mm"].values
            
            # Evaluar
            score = compute_multi_objective_score(q_obs_calib, q_sim)
            scores.append(score)
            
            # Actualizar mejor resultado
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_params = params.copy()
                
            if verbose and (i + 1) % 10 == 0:
                print(f"Iteración {i+1}/{n_iterations}: Score = {score:.4f}, Mejor = {best_score:.4f}")
                
        except Exception as e:
            scores.append(-np.inf)
            if verbose:
                print(f"Iteración {i+1}: Error en simulación - {e}")
    
    return best_params, scores, param_history

def scipy_optimization(
    df_calib: pd.DataFrame,
    q_obs_calib: np.ndarray,
    config: ModelConfig,
    initial_params: Parameters,
    bounds: Dict,
    method: str = "L-BFGS-B",
    verbose: bool = True
) -> Tuple[Parameters, List[float]]:
    """
    Optimización determinística con SciPy
    
    Returns:
    --------
    best_params : Parameters
        Parámetros optimizados
    error_history : List[float]
        Historia de errores durante optimización
    """
    
    # Parámetros a optimizar (excluyendo n_r que es entero)
    param_names = [
        "S0_max", "alpha", "k_qs", "k_inf", "k_perc",
        "phi", "k_qf", "k_bf", "f_et0", "f_et1", "k_r"
    ]
    
    # Valores iniciales
    x0 = np.array([getattr(initial_params, name) for name in param_names])
    
    # Límites para optimización
    bounds_opt = [bounds[name] for name in param_names]
    
    # Historia de errores
    error_history = []
    
    def objective_function(x):
        """Función objetivo (minimizar 1 - NSE)"""
        try:
            # Crear parámetros
            params = initial_params.copy()
            for name, val in zip(param_names, x):
                setattr(params, name, val)
            
            # Simular
            model = TankModel(params=params, config=config)
            sim_df = model.run(df_calib)
            q_sim = sim_df["Qout_mm"].values
            
            # Error = 1 - NSE (minimizar)
            nse_score = nse(q_obs_calib, q_sim)
            error = 1.0 - nse_score if not np.isnan(nse_score) else 10.0
            
            return error
            
        except Exception:
            return 10.0  # Penalización alta por error
    
    def callback_function(xk):
        """Callback para guardar progreso"""
        error = objective_function(xk)
        error_history.append(error)
        if verbose and len(error_history) % 5 == 0:
            print(f"Optimización - Iteración {len(error_history)}: Error = {error:.6f}")
    
    # Evaluación inicial
    initial_error = objective_function(x0)
    error_history.append(initial_error)
    
    if verbose:
        print(f"Error inicial: {initial_error:.6f}")
    
    # Optimización
    try:
        result = minimize(
            objective_function,
            x0,
            method=method,
            bounds=bounds_opt,
            callback=callback_function,
            options={'maxiter': 60}
        )
        
        # Crear parámetros optimizados
        best_params = initial_params.copy()
        for name, val in zip(param_names, result.x):
            setattr(best_params, name, val)
        
        if verbose:
            print(f"Optimización completada. Error final: {result.fun:.6f}")
            print(f"Éxito: {result.success}")
        
    except Exception as e:
        if verbose:
            print(f"Error en optimización: {e}")
        best_params = initial_params.copy()
    
    return best_params, error_history

# ===============================================================================
# FUNCIONES DE VISUALIZACIÓN
# ===============================================================================

def plot_error_metrics_heatmap(obs, sim, ax=None, cmap="viridis"):
    """Gráfico de heatmap con métricas de error"""
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)

    metric_funcs = {
        "NSE": nse,
        "KGE": kge, 
        "Bias%": bias_pct,
        "RMSE": rmse,
        "MAE": mae,
        "R²": r_squared,
    }
    
    labels = list(metric_funcs.keys())
    values = [metric_funcs[name](obs, sim) for name in labels]
    data = np.array(values, dtype=float).reshape(1, -1)

    if ax is None:
        fig_width = 1.2 * data.shape[1]
        _, ax = plt.subplots(figsize=(fig_width, 2))

    im = ax.imshow(data, aspect="auto", cmap=cmap)
    
    # Color de texto basado en valor
    vmin, vmax = np.nanmin(data), np.nanmax(data)
    mid = (vmax + vmin) / 2 if not np.isnan(vmax + vmin) else 0

    for j, val in enumerate(data[0]):
        if np.isnan(val):
            text_val = "NaN"
            color = "red"
        else:
            text_val = f"{val:.3f}"
            color = "white" if val < mid else "black"
        ax.text(j, 0, text_val, ha="center", va="center", color=color)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks([])
    ax.set_title("Métricas de Error")
    plt.colorbar(im, ax=ax, label="Valor")
    
    return ax

def create_comprehensive_plots(results_df, split_idx, error_history, output_dir):
    """Crear todas las gráficas de análisis"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Serie temporal
    fig_ts, ax1 = plt.subplots(figsize=(14, 8))
    
    # Caudales
    results_df["Q_sim_m3s"].plot(ax=ax1, color="red", linewidth=1.5, label="Simulado")
    results_df["Q_obs_m3s"].plot(ax=ax1, color="black", linewidth=1.5, label="Observado")
    ax1.set_ylabel("Caudal (m³/s)", fontsize=12)
    ax1.legend(loc="upper left", fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Precipitación
    ax2 = ax1.twinx()
    ax2.bar(results_df.index, results_df["P_mm"], 
            color="blue", alpha=0.4, width=0.8, label="Precipitación")
    ax2.set_ylabel("Precipitación (mm)", fontsize=12)
    ax2.legend(loc="upper right", fontsize=11)
    ax2.invert_yaxis()  # Precipitación hacia abajo
    
    # Línea de división calibración/validación
    if split_idx < len(results_df):
        split_date = results_df.index[split_idx]
        ax1.axvline(split_date, color="gray", linestyle="--", linewidth=2, alpha=0.7)
        ax1.text(split_date, ax1.get_ylim()[1]*0.9, "Inicio Validación", 
                rotation=90, va="top", ha="right", fontsize=10)
    
    ax1.set_title("Serie Temporal - Calibración y Validación", fontsize=14, fontweight='bold')
    fig_ts.tight_layout()
    fig_ts.savefig(output_dir / "01_serie_temporal.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Curvas de duración de caudales
    q_sim_calib = results_df["Q_sim_m3s"].iloc[:split_idx].values
    q_obs_calib = results_df["Q_obs_m3s"].iloc[:split_idx].values
    q_sim_valid = results_df["Q_sim_m3s"].iloc[split_idx:].values
    q_obs_valid = results_df["Q_obs_m3s"].iloc[split_idx:].values
    
    # Ordenar para curvas de duración
    q_sim_calib_sorted = np.sort(q_sim_calib)[::-1]
    q_obs_calib_sorted = np.sort(q_obs_calib)[::-1]
    ex_calib = np.arange(1, len(q_sim_calib_sorted) + 1) / (len(q_sim_calib_sorted) + 1)
    
    q_sim_valid_sorted = np.sort(q_sim_valid)[::-1]
    q_obs_valid_sorted = np.sort(q_obs_valid)[::-1]
    ex_valid = np.arange(1, len(q_sim_valid_sorted) + 1) / (len(q_sim_valid_sorted) + 1)
    
    fig_fdc, ax_fdc = plt.subplots(figsize=(10, 6))
    ax_fdc.semilogy(ex_calib, q_sim_calib_sorted, label="Sim Calibración", color="red", linewidth=2)
    ax_fdc.semilogy(ex_calib, q_obs_calib_sorted, label="Obs Calibración", color="black", linewidth=2)
    ax_fdc.semilogy(ex_valid, q_sim_valid_sorted, label="Sim Validación", color="tomato", linewidth=2, linestyle='--')
    ax_fdc.semilogy(ex_valid, q_obs_valid_sorted, label="Obs Validación", color="gray", linewidth=2, linestyle='--')
    
    ax_fdc.set_xlabel("Probabilidad de excedencia", fontsize=12)
    ax_fdc.set_ylabel("Caudal (m³/s)", fontsize=12)
    ax_fdc.set_title("Curvas de Duración de Caudales", fontsize=14, fontweight='bold')
    ax_fdc.legend(fontsize=11)
    ax_fdc.grid(True, alpha=0.3)
    fig_fdc.tight_layout()
    fig_fdc.savefig(output_dir / "02_curvas_duracion.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Evolución del error de calibración
    if len(error_history) > 0:
        fig_err, ax_err = plt.subplots(figsize=(10, 6))
        iterations = range(1, len(error_history) + 1)
        ax_err.plot(iterations, error_history, marker="o", linewidth=2, markersize=4)
        ax_err.set_xlabel("Iteración", fontsize=12)
        ax_err.set_ylabel("Error (1 - NSE)", fontsize=12)
        ax_err.set_title("Evolución del Error durante Optimización", fontsize=14, fontweight='bold')
        ax_err.grid(True, alpha=0.3)
        
        # Marcar mínimo
        min_idx = np.argmin(error_history)
        min_error = error_history[min_idx]
        ax_err.annotate(f'Mínimo: {min_error:.4f}', 
                       xy=(min_idx + 1, min_error), xytext=(10, 10),
                       textcoords='offset points', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        fig_err.tight_layout()
        fig_err.savefig(output_dir / "03_evolucion_error.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Gráfico de dispersión obs vs sim
    fig_scatter, (ax_cal, ax_val) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Calibración
    max_val_cal = max(np.max(q_obs_calib), np.max(q_sim_calib))
    ax_cal.scatter(q_obs_calib, q_sim_calib, alpha=0.6, s=20)
    ax_cal.plot([0, max_val_cal], [0, max_val_cal], 'r--', linewidth=2, label='1:1')
    ax_cal.set_xlabel("Caudal Observado (m³/s)", fontsize=11)
    ax_cal.set_ylabel("Caudal Simulado (m³/s)", fontsize=11)
    ax_cal.set_title("Calibración", fontsize=12, fontweight='bold')
    ax_cal.legend()
    ax_cal.grid(True, alpha=0.3)
    
    # Validación
    max_val_val = max(np.max(q_obs_valid), np.max(q_sim_valid))
    ax_val.scatter(q_obs_valid, q_sim_valid, alpha=0.6, s=20, color='orange')
    ax_val.plot([0, max_val_val], [0, max_val_val], 'r--', linewidth=2, label='1:1')
    ax_val.set_xlabel("Caudal Observado (m³/s)", fontsize=11)
    ax_val.set_ylabel("Caudal Simulado (m³/s)", fontsize=11)
    ax_val.set_title("Validación", fontsize=12, fontweight='bold')
    ax_val.legend()
    ax_val.grid(True, alpha=0.3)
    
    fig_scatter.tight_layout()
    fig_scatter.savefig(output_dir / "04_dispersion.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Heatmap de métricas
    fig_heatmap, (ax_heat_cal, ax_heat_val) = plt.subplots(2, 1, figsize=(10, 6))
    
    plot_error_metrics_heatmap(q_obs_calib, q_sim_calib, ax=ax_heat_cal)
    ax_heat_cal.set_title("Métricas - Calibración", fontsize=12, fontweight='bold')
    
    plot_error_metrics_heatmap(q_obs_valid, q_sim_valid, ax=ax_heat_val)
    ax_heat_val.set_title("Métricas - Validación", fontsize=12, fontweight='bold')
    
    fig_heatmap.tight_layout()
    fig_heatmap.savefig(output_dir / "05_metricas_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gráficas guardadas en: {output_dir}")

def print_calibration_summary(best_params: Parameters, scores: List[float], param_history: List[Parameters]):
    """Imprimir resumen de calibración"""
    print("\n" + "="*60)
    print("RESUMEN DE CALIBRACIÓN")
    print("="*60)
    
    print(f"Número de iteraciones: {len(scores)}")
    print(f"Mejor puntuación: {max(scores):.6f}")
    print(f"Puntuación promedio: {np.mean(scores):.6f}")
    print(f"Desviación estándar: {np.std(scores):.6f}")
    
    print(f"\nPARÁMETROS ÓPTIMOS:")
    print(f"{'Parámetro':<12} {'Valor':<12} {'Unidad'}")
    print("-" * 35)
    print(f"{'S0_max':<12} {best_params.S0_max:<12.3f} mm")
    print(f"{'alpha':<12} {best_params.alpha:<12.3f} -")
    print(f"{'k_qs':<12} {best_params.k_qs:<12.3f} /dt")
    print(f"{'k_inf':<12} {best_params.k_inf:<12.3f} /dt")
    print(f"{'k_perc':<12} {best_params.k_perc:<12.3f} /dt")
    print(f"{'phi':<12} {best_params.phi:<12.3f} -")
    print(f"{'k_qf':<12} {best_params.k_qf:<12.3f} /dt")
    print(f"{'k_bf':<12} {best_params.k_bf:<12.3f} /dt")
    print(f"{'f_et0':<12} {best_params.f_et0:<12.3f} /dt")
    print(f"{'f_et1':<12} {best_params.f_et1:<12.3f} /dt")
    print(f"{'n_r':<12} {best_params.n_r:<12d} -")
    print(f"{'k_r':<12} {best_params.k_r:<12.3f} h")

def print_performance_metrics(q_obs_calib, q_sim_calib, q_obs_valid, q_sim_valid, q_obs_full, q_sim_full):
    """Imprimir métricas de desempeño"""
    print("\n" + "="*60)
    print("MÉTRICAS DE DESEMPEÑO")
    print("="*60)
    
    # Calcular métricas
    metrics_calib = {
        'NSE': nse(q_obs_calib, q_sim_calib),
        'KGE': kge(q_obs_calib, q_sim_calib),
        'Bias%': bias_pct(q_obs_calib, q_sim_calib),
        'RMSE': rmse(q_obs_calib, q_sim_calib),
        'R²': r_squared(q_obs_calib, q_sim_calib)
    }
    
    metrics_valid = {
        'NSE': nse(q_obs_valid, q_sim_valid),
        'KGE': kge(q_obs_valid, q_sim_valid),
        'Bias%': bias_pct(q_obs_valid, q_sim_valid),
        'RMSE': rmse(q_obs_valid, q_sim_valid),
        'R²': r_squared(q_obs_valid, q_sim_valid)
    }
    
    metrics_global = {
        'NSE': nse(q_obs_full, q_sim_full),
        'KGE': kge(q_obs_full, q_sim_full),
        'Bias%': bias_pct(q_obs_full, q_sim_full),
        'RMSE': rmse(q_obs_full, q_sim_full),
        'R²': r_squared(q_obs_full, q_sim_full)
    }
    
    # Imprimir tabla
    print(f"{'Métrica':<8} {'Calibración':<12} {'Validación':<12} {'Global':<12}")
    print("-" * 50)
    for metric in ['NSE', 'KGE', 'Bias%', 'RMSE', 'R²']:
        cal_val = metrics_calib[metric]
        val_val = metrics_valid[metric]
        glo_val = metrics_global[metric]
        
        cal_str = f"{cal_val:.3f}" if not np.isnan(cal_val) else "NaN"
        val_str = f"{val_val:.3f}" if not np.isnan(val_val) else "NaN"
        glo_str = f"{glo_val:.3f}" if not np.isnan(glo_val) else "NaN"
        
        print(f"{metric:<8} {cal_str:<12} {val_str:<12} {glo_str:<12}")

# ===============================================================================
# FUNCIÓN PRINCIPAL DE CALIBRACIÓN Y VALIDACIÓN
# ===============================================================================

def main_calibration_validation():
    """Función principal que ejecuta todo el proceso de calibración y validación"""
    
    print("MODELO HIDROLÓGICO TIPO TANK - CALIBRACIÓN Y VALIDACIÓN")
    print("="*60)
    
    # ===============================================================================
    # 1. CONFIGURACIÓN Y LECTURA DE DATOS
    # ===============================================================================
    
    # Directorio de trabajo
    try:
        data_dir = Path("D:\OneDrive - Grupo EPM\Documentos\Python Scripts\Codigos_GITHUB\ModHD\data")
        data_dir.mkdir(exist_ok=True)
    except:
        data_dir = Path(".")
    
    # Configuración del modelo
    config = ModelConfig(
        area_km2=40,  # Área de la cuenca en km²
        dt_hours=24.0,   # Paso temporal diario
        route=True       # Aplicar enrutamiento Nash
    )
    
    # Lectura o generación de datos de forzantes
    forcing_file = data_dir / "Datos_prueba_2.csv"
    
    try:
        print(f"Leyendo datos de forzantes: {forcing_file}")
        df = pd.read_csv(forcing_file, parse_dates=["date"], index_col="date")
        print(f"Datos cargados: {len(df)} registros desde {df.index[0]} hasta {df.index[-1]}")
    except FileNotFoundError:
        print("Archivo de forzantes no encontrado. Generando datos sintéticos...")
        # Generar serie sintética
        idx = pd.date_range("2020-01-01", periods=730, freq="D")  # 2 años
        rng = np.random.default_rng(123)
        P = np.maximum(0, rng.gamma(2.0, 5.0, size=len(idx)) - 3)
        PET = 3.0 + 1.5 * np.sin(2 * np.pi * np.arange(len(idx)) / 365.25)  # Ciclo estacional
        df = pd.DataFrame({"P_mm": P, "PET_mm": PET}, index=idx)
        print(f"Datos sintéticos generados: {len(df)} registros")
    
    # Lectura o generación de observaciones de caudal
    try:
        df_q = pd.read_csv(forcing_file, parse_dates=["date"], index_col="date")
        if "Qobs_m3s" in df_q.columns:
            q_obs_full = df_q.reindex(df.index)["Qobs_m3s"].astype(float).values
            print("Caudales observados cargados desde archivo")
        else:
            raise KeyError("Columna Qobs_m3s no encontrada")
    except (FileNotFoundError, KeyError):
        print("Generando caudales observados sintéticos...")
        # Generar con parámetros "verdaderos"
        true_params = Parameters(
            S0_max=45.0, alpha=1.3, k_qs=0.18, k_inf=0.25, k_perc=0.06,
            phi=0.65, k_qf=0.25, k_bf=0.08, f_et0=0.12, f_et1=0.05,
            n_r=2, k_r=10.0
        )
        temp_model = TankModel(params=true_params, config=config)
        temp_sim = temp_model.run(df)
        # Añadir ruido realista
        rng_noise = np.random.default_rng(456)
        noise_factor = 0.15
        q_obs_full = temp_sim["Qout_mm"].values * (1 + rng_noise.normal(0, noise_factor, size=len(temp_sim)))
        q_obs_full = np.maximum(0, q_obs_full)  # No caudales negativos
        print("Caudales sintéticos generados con ruido")
    
    # ===============================================================================
    # 2. DIVISIÓN CALIBRACIÓN/VALIDACIÓN
    # ===============================================================================
    
    calib_fraction = 0.7
    split_idx = int(len(df) * calib_fraction)
    
    df_calib = df.iloc[:split_idx].copy()
    df_valid = df.iloc[split_idx:].copy()
    q_obs_calib = q_obs_full[:split_idx]
    q_obs_valid = q_obs_full[split_idx:]
    
    print(f"\nDivisión de datos:")
    print(f"Calibración: {len(df_calib)} registros ({df_calib.index[0]} a {df_calib.index[-1]})")
    print(f"Validación:  {len(df_valid)} registros ({df_valid.index[0]} a {df_valid.index[-1]})")
    
    # ===============================================================================
    # 3. DEFINICIÓN DE LÍMITES DE PARÁMETROS
    # ===============================================================================

    bounds = {
    "S0_max": (40, 150),        # AUMENTADO (era 20-100) Capacidad superficial
    "alpha": (1.0, 1.8),        # REDUCIDO (era 1.0-2.5) No linealidad escorrentía
    "k_qs": (0.15, 0.8),        # AUMENTADO (era 0.05-0.4) Coeficiente escorrentía
    "k_inf": (0.03, 0.3),       # REDUCIDO (era 0.05-0.5) Infiltración
    "k_perc": (0.01, 0.15),     # Mantener Percolación
    "phi": (0.3, 0.8),          # Mantener Partición S1 -> S2/S3
    "k_qf": (0.15, 0.7),        # AUMENTADO (era 0.05-0.5) Flujo rápido
    "k_bf": (0.01, 0.15),       # AUMENTADO (era 0.005-0.1) Flujo base
    "f_et0": (0.02, 0.12),      # REDUCIDO (era 0.02-0.2)  ET desde S0
    "f_et1": (0.005, 0.05),     # REDUCIDO (era 0.01-0.1)  ET desde S1
    "n_r": (2, 3),              # AUMENTADO (era 1-4) Reservorios Nash
    "k_r": (60, 150)            # CRÍTICO: AUMENTADO (era 5-25) Escala temporal Nash
}
    print(f"\nLímites de parámetros definidos para {len(bounds)} parámetros")
    
    # ===============================================================================
    # 4. CALIBRACIÓN POR BÚSQUEDA ALEATORIA
    # ===============================================================================
    
    print(f"\n{'='*60}")
    print("FASE 1: CALIBRACIÓN POR BÚSQUEDA ALEATORIA")
    print(f"{'='*60}")
    
    n_random_iterations = 200
    random_seed = 42
    
    best_params_random, scores_random, param_history = random_search_calibration(
        df_calib=df_calib,
        q_obs_calib=q_obs_calib,
        config=config,
        bounds=bounds,
        n_iterations=n_random_iterations,
        seed=random_seed,
        verbose=True
    )
    
    print_calibration_summary(best_params_random, scores_random, param_history)
    
    # ===============================================================================
    # 5. OPTIMIZACIÓN DETERMINÍSTICA
    # ===============================================================================
    
    print(f"\n{'='*60}")
    print("FASE 2: OPTIMIZACIÓN DETERMINÍSTICA (SciPy)")
    print(f"{'='*60}")
    
    best_params_scipy, error_history = scipy_optimization(
        df_calib=df_calib,
        q_obs_calib=q_obs_calib,
        config=config,
        initial_params=best_params_random,
        bounds=bounds,
        method="L-BFGS-B",
        verbose=True
    )
    
    # Usar los mejores parámetros
    best_params_final = best_params_scipy
    
    print(f"\nPARÁMETROS FINALES OPTIMIZADOS:")
    print_calibration_summary(best_params_final, [1-error_history[-1]], [best_params_final])
    
    # ===============================================================================
    # 6. SIMULACIÓN COMPLETA CON PARÁMETROS ÓPTIMOS
    # ===============================================================================
    
    print(f"\n{'='*60}")
    print("SIMULACIÓN CON PARÁMETROS ÓPTIMOS")
    print(f"{'='*60}")
    
    model_final = TankModel(params=best_params_final, config=config)
    sim_complete = model_final.run(df)
    
    # Extraer resultados
    q_sim_calib = sim_complete["Qout_mm"].iloc[:split_idx].values
    q_sim_valid = sim_complete["Qout_mm"].iloc[split_idx:].values
    q_sim_full = sim_complete["Qout_mm"].values
    
    # ===============================================================================
    # 7. CÁLCULO Y REPORTE DE MÉTRICAS
    # ===============================================================================
    
    print_performance_metrics(
        q_obs_calib, q_sim_calib,
        q_obs_valid, q_sim_valid, 
        q_obs_full, q_sim_full
    )
    
    # ===============================================================================
    # 8. VERIFICACIÓN DE BALANCE HÍDRICO
    # ===============================================================================
    
    print(f"\n{'='*60}")
    print("VERIFICACIÓN DE BALANCE HÍDRICO")
    print(f"{'='*60}")
    
    # Balance global
    P_total = df["P_mm"].sum()
    ET_total = sim_complete["ET_mm"].sum()
    Q_total = sim_complete["Qout_mm"].sum()
    
    # Cambio en almacenamiento
    storage_cols = ["S0", "S1", "S2", "S3"]
    initial_storage = 0.0  # Estados iniciales son cero
    final_storage = sim_complete[storage_cols].iloc[-1].sum()
    delta_storage = final_storage - initial_storage
    
    # Residual
    residual = P_total - ET_total - Q_total - delta_storage
    
    print(f"Precipitación total:     {P_total:10.2f} mm")
    print(f"Evapotranspiración:      {ET_total:10.2f} mm") 
    print(f"Caudal simulado:         {Q_total:10.2f} mm")
    print(f"Cambio almacenamiento:   {delta_storage:10.2f} mm")
    print(f"Residual de balance:     {residual:10.2f} mm")
    print(f"Error relativo:          {abs(residual)/P_total*100:8.4f} %")
    
    if abs(residual) < 0.01:
        print("✓ Balance hídrico verificado correctamente")
    else:
        print("⚠ Advertencia: Error en balance hídrico")
    
    # ===============================================================================
    # 9. PREPARACIÓN DE DATOS PARA GRÁFICAS
    # ===============================================================================
    
    results_df = pd.DataFrame({
        "Q_sim_m3s": sim_complete["Q_m3s"].values,
        "Q_obs_m3s": q_obs_full,
        "P_mm": df["P_mm"].values
    }, index=sim_complete.index)
    
    # ===============================================================================
    # 10. GENERACIÓN DE GRÁFICAS
    # ===============================================================================
    
    print(f"\n{'='*60}")
    print("GENERACIÓN DE GRÁFICAS")
    print(f"{'='*60}")
    
    output_dir = data_dir / "resultados"
    create_comprehensive_plots(results_df, split_idx, error_history, output_dir)
    
    # ===============================================================================
    # 11. EXPORTACIÓN DE RESULTADOS
    # ===============================================================================
    
    print(f"\n{'='*60}")
    print("EXPORTACIÓN DE RESULTADOS")
    print(f"{'='*60}")
    
    # Guardar simulación completa
    output_csv = output_dir / "simulacion_completa.csv"
    results_extended = results_df.copy()
    results_extended["ET_mm"] = sim_complete["ET_mm"].values
    results_extended["S_total"] = sim_complete[storage_cols].sum(axis=1).values
    results_extended.to_csv(output_csv, index=True)
    print(f"Simulación guardada en: {output_csv}")
    
    # Guardar parámetros óptimos
    params_csv = output_dir / "parametros_optimos.csv"
    params_data = {
        'Parametro': [],
        'Valor': [],
        'Unidad': []
    }
    
    param_units = {
        'S0_max': 'mm', 'alpha': '-', 'beta': '-',
        'k_qs': '/dt', 'k_inf': '/dt', 'k_perc': '/dt',
        'phi': '-', 'k_qf': '/dt', 'k_bf': '/dt',
        'f_et0': '/dt', 'f_et1': '/dt', 'n_r': '-', 'k_r': 'h'
    }
    
    for param_name in param_units.keys():
        if hasattr(best_params_final, param_name):
            params_data['Parametro'].append(param_name)
            params_data['Valor'].append(getattr(best_params_final, param_name))
            params_data['Unidad'].append(param_units[param_name])
    
    pd.DataFrame(params_data).to_csv(params_csv, index=False)
    print(f"Parámetros óptimos guardados en: {params_csv}")
    
    # Guardar métricas de desempeño
    metrics_data = {
        'Periodo': ['Calibracion', 'Validacion', 'Global'],
        'NSE': [nse(q_obs_calib, q_sim_calib), nse(q_obs_valid, q_sim_valid), nse(q_obs_full, q_sim_full)],
        'KGE': [kge(q_obs_calib, q_sim_calib), kge(q_obs_valid, q_sim_valid), kge(q_obs_full, q_sim_full)],
        'Bias_pct': [bias_pct(q_obs_calib, q_sim_calib), bias_pct(q_obs_valid, q_sim_valid), bias_pct(q_obs_full, q_sim_full)],
        'RMSE': [rmse(q_obs_calib, q_sim_calib), rmse(q_obs_valid, q_sim_valid), rmse(q_obs_full, q_sim_full)],
        'R2': [r_squared(q_obs_calib, q_sim_calib), r_squared(q_obs_valid, q_sim_valid), r_squared(q_obs_full, q_sim_full)]
    }
    
    metrics_csv = output_dir / "metricas_desempeno.csv"
    pd.DataFrame(metrics_data).to_csv(metrics_csv, index=False)
    print(f"Métricas de desempeño guardadas en: {metrics_csv}")
    
    print(f"\n{'='*60}")
    print("PROCESO COMPLETADO EXITOSAMENTE")
    print(f"{'='*60}")
    print(f"Todos los archivos de salida se encuentran en: {output_dir.resolve()}")
    
    return {
        'best_params': best_params_final,
        'results_df': results_df,
        'sim_complete': sim_complete,
        'config': config,
        'split_idx': split_idx
    }

# ===============================================================================
# EJECUCIÓN PRINCIPAL
# ===============================================================================

if __name__ == "__main__":
    try:
        results = main_calibration_validation()
        print("\n¡Ejecución completada sin errores!")
    except Exception as e:
        print(f"\nError durante la ejecución: {e}")
        import traceback
        traceback.print_exc()