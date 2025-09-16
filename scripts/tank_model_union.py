#!/usr/bin/env python3
"""
MODELO HIDROLÓGICO TIPO TANK - VERSIÓN MEJORADA CON VALIDACIONES COMPLETAS
==========================================================================

Versión mejorada del modelo hidrológico tipo Tank con 4 tanques, incluyendo:
- Balance de masas paso a paso y global
- Validaciones físicas comprehensivas  
- Criterios de evaluación específicos para datos diarios
- Estructura modular por bloques para revisión paso a paso
- Pruebas unitarias y verificaciones automáticas

Autor: Análisis basado en literatura científica de modelación hidrológica
Fecha: Agosto 2025
Orientado a: Simulaciones diarias en cuencas colombianas

Referencias clave:
- Nash & Sutcliffe (1970) - Métricas de eficiencia
- Gupta et al. (2009) - KGE y criterios multi-objetivo  
- Moriasi et al. (2007) - Criterios de evaluación para modelos diarios
- Krause et al. (2005) - Métricas modificadas para datos diarios
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Union, Dict, Tuple, List, Any
import warnings
import logging
from datetime import datetime
import json

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suprimir warnings menores
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ===============================================================================
# BLOQUE 1: CONFIGURACIÓN Y VALIDACIÓN DE PARÁMETROS
# ===============================================================================

@dataclass
class Parameters:
    """
    Parámetros del modelo Tank con validaciones físicas
    
    Todos los parámetros tienen rangos físicamente realistas para datos diarios
    """
    # Capacidades y forma (mm)
    S0_max: float = 50.0    # Capacidad tanque superficial [20-200 mm]
    alpha: float = 1.2      # No-linealidad escorrentía [1.0-3.0]
    beta: float = 1.0       # No-linealidad percolación [0.5-2.0]
    
    # Coeficientes de flujo por día [0-1]
    k_qs: float = 0.15      # Escorrentía directa [0.05-0.5]
    k_inf: float = 0.10     # Infiltración S0->S1 [0.02-0.6]
    k_perc: float = 0.05    # Percolación S1->(S2,S3) [0.005-0.2]
    phi: float = 0.6        # Fracción hacia flujo rápido [0.3-0.8]
    k_qf: float = 0.20      # Descarga rápida S2 [0.05-0.8]
    k_bf: float = 0.02      # Baseflow S3 [0.005-0.15]
    
    # Evapotranspiración por día [0-0.3]
    f_et0: float = 0.05     # Contribución S0 [0.01-0.2]
    f_et1: float = 0.02     # Contribución S1 [0.005-0.1]
    
    # Enrutamiento Nash (corregido para estabilidad)
    n_r: int = 2            # Número reservorios [1-4]
    k_r: float = 48.0       # Escala temporal Nash [24-240 h para dt=24h]
    
    def __post_init__(self):
        """Validar parámetros después de inicialización"""
        self.validate()
    
    def validate(self) -> Dict[str, Any]:
        """
        Validar parámetros físicos y numéricos
        
        Returns:
            Dict con resultados de validación
        """
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'parameter_status': {}
        }
        
        # Definir rangos válidos para cada parámetro
        valid_ranges = {
            'S0_max': (10, 500, 'mm'),
            'alpha': (1.0, 3.0, '-'),
            'beta': (0.5, 2.0, '-'),
            'k_qs': (0.01, 0.8, '/day'),
            'k_inf': (0.01, 0.8, '/day'),
            'k_perc': (0.001, 0.5, '/day'),
            'phi': (0.1, 0.9, '-'),
            'k_qf': (0.01, 1.0, '/day'),
            'k_bf': (0.001, 0.3, '/day'),
            'f_et0': (0.001, 0.5, '/day'),
            'f_et1': (0.001, 0.2, '/day'),
            'n_r': (1, 6, '-'),
            'k_r': (12, 500, 'h')
        }
        
        # Validar cada parámetro
        for param_name, (min_val, max_val, unit) in valid_ranges.items():
            param_value = getattr(self, param_name)
            status = 'valid'
            
            if param_value < min_val:
                validation_results['warnings'].append(
                    f"{param_name} = {param_value} < {min_val} {unit} (mínimo recomendado)"
                )
                status = 'below_range'
            elif param_value > max_val:
                validation_results['warnings'].append(
                    f"{param_name} = {param_value} > {max_val} {unit} (máximo recomendado)"
                )
                status = 'above_range'
            
            validation_results['parameter_status'][param_name] = {
                'value': param_value,
                'range': (min_val, max_val),
                'unit': unit,
                'status': status
            }
        
        # Validaciones de relaciones entre parámetros
        # 1. Estabilidad Nash
        if hasattr(self, 'k_r'):
            dt = 24.0  # Asumir datos diarios
            stability_ratio = dt / self.k_r
            if stability_ratio > 1.0:
                validation_results['errors'].append(
                    f"Nash inestable: dt/k_r = {stability_ratio:.2f} > 1.0"
                )
                validation_results['is_valid'] = False
            elif stability_ratio > 0.5:
                validation_results['warnings'].append(
                    f"Nash marginal: dt/k_r = {stability_ratio:.2f} > 0.5"
                )
        
        # 2. Balance ET
        total_et_capacity = self.f_et0 + self.f_et1
        if total_et_capacity > 0.3:
            validation_results['warnings'].append(
                f"Capacidad ET alta: {total_et_capacity:.3f} > 0.3 /day"
            )
        
        # 3. Coeficientes de flujo
        if self.k_qs + self.k_inf > 1.0:
            validation_results['warnings'].append(
                f"Flujos S0 altos: k_qs + k_inf = {self.k_qs + self.k_inf:.3f} > 1.0"
            )
        
        return validation_results
    
    def copy(self):
        """Crear copia profunda de parámetros"""
        return Parameters(
            S0_max=self.S0_max, alpha=self.alpha, beta=self.beta,
            k_qs=self.k_qs, k_inf=self.k_inf, k_perc=self.k_perc,
            phi=self.phi, k_qf=self.k_qf, k_bf=self.k_bf,
            f_et0=self.f_et0, f_et1=self.f_et1,
            n_r=self.n_r, k_r=self.k_r
        )

@dataclass
class States:
    """Estados de almacenamiento en los tanques (mm)"""
    S0: float = 0.0  # Almacenamiento superficial
    S1: float = 0.0  # Zona no saturada
    S2: float = 0.0  # Flujo rápido subsuperficial
    S3: float = 0.0  # Flujo base
    
    def total_storage(self) -> float:
        """Almacenamiento total en el sistema"""
        return self.S0 + self.S1 + self.S2 + self.S3
    
    def validate(self) -> bool:
        """Validar que todos los almacenamientos sean no-negativos"""
        return all(storage >= 0 for storage in [self.S0, self.S1, self.S2, self.S3])

@dataclass
class ModelConfig:
    """Configuración del modelo con validaciones"""
    area_km2: float                                # Área cuenca [km²]
    dt_hours: float = 24.0                        # Paso temporal [horas]
    route: bool = True                            # Aplicar enrutamiento Nash
    debug_balance: bool = True                    # Activar debug de balance
    debug_csv_path: Optional[str] = None         # Ruta para debug CSV
    mass_balance_tolerance: float = 1e-6         # Tolerancia balance de masas
    validation_level: str = 'strict'             # 'strict', 'moderate', 'lenient'
    
    def __post_init__(self):
        """Validar configuración"""
        if self.area_km2 <= 0:
            raise ValueError(f"Área debe ser > 0, recibido: {self.area_km2} km²")
        
        if self.dt_hours <= 0:
            raise ValueError(f"dt_hours debe ser > 0, recibido: {self.dt_hours}")
        
        if self.dt_hours != 24.0:
            logger.warning(f"Modelo optimizado para dt=24h, usando dt={self.dt_hours}h")
        
        # Configurar tolerancias según nivel de validación
        tolerance_levels = {
            'strict': 1e-8,
            'moderate': 1e-6,
            'lenient': 1e-4
        }
        
        if self.validation_level in tolerance_levels:
            self.mass_balance_tolerance = tolerance_levels[self.validation_level]
        else:
            logger.warning(f"Nivel '{self.validation_level}' no reconocido, usando 'moderate'")
            self.validation_level = 'moderate'
            self.mass_balance_tolerance = tolerance_levels['moderate']

# ===============================================================================
# BLOQUE 2: MÉTRICAS DE DESEMPEÑO PARA DATOS DIARIOS
# ===============================================================================

class DailyPerformanceMetrics:
    """
    Métricas de desempeño específicamente diseñadas para datos diarios
    
    Basado en Moriasi et al. (2007) y Krause et al. (2005)
    """
    
    @staticmethod
    def nse(obs: np.ndarray, sim: np.ndarray) -> float:
        """Nash-Sutcliffe Efficiency (NSE)"""
        obs, sim = DailyPerformanceMetrics._clean_data(obs, sim)
        if len(obs) < 2:
            return np.nan
        
        obs_mean = np.mean(obs)
        numerator = np.sum((obs - sim) ** 2)
        denominator = np.sum((obs - obs_mean) ** 2)
        
        return 1 - (numerator / denominator) if denominator > 0 else np.nan
    
    @staticmethod
    def log_nse(obs: np.ndarray, sim: np.ndarray) -> float:
        """NSE en escala logarítmica (mejor para flujos bajos)"""
        obs, sim = DailyPerformanceMetrics._clean_data(obs, sim)
        if len(obs) < 2 or np.any(obs <= 0) or np.any(sim <= 0):
            return np.nan
        
        log_obs = np.log(obs + 1e-6)
        log_sim = np.log(sim + 1e-6)
        
        return DailyPerformanceMetrics.nse(log_obs, log_sim)
    
    @staticmethod
    def kge(obs: np.ndarray, sim: np.ndarray) -> float:
        """Kling-Gupta Efficiency (KGE) - Gupta et al. 2009"""
        obs, sim = DailyPerformanceMetrics._clean_data(obs, sim)
        if len(obs) < 2:
            return np.nan
        
        # Componentes del KGE
        r = np.corrcoef(obs, sim)[0, 1] if np.var(obs) > 0 and np.var(sim) > 0 else 0
        alpha = np.std(sim) / np.std(obs) if np.std(obs) > 0 else np.nan
        beta = np.mean(sim) / np.mean(obs) if np.mean(obs) > 0 else np.nan
        
        if np.isnan(r) or np.isnan(alpha) or np.isnan(beta):
            return np.nan
        
        kge_val = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        return kge_val
    
    @staticmethod
    def pbias(obs: np.ndarray, sim: np.ndarray) -> float:
        """Percentage Bias (PBIAS)"""
        obs, sim = DailyPerformanceMetrics._clean_data(obs, sim)
        if len(obs) == 0:
            return np.nan
        
        obs_sum = np.sum(obs)
        if obs_sum == 0:
            return np.nan
        
        return 100.0 * (np.sum(sim) - obs_sum) / obs_sum
    
    @staticmethod
    def rmse(obs: np.ndarray, sim: np.ndarray) -> float:
        """Root Mean Square Error"""
        obs, sim = DailyPerformanceMetrics._clean_data(obs, sim)
        if len(obs) == 0:
            return np.nan
        
        return np.sqrt(np.mean((obs - sim) ** 2))
    
    @staticmethod
    def mae(obs: np.ndarray, sim: np.ndarray) -> float:
        """Mean Absolute Error"""
        obs, sim = DailyPerformanceMetrics._clean_data(obs, sim)
        if len(obs) == 0:
            return np.nan
        
        return np.mean(np.abs(obs - sim))
    
    @staticmethod
    def r_squared(obs: np.ndarray, sim: np.ndarray) -> float:
        """Coefficient of Determination (R²)"""
        obs, sim = DailyPerformanceMetrics._clean_data(obs, sim)
        if len(obs) < 2:
            return np.nan
        
        if np.var(obs) == 0 or np.var(sim) == 0:
            return np.nan
        
        correlation = np.corrcoef(obs, sim)[0, 1]
        return correlation**2 if not np.isnan(correlation) else np.nan
    
    @staticmethod
    def peak_error(obs: np.ndarray, sim: np.ndarray, percentile: float = 95.0) -> float:
        """Error en picos (percentil especificado)"""
        obs, sim = DailyPerformanceMetrics._clean_data(obs, sim)
        if len(obs) < 10:  # Mínimo para cálculo de percentiles
            return np.nan
        
        threshold = np.percentile(obs, percentile)
        peak_mask = obs >= threshold
        
        if np.sum(peak_mask) == 0:
            return np.nan
        
        obs_peaks = obs[peak_mask]
        sim_peaks = sim[peak_mask]
        
        return DailyPerformanceMetrics.pbias(obs_peaks, sim_peaks)
    
    @staticmethod
    def low_flow_error(obs: np.ndarray, sim: np.ndarray, percentile: float = 10.0) -> float:
        """Error en flujos bajos (percentil especificado)"""
        obs, sim = DailyPerformanceMetrics._clean_data(obs, sim)
        if len(obs) < 10:
            return np.nan
        
        threshold = np.percentile(obs, percentile)
        low_mask = obs <= threshold
        
        if np.sum(low_mask) == 0:
            return np.nan
        
        obs_low = obs[low_mask]
        sim_low = sim[low_mask]
        
        return DailyPerformanceMetrics.pbias(obs_low, sim_low)
    
    @staticmethod
    def _clean_data(obs: np.ndarray, sim: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Limpiar datos eliminando NaN e infinitos"""
        obs = np.asarray(obs, dtype=float)
        sim = np.asarray(sim, dtype=float)
        
        # Máscara para valores válidos
        valid_mask = (
            np.isfinite(obs) & np.isfinite(sim) & 
            (obs >= 0) & (sim >= 0)  # Caudales no pueden ser negativos
        )
        
        return obs[valid_mask], sim[valid_mask]
    
    @staticmethod
    def evaluate_daily_performance(obs: np.ndarray, sim: np.ndarray) -> Dict[str, float]:
        """
        Evaluación comprehensiva para modelos hidrológicos diarios
        
        Returns:
            Dict con todas las métricas calculadas
        """
        metrics = {}
        
        # Métricas principales
        metrics['NSE'] = DailyPerformanceMetrics.nse(obs, sim)
        metrics['LogNSE'] = DailyPerformanceMetrics.log_nse(obs, sim)
        metrics['KGE'] = DailyPerformanceMetrics.kge(obs, sim)
        metrics['PBIAS'] = DailyPerformanceMetrics.pbias(obs, sim)
        metrics['RMSE'] = DailyPerformanceMetrics.rmse(obs, sim)
        metrics['MAE'] = DailyPerformanceMetrics.mae(obs, sim)
        metrics['R2'] = DailyPerformanceMetrics.r_squared(obs, sim)
        
        # Métricas específicas para extremos
        metrics['Peak_Error_P95'] = DailyPerformanceMetrics.peak_error(obs, sim, 95.0)
        metrics['Low_Flow_Error_P10'] = DailyPerformanceMetrics.low_flow_error(obs, sim, 10.0)
        
        # Clasificación según Moriasi et al. (2007)
        metrics['Performance_Rating'] = DailyPerformanceMetrics._classify_performance(metrics)
        
        return metrics
    
    @staticmethod
    def _classify_performance(metrics: Dict[str, float]) -> str:
        """
        Clasificar desempeño según criterios de Moriasi et al. (2007)
        adaptados para datos diarios
        """
        nse = metrics.get('NSE', np.nan)
        pbias = abs(metrics.get('PBIAS', np.inf))
        
        if np.isnan(nse):
            return 'Insufficient_Data'
        
        # Criterios para datos diarios (más estrictos que mensuales)
        if nse > 0.75 and pbias < 10:
            return 'Very_Good'
        elif nse > 0.65 and pbias < 15:
            return 'Good'  
        elif nse > 0.50 and pbias < 25:
            return 'Satisfactory'
        elif nse > 0.20 and pbias < 40:
            return 'Poor'
        else:
            return 'Unsatisfactory'

# ===============================================================================
# BLOQUE 3: VALIDACIÓN DE BALANCE DE MASAS
# ===============================================================================

class MassBalanceValidator:
    """
    Validador de balance de masas para modelo hidrológico
    
    Implementa verificaciones paso a paso y globales
    """
    
    def __init__(self, tolerance: float = 1e-6, debug: bool = False):
        self.tolerance = tolerance
        self.debug = debug
        self.balance_history = []
        self.errors = []
        
    def validate_step(self, step_data: Dict[str, float], step_idx: int) -> Dict[str, Any]:
        """
        Validar balance de masas en un paso temporal
        
        Args:
            step_data: Diccionario con flujos del paso
            step_idx: Índice del paso temporal
            
        Returns:
            Dict con resultados de validación
        """
        # Extraer componentes
        P = step_data.get('P_mm', 0.0)
        ET = step_data.get('ET_mm', 0.0)
        Q_out = step_data.get('Qout_mm', 0.0)
        
        # Estados inicial y final
        S_initial = step_data.get('S_initial', 0.0)
        S_final = step_data.get('S_final', 0.0)
        delta_S = S_final - S_initial
        
        # Balance: P - ET - Q - ΔS = 0
        balance_error = P - ET - Q_out - delta_S
        
        # Evaluar error relativo
        total_input = P
        relative_error = abs(balance_error) / max(total_input, 1e-6)
        
        # Determinar estado
        if abs(balance_error) < self.tolerance:
            status = 'OK'
        elif relative_error < 0.01:  # 1%
            status = 'WARNING'
        else:
            status = 'ERROR'
        
        result = {
            'step': step_idx,
            'balance_error': balance_error,
            'relative_error': relative_error,
            'status': status,
            'components': {
                'P': P, 'ET': ET, 'Q': Q_out, 'dS': delta_S
            }
        }
        
        # Guardar historial
        self.balance_history.append(result)
        
        # Log errores significativos
        if status == 'ERROR':
            error_msg = (f"Paso {step_idx}: Error balance = {balance_error:.6f} mm "
                        f"(relativo = {relative_error:.4f})")
            self.errors.append(error_msg)
            if self.debug:
                logger.error(error_msg)
        
        return result
    
    def validate_global_balance(self, simulation_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validar balance de masas global de toda la simulación
        
        Args:
            simulation_data: DataFrame con resultados completos
            
        Returns:
            Dict con análisis de balance global
        """
        # Totales globales
        P_total = simulation_data['P_mm'].sum()
        ET_total = simulation_data['ET_mm'].sum() 
        Q_total = simulation_data['Qout_mm'].sum()

        # Cambio total de almacenamiento
        storage_cols = ['S0', 'S1', 'S2', 'S3']
        if all(col in simulation_data.columns for col in storage_cols):
            S_initial = simulation_data[storage_cols].iloc[0].sum()
            S_final = simulation_data[storage_cols].iloc[-1].sum()
            delta_S_total = S_final - S_initial
        else:
            delta_S_total = 0.0
            logger.warning("Columnas de almacenamiento no encontradas")
        
        # ===== NUEVO: almacenamiento del enrutamiento (cola que no salió) =====
        routing_tail = 0.0
        if 'Qraw_mm' in simulation_data.columns:
            Qpre_total = simulation_data['Qraw_mm'].sum()
        elif 'Qin' in simulation_data.columns:
            Qpre_total = simulation_data['Qin'].sum()
        else:
            Qpre_total = None

        if Qpre_total is not None:
            # Si el pre-ruteo supera al Qout dentro de la ventana,
            # la diferencia queda almacenada en la cascada de Nash
            routing_tail = max(0.0, Qpre_total - Q_total)

        # ===== Balance global con cola del Nash =====
        global_balance_error   = P_total - ET_total - Q_total - delta_S_total - routing_tail
        global_relative_error  = abs(global_balance_error) / max(P_total, 1e-6)

        # Estadísticas de errores por paso
        step_errors = [result['balance_error'] for result in self.balance_history]
        if step_errors:
            max_step_error = max(abs(e) for e in step_errors)
            mean_step_error = np.mean([abs(e) for e in step_errors])
            std_step_error = np.std([abs(e) for e in step_errors])
        else:
            max_step_error = mean_step_error = std_step_error = np.nan
        
        # Estado global
        if global_relative_error < 0.001:  # 0.1%
            global_status = 'EXCELLENT'
        elif global_relative_error < 0.01:  # 1%
            global_status = 'GOOD'
        elif global_relative_error < 0.05:  # 5%
            global_status = 'ACCEPTABLE'
        else:
            global_status = 'POOR'
        
        return {
            'global_balance_error': global_balance_error,
            'global_relative_error': global_relative_error,
            'global_status': global_status,
            'components': {
                'P_total': P_total,
                'ET_total': ET_total, 
                'Q_total': Q_total,
                'dS_total': delta_S_total
            },
            'step_statistics': {
                'max_error': max_step_error,
                'mean_error': mean_step_error,
                'std_error': std_step_error,
                'n_errors': len(self.errors)
            },
            'simulation_period': {
                'start_date': simulation_data.index[0] if len(simulation_data) > 0 else None,
                'end_date': simulation_data.index[-1] if len(simulation_data) > 0 else None,
                'n_days': len(simulation_data)
            }
        }
    
    def generate_balance_report(self) -> str:
        """Generar reporte de balance de masas"""
        if not self.balance_history:
            return "No hay datos de balance disponibles"
        
        n_steps = len(self.balance_history)
        n_errors = len(self.errors)
        n_warnings = sum(1 for r in self.balance_history if r['status'] == 'WARNING')
        n_ok = sum(1 for r in self.balance_history if r['status'] == 'OK')
        
        report = f"""
REPORTE DE BALANCE DE MASAS
{'='*50}
Pasos simulados: {n_steps}
Estados:
  - OK: {n_ok} ({n_ok/n_steps*100:.1f}%)
  - WARNING: {n_warnings} ({n_warnings/n_steps*100:.1f}%)
  - ERROR: {n_errors} ({n_errors/n_steps*100:.1f}%)

Tolerancia configurada: {self.tolerance:.2e}
        """
        
        if self.errors:
            report += f"\nPrimeros errores:\n"
            for error in self.errors[:5]:
                report += f"  - {error}\n"
            if len(self.errors) > 5:
                report += f"  ... y {len(self.errors)-5} errores más\n"
        
        return report

# ===============================================================================
# BLOQUE 4: ENRUTAMIENTO NASH CORREGIDO CON VALIDACIONES
# ===============================================================================

def nash_cascade_validated(inflow: np.ndarray, n: int, k: float, dt: float, 
                          validate: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Enrutamiento Nash con validaciones automáticas
    
    Returns:
        Tuple de (serie_enrutada, reporte_validacion)
    """
    validation_report = {
        'is_stable': True,
        'mass_conserved': True,
        'warnings': [],
        'errors': [],
        'statistics': {}
    }
    
    # Validar parámetros de entrada
    if n <= 0:
        validation_report['errors'].append("n debe ser >= 1")
        validation_report['is_stable'] = False
        return np.array(inflow), validation_report
    
    if k <= 0:
        validation_report['errors'].append("k debe ser > 0")
        validation_report['is_stable'] = False
        return np.array(inflow), validation_report
    
    # Verificar estabilidad numérica
    stability_ratio = dt / k
    if stability_ratio > 1.0:
        validation_report['errors'].append(
            f"Nash inestable: dt/k = {stability_ratio:.3f} > 1.0"
        )
        validation_report['is_stable'] = False
    elif stability_ratio > 0.5:
        validation_report['warnings'].append(
            f"Nash marginal: dt/k = {stability_ratio:.3f} > 0.5"
        )
    
    # Limpiar datos de entrada
    inflow = np.asarray(inflow, dtype=float)
    inflow = np.nan_to_num(inflow, nan=0.0, posinf=0.0, neginf=0.0)
    
    if len(inflow) == 0:
        return np.array([]), validation_report
    
    # Implementar algoritmo estable
    outflow = np.zeros_like(inflow)
    state = np.zeros(n)
    
    # Coeficientes para esquema implícito
    alpha = dt / k
    recession_factor = 1.0 / (1.0 + alpha)
    input_factor = alpha / (1.0 + alpha)
    
    # Ejecutar enrutamiento
    for i in range(len(inflow)):
        q_in = max(0.0, inflow[i])  # Prevenir valores negativos
        current_input = q_in
        
        for j in range(n):
            # Actualizar estado con esquema implícito
            state[j] = recession_factor * (state[j] + input_factor * current_input / recession_factor)
            current_output = state[j]
            current_input = current_output
        
        outflow[i] = current_output
    
    # Validar conservación de masa
    if validate:
        mass_input = np.sum(inflow)
        mass_output = np.sum(outflow)
        mass_error = abs(mass_input - mass_output)
        relative_mass_error = mass_error / max(mass_input, 1e-10)
        
        if relative_mass_error > 0.01:  # 1%
            validation_report['mass_conserved'] = False
            validation_report['errors'].append(
                f"Error conservación masa: {relative_mass_error:.4f}"
            )
        elif relative_mass_error > 0.001:  # 0.1%
            validation_report['warnings'].append(
                f"Error conservación masa: {relative_mass_error:.4f}"
            )
        
        # Estadísticas
        validation_report['statistics'] = {
            'mass_input': mass_input,
            'mass_output': mass_output,
            'mass_error': mass_error,
            'relative_mass_error': relative_mass_error,
            'peak_input': np.max(inflow),
            'peak_output': np.max(outflow),
            'peak_delay_steps': np.argmax(outflow) - np.argmax(inflow) if np.max(inflow) > 0 else 0,
            'theoretical_peak_delay': (n - 1) * k / dt
        }
    
    return outflow, validation_report

# ===============================================================================
# BLOQUE 5: MODELO TANK CON VALIDACIONES INTEGRADAS
# ===============================================================================

class TankModelValidated:
    """
    Modelo Tank con validaciones automáticas y balance de masas
    """
    
    def __init__(self, params: Parameters, config: ModelConfig, init_states: States = None):
        self.params = params
        self.config = config
        self.states = init_states if init_states is not None else States()
        
        # Validar parámetros al inicializar
        param_validation = self.params.validate()
        if not param_validation['is_valid']:
            logger.error("Parámetros inválidos:")
            for error in param_validation['errors']:
                logger.error(f"  - {error}")
            raise ValueError("Parámetros del modelo no válidos")
        
        # Advertir sobre parámetros en rangos marginales
        for warning in param_validation['warnings']:
            logger.warning(warning)
        
        # Inicializar validador de balance
        self.balance_validator = MassBalanceValidator(
            tolerance=config.mass_balance_tolerance,
            debug=config.debug_balance
        )
        
        # Historial de simulación para debug
        self.simulation_history = []
        
    def step_with_validation(self, P: float, PET: float, step_idx: int = 0) -> Dict[str, Any]:
        """
        Un paso del modelo con validaciones completas
        
        Args:
            P: Precipitación [mm]
            PET: Evapotranspiración potencial [mm]  
            step_idx: Índice del paso (para debug)
            
        Returns:
            Dict con resultados y validaciones
        """
        # Validar entradas
        P = max(0.0, float(P) if np.isfinite(P) else 0.0)
        PET = max(0.0, float(PET) if np.isfinite(PET) else 0.0)
        
        # Almacenar estado inicial para balance
        S_initial = self.states.total_storage()
        
        # Ejecutar paso del modelo
        step_result = self._execute_tank_step(P, PET)
        
        # Almacenar estado final
        S_final = self.states.total_storage()
        
        # Validar balance de masas
        balance_data = {
            'P_mm': P,
            'ET_mm': step_result['ET_mm'],
            'Qout_mm': step_result['Qin'],  # Antes del enrutamiento
            'S_initial': S_initial,
            'S_final': S_final
        }
        
        balance_result = self.balance_validator.validate_step(balance_data, step_idx)
        step_result['balance_validation'] = balance_result
        
        # Validar estados físicos
        if not self.states.validate():
            logger.error(f"Paso {step_idx}: Estados negativos detectados")
            step_result['physical_validation'] = 'ERROR'
        else:
            step_result['physical_validation'] = 'OK'
        
        # Agregar información de estado
        step_result.update({
            'S0': self.states.S0,
            'S1': self.states.S1, 
            'S2': self.states.S2,
            'S3': self.states.S3,
            'S_total': S_final
        })
        
        # Guardar en historial si está habilitado el debug
        if self.config.debug_balance:
            debug_data = {
                'step': step_idx,
                'inputs': {'P': P, 'PET': PET},
                'outputs': step_result.copy(),
                'states': {
                    'S0': self.states.S0, 'S1': self.states.S1,
                    'S2': self.states.S2, 'S3': self.states.S3
                }
            }
            self.simulation_history.append(debug_data)
        
        return step_result
    
    def _execute_tank_step(self, P: float, PET: float) -> Dict[str, float]:
        """Ejecutar lógica básica del modelo Tank"""
        p, s = self.params, self.states
        
        # === EVAPOTRANSPIRACIÓN ===
        ET_capacity = p.f_et0 * s.S0 + p.f_et1 * s.S1
        ET = min(PET, ET_capacity)
        
        # === TANQUE S0 (SUPERFICIE) ===
        S0 = s.S0 + P - ET
        S0 = max(0.0, S0)
        
        # Flujos desde S0
        Qs = p.k_qs * (S0 ** p.alpha) if S0 > 0 else 0.0
        I = p.k_inf * S0 if S0 > 0 else 0.0
        
        # Manejo de exceso de capacidad
        if S0 > p.S0_max:
            excess = S0 - p.S0_max
            Qs += excess
            S0 = p.S0_max
        
        # Verificar disponibilidad de agua
        total_out_S0 = Qs + I
        if total_out_S0 > S0 and total_out_S0 > 0:
            factor = S0 / total_out_S0
            Qs *= factor
            I *= factor
            S0 = 0.0
        else:
            S0 = max(0.0, S0 - total_out_S0)
        
        # === TANQUE S1 (ZONA NO SATURADA) ===
        S1 = s.S1 + I
        Perc = p.k_perc * (S1 ** p.beta) if S1 > 0 else 0.0
        Perc = min(Perc, S1)  # No puede perder más de lo que tiene
        S1 = max(0.0, S1 - Perc)
        
        # Partición de percolación
        to_S2 = p.phi * Perc
        to_S3 = (1 - p.phi) * Perc
        
        # === TANQUE S2 (FLUJO RÁPIDO) ===
        S2 = s.S2 + to_S2
        Qf = p.k_qf * S2 if S2 > 0 else 0.0
        Qf = min(Qf, S2)  # No puede descargar más de lo que tiene
        S2 = max(0.0, S2 - Qf)
        
        # === TANQUE S3 (FLUJO BASE) ===
        S3 = s.S3 + to_S3
        Qb = p.k_bf * S3 if S3 > 0 else 0.0
        Qb = min(Qb, S3)  # No puede descargar más de lo que tiene
        S3 = max(0.0, S3 - Qb)
        
        # Actualizar estados
        self.states = States(S0=S0, S1=S1, S2=S2, S3=S3)
        
        # Caudal total
        Qin = Qs + Qf + Qb
        
        return {
            "ET_mm": ET, "I_mm": I, "Perc_mm": Perc,
            "Qs_mm": Qs, "Qf_mm": Qf, "Qb_mm": Qb,
            "Qin": Qin
        }
    
    def run_validated(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Ejecutar modelo completo con validaciones
        
        Returns:
            Tuple de (resultados_simulacion, reporte_validacion)
        """
        logger.info("Iniciando simulación con validaciones...")
        
        # Verificar columnas requeridas
        required_cols = ["P_mm", "PET_mm"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Faltan columnas: {missing_cols}")
        
        # Limpiar datos de entrada
        df_clean = self._clean_input_data(df.copy())
        
        # Reinicializar estados y validador
        self.states = States()
        self.balance_validator = MassBalanceValidator(
            tolerance=self.config.mass_balance_tolerance,
            debug=self.config.debug_balance
        )
        
        # Ejecutar simulación paso a paso
        results = []
        logger.info(f"Simulando {len(df_clean)} pasos temporales...")
        
        for i in range(len(df_clean)):
            if i % 100 == 0:  # Log progreso cada 100 pasos
                logger.info(f"Procesando paso {i}/{len(df_clean)}")
                
            step_result = self.step_with_validation(
                df_clean.iloc[i]["P_mm"],
                df_clean.iloc[i]["PET_mm"],
                step_idx=i
            )
            results.append(step_result)
        
        # Convertir a DataFrame
        out_df = pd.DataFrame(results, index=df_clean.index)
        out_df['P_mm']   = df_clean['P_mm'].values
        out_df['PET_mm'] = df_clean['PET_mm'].values

        # === ENRUTAMIENTO NASH CON VALIDACIÓN ===
        routing_report = {'applied': False}
        if self.config.route and self.params.n_r > 0 and self.params.k_r > 0:
            logger.info("Aplicando enrutamiento Nash...")
            routed, routing_report = nash_cascade_validated(
                out_df["Qin"].values,
                n=int(self.params.n_r),
                k=self.params.k_r,
                dt=self.config.dt_hours
            )
            out_df["Qout_mm"] = routed
            out_df["Qraw_mm"] = out_df["Qin"].copy()
            routing_report['applied'] = True
        else:
            out_df["Qout_mm"] = out_df["Qin"].copy()
            out_df["Qraw_mm"] = out_df["Qin"].copy()
        
        # Conversión a m³/s
        seconds_per_dt = self.config.dt_hours * 3600.0
        mm_to_m3s_factor = self.config.area_km2 * 1e6 * 1e-3 / seconds_per_dt
        out_df["Q_m3s"] = out_df["Qout_mm"] * mm_to_m3s_factor
        
        # === VALIDACIÓN GLOBAL ===
        logger.info("Validando balance global...")
        global_balance = self.balance_validator.validate_global_balance(out_df)
        
        # Compilar reporte de validación
        validation_report = {
            'parameter_validation': self.params.validate(),
            'balance_validation': global_balance,
            'routing_validation': routing_report,
            'simulation_summary': {
                'n_steps': len(out_df),
                'period': (df_clean.index[0], df_clean.index[-1]) if len(df_clean) > 0 else (None, None),
                'data_quality': self._assess_data_quality(df_clean)
            }
        }
        
        # Log resumen
        self._log_validation_summary(validation_report)
        
        return out_df, validation_report
    
    def _clean_input_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpiar y validar datos de entrada"""
        required_cols = ["P_mm", "PET_mm"]
        
        for col in required_cols:
            # Reemplazar valores inválidos
            invalid_mask = df[col].isna() | (df[col] < 0) | ~np.isfinite(df[col])
            n_invalid = invalid_mask.sum()
            
            if n_invalid > 0:
                logger.warning(f"Limpiando {n_invalid} valores inválidos en {col}")
                
                # Interpolación para gaps pequeños
                df[col] = df[col].mask(invalid_mask).interpolate(
                    method='linear', limit=7  # Máximo 7 días consecutivos
                ).fillna(0.0)  # Rellenar restantes con 0
        
        return df
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluar calidad de los datos de entrada"""
        quality_report = {}
        
        for col in ["P_mm", "PET_mm"]:
            if col in df.columns:
                data = df[col]
                quality_report[col] = {
                    'n_total': len(data),
                    'n_missing': data.isna().sum(),
                    'n_zero': (data == 0).sum(),
                    'mean': float(data.mean()),
                    'std': float(data.std()),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'quality_score': self._calculate_quality_score(data)
                }
        
        return quality_report
    
    def _calculate_quality_score(self, data: pd.Series) -> float:
        """Calcular score de calidad de datos (0-1)"""
        if len(data) == 0:
            return 0.0
        
        # Penalizar por valores faltantes
        missing_penalty = data.isna().sum() / len(data)
        
        # Penalizar por valores extremos (outliers)
        if data.std() > 0:
            z_scores = np.abs((data - data.mean()) / data.std())
            outlier_penalty = (z_scores > 3).sum() / len(data)
        else:
            outlier_penalty = 0.0
        
        # Score final
        quality_score = 1.0 - missing_penalty - outlier_penalty * 0.5
        return max(0.0, min(1.0, quality_score))
    
    def _log_validation_summary(self, validation_report: Dict[str, Any]):
        """Log resumen de validación"""
        logger.info("="*60)
        logger.info("RESUMEN DE VALIDACIÓN")
        logger.info("="*60)
        
        # Balance de masas
        balance = validation_report['balance_validation']
        logger.info(f"Balance global: {balance['global_status']}")
        logger.info(f"Error relativo: {balance['global_relative_error']:.6f}")
        
        # Enrutamiento
        routing = validation_report['routing_validation']
        if routing['applied']:
            if routing.get('is_stable', True):
                logger.info("Enrutamiento Nash: ESTABLE")
            else:
                logger.warning("Enrutamiento Nash: INESTABLE")
        
        # Datos
        sim_summary = validation_report['simulation_summary']
        logger.info(f"Pasos simulados: {sim_summary['n_steps']}")
        
        # Balance por pasos
        balance_report = self.balance_validator.generate_balance_report()
        logger.info(balance_report)
    
    def export_debug_data(self, filepath: str):
        """Exportar datos de debug a CSV"""
        if not self.simulation_history:
            logger.warning("No hay datos de debug para exportar")
            return
        
        # Convertir historial a DataFrame
        debug_rows = []
        for step_data in self.simulation_history:
            row = {
                'step': step_data['step'],
                'P_mm': step_data['inputs']['P'],
                'PET_mm': step_data['inputs']['PET'],
                **step_data['states'],
                **{k: v for k, v in step_data['outputs'].items() 
                   if isinstance(v, (int, float, np.number))}
            }
            debug_rows.append(row)
        
        debug_df = pd.DataFrame(debug_rows)
        debug_df.to_csv(filepath, index=False)
        logger.info(f"Datos debug exportados a: {filepath}")

# ===============================================================================  
# BLOQUE 6: FUNCIONES DE PRUEBA Y VALIDACIÓN
# ===============================================================================

def test_model_components():
    """Ejecutar pruebas unitarias de componentes del modelo"""
    
    print("="*60)
    print("PRUEBAS UNITARIAS DEL MODELO")
    print("="*60)
    
    # Test 1: Validación de parámetros
    print("\n1. Test de validación de parámetros...")
    
    # Parámetros válidos
    params_valid = Parameters()
    validation = params_valid.validate()
    assert validation['is_valid'], "Parámetros válidos fallan validación"
    print("   ✓ Parámetros válidos pasan validación")
    
    # Parámetros inválidos (Nash inestable)
    params_invalid = Parameters(k_r=10.0)  # dt=24, k=10 -> ratio > 1
    validation = params_invalid.validate()
    assert not validation['is_valid'], "Parámetros inválidos pasan validación"
    print("   ✓ Parámetros inválidos detectados correctamente")
    
    # Test 2: Balance de masas
    print("\n2. Test de balance de masas...")
    
    validator = MassBalanceValidator(tolerance=1e-6)
    
    # Caso perfecto (P = Q, sin ET ni cambio de almacenamiento)
    perfect_balance = {
        'P_mm': 10.0, 'ET_mm': 0.0, 'Qout_mm': 10.0,
        'S_initial': 0.0, 'S_final': 0.0
    }
    result = validator.validate_step(perfect_balance, 0)
    assert result['status'] == 'OK', "Balance perfecto falla"
    print("   ✓ Balance perfecto validado")
    
    # Caso con error
    error_balance = {
        'P_mm': 10.0, 'ET_mm': 0.0, 'Qout_mm': 15.0,  # Q > P (imposible)
        'S_initial': 0.0, 'S_final': 0.0
    }
    result = validator.validate_step(error_balance, 1)
    assert result['status'] == 'ERROR', "Error de balance no detectado"
    print("   ✓ Error de balance detectado correctamente")
    
    # Test 3: Enrutamiento Nash
    print("\n3. Test de enrutamiento Nash...")
    
    # Impulso unitario
    impulse = np.array([1.0] + [0.0] * 99)
    routed, report = nash_cascade_validated(impulse, n=2, k=48.0, dt=24.0)
    
    assert report['is_stable'], "Enrutamiento estable reportado como inestable"
    assert report['mass_conserved'], "Masa no conservada en enrutamiento"
    print("   ✓ Enrutamiento Nash estable y conserva masa")
    
    # Test 4: Modelo completo con datos sintéticos
    print("\n4. Test de modelo completo...")
    
    # Generar datos sintéticos
    dates = pd.date_range('2020-01-01', periods=30, freq='D')
    df_test = pd.DataFrame({
        'P_mm': np.random.exponential(2.0, 30),
        'PET_mm': np.full(30, 3.0)
    }, index=dates)
    
    # Configurar modelo
    config = ModelConfig(area_km2=1.0, validation_level='strict')
    params = Parameters(k_r=48.0)  # Asegurar estabilidad
    model = TankModelValidated(params, config)
    
    # Simular
    try:
        results, validation = model.run_validated(df_test)
        print("   ✓ Simulación completa exitosa")
        
        # Verificar balance global
        balance_status = validation['balance_validation']['global_status']
        assert balance_status in ['EXCELLENT', 'GOOD', 'ACCEPTABLE'], f"Balance {balance_status}"
        print(f"   ✓ Balance global: {balance_status}")
        
    except Exception as e:
        print(f"   ✗ Error en simulación: {e}")
        raise
    
    print("\n" + "="*60)
    print("TODAS LAS PRUEBAS PASARON EXITOSAMENTE")
    print("="*60)

if __name__ == "__main__":
    # Ejecutar pruebas si se ejecuta directamente
    test_model_components()
    
# ===============================================================================
# BLOQUE 7: CALIBRACIÓN MEJORADA CON VALIDACIONES MÚLTIPLES
# ===============================================================================

class ImprovedCalibrator:
    """
    Calibrador mejorado con múltiples criterios de evaluación
    """
    
    def __init__(self, config: ModelConfig, validation_level: str = 'strict'):
        self.config = config
        self.validation_level = validation_level
        self.calibration_history = []
        
    def multi_objective_evaluation(self, obs: np.ndarray, sim: np.ndarray, 
                                 weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Evaluación multi-objetivo específica para datos diarios
        
        Combina múltiples métricas con pesos específicos para diferentes
        aspectos del comportamiento hidrológico
        """
        if weights is None:
            weights = {
                'nse': 0.2,           # Ajuste general
                'log_nse': 0.15,       # Flujos bajos
                'kge': 0.20,           # Balance general
                'peak_error': 0.45,    # Picos (P95)
                'low_flow_error': 0.15 # Flujos bajos (P10)
            }
        
        # Calcular métricas individuales
        metrics = DailyPerformanceMetrics.evaluate_daily_performance(obs, sim)
        
        # Normalizar métricas a scores (0-1, donde 1 es perfecto)
        scores = {}
        
        # NSE: 1 es perfecto, -∞ es pésimo -> normalizar a (0,1)
        nse_val = metrics.get('NSE', -999)
        scores['nse'] = max(0, nse_val) if nse_val > 0 else 0
        
        # Log NSE: similar al NSE
        log_nse_val = metrics.get('LogNSE', -999)
        scores['log_nse'] = max(0, log_nse_val) if log_nse_val > 0 else 0
        
        # KGE: 1 es perfecto, -∞ es pésimo
        kge_val = metrics.get('KGE', -999)
        scores['kge'] = max(0, kge_val) if kge_val > 0 else 0
        
        # Error en picos: convertir PBIAS a score (0% error = 1, 100% error = 0)
        peak_error = abs(metrics.get('Peak_Error_P95', 100))
        scores['peak_error'] = max(0, 1 - peak_error / 100)
        
        # Error en flujos bajos: similar
        low_flow_error = abs(metrics.get('Low_Flow_Error_P10', 100))
        scores['low_flow_error'] = max(0, 1 - low_flow_error / 100)
        
        # Score combinado
        combined_score = sum(weights[key] * scores[key] for key in weights.keys())
        
        # Penalizaciones por problemas físicos
        penalties = self._calculate_penalties(obs, sim)
        final_score = combined_score * (1 - penalties)
        
        result = {
            'combined_score': final_score,
            'individual_scores': scores,
            'metrics': metrics,
            'penalties': penalties,
            'weights': weights
        }
        
        return result
    
    def _calculate_penalties(self, obs: np.ndarray, sim: np.ndarray) -> float:
        """Calcular penalizaciones por comportamientos no físicos"""
        penalty = 0.0
        
        # Penalizar valores negativos
        if np.any(sim < 0):
            penalty += 0.5
            
        # Penalizar bias extremo (>±50%)
        bias = DailyPerformanceMetrics.pbias(obs, sim)
        if abs(bias) > 50:
            penalty += 0.2
            
        # Penalizar correlación muy baja
        r2 = DailyPerformanceMetrics.r_squared(obs, sim)
        if r2 < 0.3:
            penalty += 0.1
            
        return min(1.0, penalty)  # Máximo penalty = 1.0
    
    def calibrate_with_validation(self, df_calib: pd.DataFrame, q_obs_calib: np.ndarray,
                                bounds: Dict[str, Tuple[float, float]], 
                                n_iterations: int = 500) -> Dict[str, Any]:
        """
        Calibración con validaciones integradas
        """
        logger.info("Iniciando calibración mejorada...")
        logger.info(f"Iteraciones: {n_iterations}")
        logger.info(f"Nivel de validación: {self.validation_level}")
        
        # Inicializar resultados
        best_score = -np.inf
        best_params = None
        best_validation = None
        
        rng = np.random.default_rng(42)
        
        for iteration in range(n_iterations):
            try:
                # Generar parámetros aleatorios
                params = self._generate_random_parameters(bounds, rng)
                
                # Crear modelo
                model = TankModelValidated(params, self.config)
                
                # Simular
                sim_results, validation = model.run_validated(df_calib)
                
                # Verificar que la simulación sea válida
                if not self._is_simulation_valid(validation):
                    continue
                
                # Evaluar desempeño
                q_sim = sim_results["Qout_mm"].values
                evaluation = self.multi_objective_evaluation(q_obs_calib, q_sim)
                score = evaluation['combined_score']
                
                # Guardar en historial
                self.calibration_history.append({
                    'iteration': iteration,
                    'params': params,
                    'score': score,
                    'evaluation': evaluation,
                    'validation': validation
                })
                
                # Actualizar mejor resultado
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    best_validation = validation
                    
                # Log progreso
                if (iteration + 1) % 50 == 0:
                    logger.info(f"Iteración {iteration+1}: Score actual = {score:.4f}, "
                              f"Mejor = {best_score:.4f}")
                    
            except Exception as e:
                logger.debug(f"Error en iteración {iteration}: {e}")
                continue
        
        # Compilar resultados
        calibration_results = {
            'best_params': best_params,
            'best_score': best_score,
            'best_validation': best_validation,
            'n_successful_iterations': len(self.calibration_history),
            'success_rate': len(self.calibration_history) / n_iterations,
            'history': self.calibration_history
        }
        
        logger.info(f"Calibración completada: {len(self.calibration_history)}/{n_iterations} "
                   f"iteraciones exitosas ({calibration_results['success_rate']:.2%})")
        
        return calibration_results
    
    def _generate_random_parameters(self, bounds: Dict[str, Tuple[float, float]], 
                                  rng: np.random.Generator) -> Parameters:
        """Generar parámetros aleatorios dentro de bounds válidos"""
        params = Parameters()
        
        for name, (low, high) in bounds.items():
            if name == "n_r":
                # Parámetro entero
                setattr(params, name, int(rng.integers(low, high + 1)))
            else:
                # Parámetro continuo
                setattr(params, name, rng.uniform(low, high))
        
        return params
    
    def _is_simulation_valid(self, validation: Dict[str, Any]) -> bool:
        """Verificar si una simulación es válida según criterios estrictos"""
        
        # Balance de masas debe ser aceptable
        balance_status = validation['balance_validation']['global_status']
        if balance_status not in ['EXCELLENT', 'GOOD', 'ACCEPTABLE']:
            return False
        
        # Enrutamiento debe ser estable si se aplica
        routing = validation.get('routing_validation', {})
        if routing.get('applied', False) and not routing.get('is_stable', True):
            return False
        
        # Parámetros deben ser válidos
        param_valid = validation.get('parameter_validation', {})
        if not param_valid.get('is_valid', False):
            return False
        
        return True

# ===============================================================================
# BLOQUE 8: FUNCIÓN PRINCIPAL INTEGRADA
# ===============================================================================

def run_complete_analysis(data_file: str, area_km2: float, 
                         output_dir: str = "resultados_mejorados") -> Dict[str, Any]:
    """
    Función principal para análisis completo del modelo Tank
    
    Args:
        data_file: Ruta al archivo CSV con datos
        area_km2: Área de la cuenca en km²
        output_dir: Directorio de salida
        
    Returns:
        Dict con resultados completos
    """
    
    print("="*80)
    print("ANÁLISIS COMPLETO DEL MODELO TANK CON VALIDACIONES")
    print("="*80)
    
    # Crear directorio de salida
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # PASO 1: Carga y validación de datos
    print("\n1. CARGANDO Y VALIDANDO DATOS...")
    print("-" * 40)
    
    try:
        df = pd.read_csv(data_file, parse_dates=["date"], index_col="date")
        print(f"✓ Datos cargados: {len(df)} registros")
        print(f"  Período: {df.index[0]} - {df.index[-1]}")
        
        # Verificar columnas requeridas
        required_cols = ["P_mm", "PET_mm", "Qobs_m3s"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas: {missing}")
        
    except Exception as e:
        print(f"✗ Error cargando datos: {e}")
        return {"error": str(e)}
    
    # PASO 2: Configuración del modelo
    print("\n2. CONFIGURANDO MODELO...")
    print("-" * 40)
    
    config = ModelConfig(
        area_km2=area_km2,
        dt_hours=24.0,
        route=True,
        debug_balance=True,
        validation_level='strict'
    )
    
    print(f"✓ Área cuenca: {config.area_km2} km²")
    print(f"✓ Paso temporal: {config.dt_hours} horas")
    print(f"✓ Nivel validación: {config.validation_level}")
    
    # PASO 3: División calibración/validación
    print("\n3. DIVIDIENDO DATOS CALIBRACIÓN/VALIDACIÓN...")
    print("-" * 40)
    
    split_fraction = 0.7
    split_idx = int(len(df) * split_fraction)
    
    df_calib = df.iloc[:split_idx].copy()
    df_valid = df.iloc[split_idx:].copy()
    
    # Convertir caudales observados a mm
    seconds_per_day = 24 * 3600
    m3s_to_mm_factor = seconds_per_day / (area_km2 * 1e6) * 1000
    
    q_obs_calib_mm = df_calib["Qobs_m3s"].values * m3s_to_mm_factor
    q_obs_valid_mm = df_valid["Qobs_m3s"].values * m3s_to_mm_factor
    
    print(f"✓ Calibración: {len(df_calib)} días ({df_calib.index[0]} - {df_calib.index[-1]})")
    print(f"✓ Validación: {len(df_valid)} días ({df_valid.index[0]} - {df_valid.index[-1]})")
    
    # PASO 4: Definir límites de parámetros para datos diarios
    print("\n4. DEFINIENDO LÍMITES DE PARÁMETROS...")
    print("-" * 40)
    
    bounds = {
        "S0_max": (40, 150),        # AUMENTADO (era 20-100)
        "alpha": (1.0, 1.8),        # REDUCIDO (era 1.0-2.5) 
        "k_qs": (0.15, 0.8),        # AUMENTADO (era 0.05-0.4)
        "k_inf": (0.03, 0.3),       # REDUCIDO (era 0.05-0.5)
        "k_perc": (0.01, 0.15),     # Mantener
        "phi": (0.3, 0.8),          # Mantener
        "k_qf": (0.15, 0.7),        # AUMENTADO (era 0.05-0.5)
        "k_bf": (0.01, 0.15),       # AUMENTADO (era 0.005-0.1)
        "f_et0": (0.02, 0.12),      # REDUCIDO (era 0.02-0.2)
        "f_et1": (0.005, 0.05),     # REDUCIDO (era 0.01-0.1)
        "n_r": (2, 3),              # AUMENTADO (era 1-4)
        "k_r": (60, 150)            # CRÍTICO: AUMENTADO (era 5-25)
    }
    
    print(f"✓ Parámetros a calibrar: {len(bounds)}")
    for param, (low, high) in bounds.items():
        print(f"  {param}: [{low}, {high}]")
    
    # PASO 5: Ejecutar pruebas unitarias
    print("\n5. EJECUTANDO PRUEBAS UNITARIAS...")
    print("-" * 40)
    
    try:
        test_model_components()
        print("✓ Todas las pruebas unitarias pasaron")
    except Exception as e:
        print(f"✗ Error en pruebas: {e}")
        return {"error": f"Pruebas fallaron: {e}"}
    
    # PASO 6: Calibración
    print("\n6. CALIBRANDO MODELO...")
    print("-" * 40)
    
    calibrator = ImprovedCalibrator(config, validation_level='strict')
    
    try:
        calibration_results = calibrator.calibrate_with_validation(
            df_calib, q_obs_calib_mm, bounds, n_iterations=200
        )
        
        best_params = calibration_results['best_params']
        best_score = calibration_results['best_score']
        
        print(f"✓ Calibración completada")
        print(f"✓ Mejor score: {best_score:.4f}")
        print(f"✓ Tasa éxito: {calibration_results['success_rate']:.2%}")
        
    except Exception as e:
        print(f"✗ Error en calibración: {e}")
        return {"error": f"Calibración falló: {e}"}
    
    # PASO 7: Simulación con parámetros óptimos
    print("\n7. SIMULANDO CON PARÁMETROS ÓPTIMOS...")
    print("-" * 40)
    
    model_final = TankModelValidated(best_params, config)
    
    try:
        # Simulación completa
        sim_complete, validation_complete = model_final.run_validated(df)
        
        # Extraer series para evaluación
        q_sim_calib_mm = sim_complete["Qout_mm"].iloc[:split_idx].values
        q_sim_valid_mm = sim_complete["Qout_mm"].iloc[split_idx:].values
        
        print("✓ Simulación completa exitosa")
        
        # Validar balance final
        balance_status = validation_complete['balance_validation']['global_status']
        print(f"✓ Balance global: {balance_status}")
        
    except Exception as e:
        print(f"✗ Error en simulación final: {e}")
        return {"error": f"Simulación final falló: {e}"}
    
    # PASO 8: Evaluación de desempeño
    print("\n8. EVALUANDO DESEMPEÑO...")
    print("-" * 40)
    
    # Métricas calibración
    metrics_calib = DailyPerformanceMetrics.evaluate_daily_performance(
        q_obs_calib_mm, q_sim_calib_mm
    )
    
    # Métricas validación  
    metrics_valid = DailyPerformanceMetrics.evaluate_daily_performance(
        q_obs_valid_mm, q_sim_valid_mm
    )
    
    # Métricas globales
    q_obs_full_mm = df["Qobs_m3s"].values * m3s_to_mm_factor
    q_sim_full_mm = sim_complete["Qout_mm"].values
    metrics_global = DailyPerformanceMetrics.evaluate_daily_performance(
        q_obs_full_mm, q_sim_full_mm
    )
    
    print("✓ Métricas calculadas:")
    print(f"  Calibración - NSE: {metrics_calib.get('NSE', 'N/A'):.3f}, "
          f"KGE: {metrics_calib.get('KGE', 'N/A'):.3f}")
    print(f"  Validación - NSE: {metrics_valid.get('NSE', 'N/A'):.3f}, "
          f"KGE: {metrics_valid.get('KGE', 'N/A'):.3f}")
    print(f"  Global - NSE: {metrics_global.get('NSE', 'N/A'):.3f}, "
          f"KGE: {metrics_global.get('KGE', 'N/A'):.3f}")
    
    # PASO 9: Exportar resultados
    print("\n9. EXPORTANDO RESULTADOS...")
    print("-" * 40)
    
    # Exportar simulación completa
    results_df = sim_complete.copy()
    results_df["Q_obs_m3s"] = df["Qobs_m3s"]
    results_df["P_mm"] = df["P_mm"] 
    results_df["PET_mm"] = df["PET_mm"]
    
    results_file = output_path / "simulacion_completa_validada.csv"
    results_df.to_csv(results_file)
    print(f"✓ Simulación exportada: {results_file}")
    
    # Exportar parámetros óptimos
    params_dict = {attr: getattr(best_params, attr) for attr in dir(best_params) 
                   if not attr.startswith('_') and not callable(getattr(best_params, attr))}
    
    params_file = output_path / "parametros_optimos_validados.json"
    with open(params_file, 'w') as f:
        json.dump(params_dict, f, indent=2)
    print(f"✓ Parámetros exportados: {params_file}")
    
    # Exportar métricas
    metrics_combined = {
        'calibracion': metrics_calib,
        'validacion': metrics_valid, 
        'global': metrics_global
    }
    
    metrics_file = output_path / "metricas_desempeno_validadas.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics_combined, f, indent=2, default=str)
    print(f"✓ Métricas exportadas: {metrics_file}")
    
    # Exportar validaciones
    validation_file = output_path / "reporte_validaciones.json"
    with open(validation_file, 'w') as f:
        json.dump(validation_complete, f, indent=2, default=str)
    print(f"✓ Validaciones exportadas: {validation_file}")
    
    # PASO 10: Resumen final
    print("\n" + "="*80)
    print("ANÁLISIS COMPLETADO EXITOSAMENTE")
    print("="*80)
    
    print(f"\nRESUMEN DE RESULTADOS:")
    print(f"• Período simulado: {len(df)} días")
    print(f"• Balance de masas: {balance_status}")
    print(f"• Desempeño calibración: {metrics_calib.get('Performance_Rating', 'N/A')}")
    print(f"• Desempeño validación: {metrics_valid.get('Performance_Rating', 'N/A')}")
    print(f"• NSE global: {metrics_global.get('NSE', 'N/A'):.3f}")
    print(f"• Archivos generados: {len(list(output_path.glob('*.csv'))) + len(list(output_path.glob('*.json')))}")
    
    # Retornar resultados completos
    return {
        'success': True,
        'simulation_results': results_df,
        'best_parameters': best_params,
        'performance_metrics': metrics_combined,
        'validation_report': validation_complete,
        'calibration_info': calibration_results,
        'output_directory': str(output_path)
    }

# ===============================================================================
# EJEMPLO DE USO
# ===============================================================================

if __name__ == "__main__":
    # Configurar ejemplo de uso
    ejemplo_data_file = "D:/OneDrive - Grupo EPM/Documentos/Python Scripts/Codigos_GITHUB/ModHD/data/Datos_prueba_2.csv"
    ejemplo_area = 40  # km²
    
    # Ejecutar análisis si el archivo existe
    if Path(ejemplo_data_file).exists():
        print("Ejecutando análisis de ejemplo...")
        resultados = run_complete_analysis(ejemplo_data_file, ejemplo_area)
        
        if resultados.get('success', False):
            print("\n¡Análisis de ejemplo completado exitosamente!")
        else:
            print(f"\nError en análisis de ejemplo: {resultados.get('error', 'Desconocido')}")
    else:
        print(f"Archivo de ejemplo no encontrado: {ejemplo_data_file}")
        print("Ejecutando solo pruebas unitarias...")
        test_model_components()