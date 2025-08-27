
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .parameters import Parameters
from .states import States
from .routing import nash_cascade
from typing import Union, Optional
from pathlib import Path

@dataclass
class ModelConfig:
    area_km2: float            # para conversión a m3/s
    dt_hours: float = 24.0     # 1 día por defecto
    route: bool = True         # aplicar Nash en la salida
    debug_balance: bool = False
    debug_csv_path: Optional[str] = None

    def __post_init__(self):
        if self.area_km2 == 1.0:
            raise ValueError(
                "Debe proporcionar el área real en km² al construir ModelConfig."
            )

class TankModel:
    def __init__(self, params: Parameters, config: ModelConfig, init_states: States = None):
        self.p = params
        self.c = config
        self.s = init_states if init_states is not None else States()

    def step(self, P, PET):
        p, s = self.p, self.s

        # Aportes y pérdidas en S0
        ET_pot = PET
        ET_cap = p.f_et0 * s.S0 + p.f_et1 * s.S1
        ET = min(ET_pot, ET_cap)

        S0_in = P
        S0 = s.S0 + S0_in - ET
        if S0 < 0:  # no permitir negativos
            ET += S0  # reducir ET si S0 insuficiente
            S0 = 0.0

        # Flujos desde S0
        Qs = p.k_qs * (S0 ** p.alpha)
        I = p.k_inf * S0

        # Exceso sobre capacidad S0
        excess = max(0.0, S0 - p.S0_max)
        Qs += excess

        # Verificar si hay agua suficiente en S0 para Qs + I
        total_out = Qs + I
        if total_out > S0:
            factor = S0 / total_out if total_out > 0 else 0.0
            Qs *= factor
            I *= factor
            S0 = 0.0
        else:
            S0 = S0 - total_out

        S0 = max(0.0, S0)

        # Tanque S1 (no saturado)
        S1 = s.S1 + I
        Perc = p.k_perc * (S1 ** p.beta)
        S1 = S1 - Perc
        if S1 < 0:
            Perc += S1
            S1 = 0.0
        # Partición de percolación
        to_S2 = p.phi * Perc
        to_S3 = (1 - p.phi) * Perc

        # Tanque S2 (rápido)
        S2 = s.S2 + to_S2
        Qf = p.k_qf * S2
        S2 = S2 - Qf
        if S2 < 0:
            Qf += S2
            S2 = 0.0

        # Tanque S3 (lento)
        S3 = s.S3 + to_S3
        Qb = p.k_bf * S3
        S3 = S3 - Qb
        if S3 < 0:
            Qb += S3
            S3 = 0.0

        # Actualizar estados
        self.s = States(S0=S0, S1=S1, S2=S2, S3=S3)

        # Caudal total en mm/dt antes de enrutamiento
        Qin = Qs + Qf + Qb
        return {
            "ET": ET, "I": I, "Perc": Perc, "Qs": Qs, "Qf": Qf, "Qb": Qb,
            "Qin": Qin, "S0": S0, "S1": S1, "S2": S2, "S3": S3
        }

    def run(self, df):
        """df con columnas: P_mm, PET_mm (index fecha opcional)"""
        cols = ["P_mm", "PET_mm"]
        for c in cols:
            if c not in df.columns:
                raise ValueError(f"Falta columna {c}")
        # Reemplazar valores NaN o negativos mediante interpolación
        invalid_mask = df[cols].isna() | (df[cols] < 0)
        invalid_count = int(invalid_mask.sum().sum())
        if invalid_count > 0:
            df[cols] = df[cols].mask(invalid_mask).interpolate(limit_direction="both")
            print(f"Se interpolaron {invalid_count} valores inválidos en P_mm/PET_mm.")
        out = []
        for i in range(len(df)):
            rec = self.step(df.iloc[i]["P_mm"], df.iloc[i]["PET_mm"])
            out.append(rec)
        out = pd.DataFrame(out, index=df.index)

        if self.c.route:
            seconds = self.c.dt_hours * 3600.0
            routed = nash_cascade(out["Qin"].values, n=self.p.n_r, k=self.p.k_r, dt=self.c.dt_hours)
            out["Qout_mm"] = routed
            out["Qraw_mm"] = out["Qin"].values
            out["Q_m3s"] = routed * self.c.area_km2 * 1e6 * 1e-3 / seconds
        else:
            out["Qout_mm"] = out["Qin"].values
            out["Q_m3s"] = np.nan

        # === DIAGNÓSTICO DE BALANCE HÍDRICO (opcional) ================================
        # Ubicar esto al final de TankModel.run(), antes de "return out"
        # Requiere que existan en "out" las columnas: "Qout_mm" y (si hay) "S0","S1","S2","S3".
        # El DataFrame de entrada "df" debe tener "P_mm" y "PET_mm".

        # 1) parámetros de depuración (desde config o argumentos)
        debug = getattr(self.c, "debug_balance", False)
        debug_csv_path = getattr(self.c, "debug_csv_path", None)
        # Si prefieres llamar run(df, debug=False, debug_csv_path=None), descomenta esto:
        # debug = debug or locals().get("debug", False)
        # debug_csv_path = debug_csv_path or locals().get("debug_csv_path", None)

        if debug:

            # 2) armar tabla de diagnóstico con lo que exista
            diag_cols = {}

            # Forzantes principales (si están en df)
            if "P_mm" in df.columns:
                diag_cols["P_mm"] = df["P_mm"].astype(float)
            if "PET_mm" in df.columns:
                diag_cols["PET_mm"] = df["PET_mm"].astype(float)

            # Salidas del modelo (en mm/∆t) y estados
            if "Qout_mm" in out.columns:
                diag_cols["Q_mm"] = out["Qout_mm"].astype(float)

            for s in ("S0","S1","S2","S3"):
                if s in out.columns:
                    diag_cols[s] = out[s].astype(float)

            # Componentes internos si existen (nombres opcionales/configurables)
            # Agrega aquí los nombres que use tu modelo para particiones:
            optional_terms = [
                "Qf_mm","Qb_mm","Qs_mm",       # flujos rápido, base, superficial
                "Infiltration_mm","Perc_mm",   # infiltración, percolación
                "ET_mm","ET0_mm","ET1_mm"      # evapotranspiración efectiva/por tanque
            ]
            for col in optional_terms:
                if col in out.columns:
                    diag_cols[col] = out[col].astype(float)

            diag = pd.DataFrame(diag_cols, index=out.index)

            # 3) cierre instantáneo y acumulado
            # Estados totales (si existen)
            stor_cols = [c for c in ("S0","S1","S2","S3") if c in diag.columns]
            S_total = diag[stor_cols].sum(axis=1) if stor_cols else pd.Series(0.0, index=diag.index)

            P   = diag.get("P_mm",  pd.Series(0.0, index=diag.index))
            PET = diag.get("PET_mm",pd.Series(0.0, index=diag.index))
            Q   = diag.get("Q_mm",  pd.Series(0.0, index=diag.index))

            # Cierre simple: P - PET - Q - ΔS
            dS   = S_total.diff().fillna(0.0)
            close_step = P - PET - Q - dS
            close_cum  = close_step.cumsum()

            diag["S_total_mm"]   = S_total
            diag["dS_mm"]        = dS
            diag["resid_step_mm"] = close_step
            diag["resid_cum_mm"]  = close_cum

            # 4) imprimir resumen
            sumP   = float(np.nansum(P.values))
            sumPET = float(np.nansum(PET.values))
            sumQ   = float(np.nansum(Q.values))
            dS_tot = float((S_total.iloc[-1] - S_total.iloc[0])) if len(S_total) else 0.0
            resid  = sumP - sumPET - sumQ - dS_tot

            print("\n=== DIAGNÓSTICO DE BALANCE (run) ===")
            print(f"ΣP     = {sumP:12.2f} mm")
            print(f"ΣPET   = {sumPET:12.2f} mm")
            print(f"ΣQsim  = {sumQ:12.2f} mm")
            if len(S_total):
                print(f"ΔS     = {dS_tot:12.2f} mm (Sini={float(S_total.iloc[0]):.2f} → Sfin={float(S_total.iloc[-1]):.2f})")
            else:
                print("ΔS     =     N/A (no hay columnas S0..S3 en la salida)")
            print(f"Residuo= {resid:12.2f} mm  (ideal ≈ 0)")

            # 5) (opcional) desgloses que existan
            for name in ("Qs_mm","Qf_mm","Qb_mm","ET_mm","Infiltration_mm","Perc_mm"):
                if name in diag.columns:
                    print(f"Σ{name:<16}= {float(np.nansum(diag[name].values)):12.2f} mm")

            # 6) guardar CSV de diagnóstico (opcional)
            if debug_csv_path:
                try:
                    Path(debug_csv_path).parent.mkdir(parents=True, exist_ok=True)
                    diag.to_csv(debug_csv_path, index=True)
                    print(f"[Debug] Diagnóstico de balance guardado en: {debug_csv_path}")
                except Exception as e:
                    print(f"[Debug] No se pudo guardar diagnóstico en {debug_csv_path}: {e}")

            # 7) opcional: anexar al DataFrame de salida para inspección externa
            # (descomenta si quieres devolver estas columnas)
            out["S_total_mm"]    = diag["S_total_mm"]
            out["resid_step_mm"] = diag["resid_step_mm"]
            out["resid_cum_mm"]  = diag["resid_cum_mm"]
        return out
