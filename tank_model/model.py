
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .parameters import Parameters
from .states import States
from .routing import nash_cascade

@dataclass
class ModelConfig:
    dt_hours: float = 24.0     # 1 día por defecto
    area_km2: float = 1.0      # para conversión a m3/s (opcional)
    route: bool = True         # aplicar Nash en la salida

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
        return out
