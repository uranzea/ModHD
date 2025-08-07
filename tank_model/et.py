
"""Módulos de evapotranspiración (PET)."""
import numpy as np
import pandas as pd

def et_cenicafe(Tmean_C, Rs_MJ_m2_d, a=0.0135, b=17.78):
    """Calcula PET (mm/día) con la ecuación empírica tipo Cenicafé.
    
    PET = a * (Tmean + b) * Rs
    
    Donde:
    - Tmean_C: temperatura media (°C)
    - Rs_MJ_m2_d: radiación solar global (MJ/m²/día)
    - a, b: coeficientes empíricos (ajustables por región/altitud)
    
    Nota: 1 MJ/m² ≈ 0.408 mm de equivalente evaporativo si se usa FAO56.
    Esta forma empírica usa un factor a ya "ajustado"; PET queda en mm/día.
    
    Devuelve:
    - PET en mm/día (numpy array / pandas Series)
    """
    T = np.asarray(Tmean_C, dtype=float)
    Rs = np.asarray(Rs_MJ_m2_d, dtype=float)
    PET = a * (T + b) * Rs
    PET = np.clip(PET, 0.0, None)
    return PET

def et_hargreaves(Tmean_C, Tmax_C, Tmin_C, Ra_MJ_m2_d):
    """Hargreaves-Samani (mm/día) — referencia FAO-56.
    PET = 0.0023 * Ra * (Tmean + 17.8) * sqrt(Tmax - Tmin)
    Ra en MJ/m²/día.
    """
    Tmean = np.asarray(Tmean_C, dtype=float)
    Ra = np.asarray(Ra_MJ_m2_d, dtype=float)
    dT = np.maximum(0.0, np.asarray(Tmax_C) - np.asarray(Tmin_C))
    PET = 0.0023 * Ra * (Tmean + 17.8) * np.sqrt(dT)
    PET = np.clip(PET, 0.0, None)
    return PET

def ensure_pet(df, method="column", **kwargs):
    """Asegura que el DataFrame tenga una columna PET_mm.
    method:
      - "column": usa df['PET_mm'] (lanza si no existe)
      - "cenicafe": requiere 'Tmean_C' y 'Rs_MJ_m2_d'
      - "hargreaves": requiere 'Tmean_C','Tmax_C','Tmin_C','Ra_MJ_m2_d'
    """
    if method == "column":
        if "PET_mm" not in df.columns:
            raise ValueError("Falta columna PET_mm y method='column'")
        return df
    out = df.copy()
    if method == "cenicafe":
        for c in ["Tmean_C","Rs_MJ_m2_d"]:
            if c not in out.columns:
                raise ValueError(f"Falta columna {c} para Cenicafé")
        out["PET_mm"] = et_cenicafe(out["Tmean_C"], out["Rs_MJ_m2_d"], 
                                     a=kwargs.get("a",0.0135), b=kwargs.get("b",17.78))
        return out
    if method == "hargreaves":
        req = ["Tmean_C","Tmax_C","Tmin_C","Ra_MJ_m2_d"]
        for c in req:
            if c not in out.columns:
                raise ValueError(f"Falta columna {c} para Hargreaves")
        out["PET_mm"] = et_hargreaves(out["Tmean_C"], out["Tmax_C"], out["Tmin_C"], out["Ra_MJ_m2_d"])
        return out
    raise ValueError(f"method no soportado: {method}")
