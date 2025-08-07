
import numpy as np

def nash_cascade(inflow, n=2, k=12.0, dt=1.0):
    """Enruta un hidrograma (mm/dt) con un Nash cascade.
    inflow: array (mm por dt)
    n: entero >=1
    k: tiempo de residencia (mismo tiempo que dt en unidades)
    dt: tamaño del paso (horas o días). Si k y dt comparten unidad, ok.
    """
    inflow = np.asarray(inflow, dtype=float)
    if n < 1:
        return inflow.copy()
    # Coeficientes lineales equivalentes (discretización exponencial)
    out = inflow.copy()
    for _ in range(n):
        y = 0.0
        routed = np.zeros_like(out)
        c = np.exp(-dt / k) if k > 1e-9 else 0.0
        for t in range(len(out)):
            y = c * y + (1 - c) * out[t]
            routed[t] = y
        out = routed
    return out

def to_discharge(q_mm, area_km2, seconds_per_dt):
    """Convierte de mm/dt a m3/s.

    q_mm: array (mm por dt)

    area_km2: área de cuenca

    seconds_per_dt: segundos en el dt

    """
    q_mm = np.asarray(q_mm, dtype=float)
    return q_mm * area_km2 * 1e6 * 1e-3 / seconds_per_dt
