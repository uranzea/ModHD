
import numpy as np

def nash_cascade(
    inflow_mm,
    n=2,
    k_hours=None,
    dt_hours=None,
    # --- alias legacy para compatibilidad con model.py ---
    k=None,
    dt=None,
    # --- opciones de cola (por defecto desactivadas para no cambiar longitudes) ---
    flush_tail=False,
    tail_mode="fixed",      # "fixed" o "relative"
    tail_factor=15,         # usado si tail_mode="fixed": pasos ≈ tail_factor * (k/dt)
    allow_nan=False,
    verbose=False,
):
    """
    Enrutamiento Nash discreto (mm/∆t) con compatibilidad de firmas:
    - model.py puede llamar con (k=..., dt=...)
    - nuevo código puede llamar con (k_hours=..., dt_hours=...)

    Por defecto, flush_tail=False para NO cambiar la longitud de salida (comportamiento
    esperado por la mayoría de pipelines). Actívalo sólo si quieres drenar la cola.
    """
    x = np.asarray(inflow_mm, dtype=float)
    if not allow_nan:
        x = np.nan_to_num(x, nan=0.0)

    # Resolver aliases
    if k_hours is None:
        k_hours = float(k) if k is not None else None
    if dt_hours is None:
        dt_hours = float(dt) if dt is not None else None
    if k_hours is None or dt_hours is None:
        raise ValueError("Debes especificar k_hours y dt_hours (o sus alias k y dt).")
    if k_hours <= 0:
        raise ValueError("k_hours debe ser > 0.")
    n = int(n)
    if n < 1:
        return x.copy()

    # Coeficiente discreto
    c = float(np.exp(-dt_hours / k_hours))
    if c < 0.0: c = 0.0
    if c > 1.0: c = 1.0
    one_minus_c = 1.0 - c

    # --- ruteo base sobre la longitud original ---
    out = x.copy()
    for _ in range(n):
        y_prev = 0.0
        routed = np.zeros_like(out)
        for t in range(len(out)):
            y_prev = c * y_prev + one_minus_c * out[t]
            routed[t] = y_prev
        out = routed

    if not flush_tail:
        # Conservación en ventana (no incluye cola)
        if verbose:
            sum_in  = float(np.nansum(x))
            sum_out = float(np.nansum(out))
            print(f"[Nash] Σin={sum_in:.6f} mm, Σout={sum_out:.6f} mm, diff={sum_out - sum_in:.6e} (sin cola)")
        return out

    # ---------- drenaje de cola con número de pasos determinista ----------
    if tail_mode == "fixed":
        tail_steps = int(np.ceil(tail_factor * (k_hours / dt_hours)))
    else:  # "relative": pasos suficientes para una masa acumulada ~ 1 - 1e-3
        if c >= 1.0:
            tail_steps = 0
        else:
            # m >= ln(1-0.999)/ln(c)  -> masa ~ 99.9%
            tail_steps = int(np.ceil(np.log(1.0 - 0.999) / np.log(c)))
            tail_steps = max(tail_steps, 1)

    # Reconstruir estados finales de cada reservorio
    states = []
    tmp = x.copy()
    for _ in range(n):
        y_prev = 0.0
        for t in range(len(tmp)):
            y_prev = c * y_prev + one_minus_c * tmp[t]
        states.append(y_prev)
        # reconstruir la salida para alimentar el siguiente eslabón
        y_prev = 0.0
        routed = np.zeros_like(tmp)
        for t in range(len(tmp)):
            y_prev = c * y_prev + one_minus_c * tmp[t]
            routed[t] = y_prev
        tmp = routed

    tail = []
    for _ in range(tail_steps):
        y_in = 0.0
        new_states = []
        for s in states:
            y_out = c * s + one_minus_c * y_in
            new_states.append(y_out)
            y_in = y_out
        states = new_states
        tail.append(y_in)

    out_full = np.concatenate([out, np.array(tail, float)])

    if verbose:
        sum_in  = float(np.nansum(x))
        sum_out = float(np.nansum(out))
        sum_out_full = float(np.nansum(out_full))
        print(f"[Nash] Σin={sum_in:.6f} mm, Σout={sum_out:.6f} mm, diff={sum_out - sum_in:.6e} (antes de cola)")
        print(f"[Nash] Cola añadida: {len(tail)} pasos")
        print(f"[Nash] Σout_final={sum_out_full:.6f} mm, diff_final={sum_out_full - sum_in:.6e} (≈0 esperado)")

    return out_full


def to_discharge(q_mm, area_km2, seconds_per_dt=None, dt_hours=None):
    """
    Convierte mm/∆t -> m³/s. Compatibilidad:
    - Original: to_discharge(q_mm, area_km2, seconds_per_dt=...)
    - Nuevo:    to_discharge(q_mm, area_km2, dt_hours=...)
    """
    q_mm = np.asarray(q_mm, dtype=float)
    if seconds_per_dt is None:
        if dt_hours is None:
            raise ValueError("Especifica seconds_per_dt o dt_hours.")
        seconds_per_dt = float(dt_hours) * 3600.0
    return q_mm * (area_km2 * 1e6) * 1e-3 / float(seconds_per_dt)


def from_discharge(q_m3s, area_km2, seconds_per_dt=None, dt_hours=None):
    """
    Convierte m³/s -> mm/∆t (útil para chequeos).
    """
    q_m3s = np.asarray(q_m3s, dtype=float)
    if seconds_per_dt is None:
        if dt_hours is None:
            raise ValueError("Especifica seconds_per_dt o dt_hours.")
        seconds_per_dt = float(dt_hours) * 3600.0
    return q_m3s * float(seconds_per_dt) / (area_km2 * 1e6) * 1e3



""" # --------- Ejemplo de uso y chequeos (puedes borrar esto en producción) ---------
# Parámetros de prueba
area_km2  = 0.173   # área de tu cuenca
dt_hours  = 24.0    # paso de tiempo (horas)
n         = 3       # número de reservorios
k_hours   = 48.0    # tiempo de residencia por reservorio (horas)

# 1) Pulso de entrada en mm/dt (10 mm en el tercer paso)
inflow_mm = np.array([0, 0, 10.0, 0, 0, 0, 0, 0, 0, 0], dtype=float)

# 2) Ruteo en mm/∆t
out_mm = nash_cascade(inflow_mm, n=n, k_hours=k_hours, dt_hours=dt_hours,
                      flush_tail=True, verbose=True)

# 3) Conversión a m³/s
out_m3s = to_discharge(out_mm, area_km2, dt_hours=dt_hours)
inflow_m3s = to_discharge(inflow_mm, area_km2, dt_hours=dt_hours)

# 4) Graficar resultados
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.stem(range(len(inflow_mm)), inflow_mm, basefmt=" ", linefmt="C0-", markerfmt="C0o", label="Inflow mm")
plt.plot(out_mm, "C1-", marker="o", label="Outflow mm")
plt.title(f"Nash cascade: n={n}, k={k_hours} h, dt={dt_hours} h")
plt.xlabel("Paso de tiempo")
plt.ylabel("Escorrentía (mm/∆t)")
plt.legend()
plt.grid(True)
plt.show()

# 5) Chequeos
print("\n=== CHEQUEOS ===")
print(f"Suma entrada (mm): {np.sum(inflow_mm):.3f}")
print(f"Suma salida  (mm): {np.sum(out_mm):.3f}")
print(f"Diferencia        : {np.sum(out_mm) - np.sum(inflow_mm):.3e} (≈0)")

# Consistencia unidades
out_m3s_direct = nash_cascade(inflow_m3s, n=n, k_hours=k_hours, dt_hours=dt_hours,
                              flush_tail=True)
diff = np.mean(np.abs(out_m3s - out_m3s_direct))
print(f"Diferencia media entre ruteo mm→m3/s y directo en m3/s: {diff:.3e} m³/s (≈0 esperado)") """