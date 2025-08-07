
from dataclasses import dataclass

@dataclass
class Parameters:
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
