
from dataclasses import dataclass

@dataclass
class States:
    S0: float = 0.0  # mm - superficial
    S1: float = 0.0  # mm - zona no saturada
    S2: float = 0.0  # mm - subsuperficial rápido
    S3: float = 0.0  # mm - subsuperficial lento
