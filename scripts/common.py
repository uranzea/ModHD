from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
LOGS_DIR = ROOT / "logs"
RESULTS_DIR = OUTPUT_DIR
DEFAULT_FORCING_PATH = INPUT_DIR / "Datos_prueba_2.csv"


def configure_matplotlib_backend() -> None:
    """Use a non-interactive backend unless plots were explicitly requested."""
    if os.environ.get("MODHD_SHOW_PLOTS", "0") != "1":
        import matplotlib

        matplotlib.use("Agg")


def ensure_output_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def infer_dt_hours(index: pd.Index, default: float = 24.0) -> float:
    if len(index) <= 1:
        return default
    diffs = pd.Series(index).diff().iloc[1:]
    median_delta = pd.to_timedelta(diffs.median())
    dt_hours = median_delta.total_seconds() / 3600.0
    return dt_hours if dt_hours > 0 else default


def load_default_forcing(path: Path = DEFAULT_FORCING_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de forcing: {path}")
    df = pd.read_csv(path, parse_dates=["date"], index_col="date").sort_index()
    for column in ("P_mm", "PET_mm"):
        if column not in df.columns:
            raise ValueError(f"Falta columna requerida '{column}' en {path.name}")
    return df


def save_figure(fig, output_path: Path, *, dpi: int = 300) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"[Plot] Guardado en: {output_path}")
    if os.environ.get("MODHD_SHOW_PLOTS", "0") == "1":
        import matplotlib.pyplot as plt

        plt.show()
    import matplotlib.pyplot as plt

    plt.close(fig)
