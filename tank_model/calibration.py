
import numpy as np
import os
from datetime import datetime
from .parameters import Parameters
from .metrics import kge, nse
from typing import Union, Optional

def random_search(model_factory, df, q_obs, n_iter=200, seed=42, bounds=None,
                  catchment_name: Optional[str] = None, log_path: str = "calibration_log.csv"):
    """Búsqueda aleatoria simple maximizando NSE.

    Args:
        model_factory: función que crea un modelo dado un Parameters.
        df: DataFrame de forzantes.
        q_obs: serie de caudal observado para calcular la métrica.
        n_iter: número de iteraciones de búsqueda aleatoria.
        seed: semilla para reproducibilidad.
        bounds: diccionario con (min, max) por parámetro a calibrar.
        catchment_name: si se proporciona, los mejores parámetros se registrarán en un log.
        log_path: ruta al archivo CSV de log (ver :func:`log_calibration_results`).

    Returns:
        Tuple (best_params, best_score).
    """
    rng = np.random.default_rng(seed)
    if bounds is None:
        # bounds = {
        #     "k_qs": (0.01, 0.5),
        #     "alpha": (1.0, 2.5),
        #     "S0_max": (5, 150),
        #     "k_inf": (0.01, 0.5),
        #     "k_perc": (0.005, 0.2),
        #     "beta": (0.8, 1.5),
        #     "phi": (0.2, 0.9),
        #     "k_qf": (0.05, 0.8),
        #     "k_bf": (0.001, 0.2),
        #     "n_r": (1, 4),
        #     "k_r": (1.0, 72.0),
        #     "f_et0": (0.0, 0.2),
        #     "f_et1": (0.0, 0.1),
        # }
        bounds = {
            "S0_max": (25, 80),
            "alpha":  (1.30, 2.00),
            "k_qs":   (0.12, 0.30),
            "k_inf":  (0.15, 0.35),
            "k_perc": (0.03, 0.10),
            "phi":    (0.55, 0.80),
            "k_qf":   (0.15, 0.35),
            "k_bf":   (0.04, 0.15),
            "f_et0":  (0.08, 0.16),
            "f_et1":  (0.03, 0.08),
            "n_r":    (1, 2),
            "k_r":    (6.0, 18.0),  # horas
        }
    best_score = -np.inf
    best_p = None

    keys = list(bounds.keys())
    for _ in range(n_iter):
        p = Parameters()
        for k in keys:
            lo, hi = bounds[k]
            if k == "n_r":
                setattr(p, k, int(rng.integers(lo, hi+1)))
            else:
                setattr(p, k, rng.uniform(lo, hi))
        m = model_factory(p)
        sim = m.run(df)["Q_m3s"].values
        # score = kge(q_obs, sim)

        # dentro del loop de calibración
        q_sim = sim["Q_m3s"].values
        nse_lin = nse(q_obs, q_sim)
        nse_log = nse(np.log1p(q_obs), np.log1p(q_sim))  # sensibilidad a bajos-medios
        # error de picos (top 1% observado)
        thr = np.nanpercentile(q_obs, 99)
        peak_bias = (np.nansum(q_sim[q_obs>=thr]) - np.nansum(q_obs[q_obs>=thr])) / (np.nansum(q_obs[q_obs>=thr]) + 1e-9)
        peak_score = -abs(peak_bias)  # 0 es perfecto, penaliza desbalance de picos
        score = 0.45*nse_lin + 0.35*nse_log + 0.20*peak_score

        if np.isfinite(score) and score > best_score:
            best_score = score
            best_p = p

    # Guardar log si se indica la cuenca
    if catchment_name is not None and best_p is not None:
        try:
            log_calibration_results(best_p, best_score, catchment_name, log_path=log_path)
        except Exception as e:
            # No interrumpir calibración por fallo en logging; solo informar en consola
            print(f"Advertencia: no se pudo guardar log de calibración: {e}")
    return best_p, best_score

def log_calibration_results(best_params: Parameters, best_score: float, catchment_name: str,
                             log_path: str = "calibration_log.csv") -> None:
    """Guarda en un registro los parámetros calibrados y el KGE obtenido.

    Cada llamada añade una línea al archivo `log_path`. Se incluye la fecha y hora
    (en formato ISO 8601), el nombre de la cuenca y todos los parámetros del modelo.

    Args:
        best_params: instancia de Parameters con los valores calibrados.
        best_score: valor de KGE u otra métrica a registrar.
        catchment_name: identificador de la cuenca o estación para diferenciar el registro.
        log_path: ruta del archivo CSV de log. Si no existe, se crea con encabezados.
    """
    # Preparar datos a escribir
    timestamp = datetime.now().isoformat(timespec='seconds')
    param_dict = vars(best_params)
    # Asegurar directorio
    dir_name = os.path.dirname(log_path)
    if dir_name and not os.path.isdir(dir_name):
        os.makedirs(dir_name, exist_ok=True)
    # Preparar fila en orden: fecha, cuenca, score y parámetros
    # Determinar el próximo ID consecutivo
    next_id = 0
    if os.path.exists(log_path):
        try:
            import csv as _csv
            with open(log_path, newline="", encoding="utf-8") as f:
                reader = _csv.DictReader(f)
                # Buscar la mayor ID existente; si no hay columna 'id', contar filas
                ids = []
                for i, r in enumerate(reader):
                    if 'id' in r and r['id'] != '':
                        try:
                            ids.append(int(float(r['id'])))
                        except Exception:
                            pass
                    else:
                        ids.append(i)
                if ids:
                    next_id = max(ids) + 1
        except Exception:
            # en caso de error, simplemente utilizar el número de líneas existentes como ID
            with open(log_path, 'r', encoding='utf-8') as f:
                next_id = sum(1 for _ in f) - 1  # descontar encabezado
    # Construir fila con ID y resto de campos
    row = {
        "id": next_id,
        "timestamp": timestamp,
        "catchment": catchment_name,
        "score": best_score,
    }
    row.update(param_dict)
    # Asegurar que el encabezado tenga 'id' al inicio
    fieldnames = list(row.keys())
    # Escribir en modo append
    import csv
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_calibration(catchment_name: str,
                     calibration_id: Union[int, str, None] = None,
                     log_path: str = "calibration_log.csv") -> Parameters:
    """Carga una calibración previamente registrada en el log y devuelve un objeto Parameters.

    Se buscan las entradas del log correspondientes a la cuenca `catchment_name`.  Si
    `calibration_id` es ``None`` se selecciona la calibración más reciente (según
    la columna ``timestamp``).  Si es un entero, se selecciona la fila por su
    posición (0 = primera calibración registrada para esa cuenca).  Si es una
    cadena, se interpreta como un valor de la columna ``timestamp``.

    Args:
        catchment_name: nombre de la cuenca registrada en el log.
        calibration_id: índice entero o marca de tiempo (cadena ISO) de la calibración a cargar.  Por defecto se usa la última calibración.
        log_path: ruta del archivo CSV de log.

    Returns:
        Instancia de Parameters con los valores almacenados.

    Raises:
        FileNotFoundError: si el archivo de log no existe.
        ValueError: si no hay registros para la cuenca o no se encuentra el ID solicitado.
    """
    import csv
    import pandas as pd  # usado para ordenar por fecha
    
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"No se encontró el archivo de log: {log_path}")
    # Leer log completo
    with open(log_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    # Filtrar por cuenca
    entries = [row for row in rows if row.get("catchment") == catchment_name]
    if not entries:
        raise ValueError(f"No hay calibraciones registradas para la cuenca: {catchment_name}")
    # Seleccionar calibración
    selected = None
    # Convertir a DataFrame para ordenar
    try:
        df_entries = pd.DataFrame(entries)
        # Asegurar que 'id' sea numérico si existe
        if 'id' in df_entries.columns:
            df_entries['id_num'] = pd.to_numeric(df_entries['id'], errors='coerce')
        else:
            df_entries['id_num'] = pd.Series(range(len(df_entries)))
        # Convertir timestamp para ordenar por fecha
        df_entries['timestamp_dt'] = pd.to_datetime(df_entries['timestamp'], errors='coerce')
        # Ordenar por id_num asc (antiguos a recientes) si id existe, en su defecto por fecha
        df_entries = df_entries.sort_values(['id_num', 'timestamp_dt'])
        entries = df_entries.to_dict(orient='records')
    except Exception:
        pass
    if calibration_id is None:
        # Seleccionar la última calibración (id mayor o fecha más reciente)
        selected = entries[-1]
    else:
        selected = None
        # Si se proporciona un entero, intentar coincidir con la columna 'id'
        if isinstance(calibration_id, int):
            # Buscar por id
            for row in entries:
                try:
                    if 'id' in row and int(float(row['id'])) == calibration_id:
                        selected = row
                        break
                except Exception:
                    pass
            # Si no se encontró por 'id', tratar como índice posicional
            if selected is None:
                if calibration_id < 0 or calibration_id >= len(entries):
                    raise ValueError(f"Índice de calibración fuera de rango: {calibration_id}")
                selected = entries[calibration_id]
        else:
            # Buscar por marca de tiempo exacta
            for row in entries:
                if row.get('timestamp') == str(calibration_id):
                    selected = row
                    break
            if selected is None:
                raise ValueError(f"No se encontró calibración con timestamp {calibration_id}")
    # Crear Parameters
    param_fields = [field for field in vars(Parameters()).keys()]
    params_kwargs = {}
    for k in param_fields:
        if k in selected:
            try:
                # convert string to float or int
                if k == 'n_r':
                    params_kwargs[k] = int(float(selected[k]))
                else:
                    params_kwargs[k] = float(selected[k])
            except Exception:
                # si no se puede convertir, usar valor por defecto
                pass
    return Parameters(**params_kwargs)


def run_with_saved_calibration(model_factory,
                               df,
                               catchment_name: str,
                               calibration_id: Optional[Union[int, str]] = None,
                               log_path: str = "calibration_log.csv"):
    """Ejecuta el modelo con una calibración previamente guardada.

    Carga los parámetros de calibración del archivo de log y genera la simulación
    para el DataFrame de forzantes ``df``.  Si `calibration_id` es ``None`` se
    utiliza la última calibración registrada para la cuenca.

    Args:
        model_factory: función que acepta un objeto Parameters y devuelve una instancia de TankModel.
        df: DataFrame con columnas de forzantes (P_mm, PET_mm).
        catchment_name: nombre de la cuenca para seleccionar la calibración.
        calibration_id: índice entero o timestamp de la calibración a cargar (opcional).
        log_path: ruta del archivo de log.

    Returns:
        DataFrame con la simulación generada por el modelo calibrado.
    """
    params = load_calibration(catchment_name, calibration_id=calibration_id, log_path=log_path)
    model = model_factory(params)
    return model.run(df)


def list_recent_calibrations(catchment_name: str, n: int = 5, log_path: str = "calibration_log.csv"):
    """Devuelve las últimas *n* calibraciones registradas para la cuenca indicada.

    Se lee el archivo de log y se filtran las filas pertenecientes a la cuenca
    `catchment_name`, ordenadas por fecha descendente (o por ID si existe).  Se
    devuelven como una lista de diccionarios con todos los campos almacenados.

    Args:
        catchment_name: nombre de la cuenca para filtrar.
        n: número de calibraciones recientes a devolver.  Por defecto 5.
        log_path: ruta del archivo CSV de log.

    Returns:
        Lista de hasta `n` entradas de calibración (cada una como `dict`). Si no
        hay calibraciones para la cuenca se devuelve una lista vacía.
    """
    import csv
    import pandas as pd
    if not os.path.exists(log_path):
        return []
    with open(log_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if row.get('catchment') == catchment_name]
    if not rows:
        return []
    # Convertir a DataFrame para ordenar
    df_rows = pd.DataFrame(rows)
    # Si existe 'id', usarlo para ordenar; si no, crear orden por timestamp
    if 'id' in df_rows.columns:
        # Convertir a numérico; los NaN se colocan al final
        df_rows['id_num'] = pd.to_numeric(df_rows['id'], errors='coerce')
        df_rows = df_rows.sort_values('id_num', ascending=False)
    else:
        # Convertir timestamp y ordenar por fecha descendente
        df_rows['timestamp_dt'] = pd.to_datetime(df_rows['timestamp'], errors='coerce')
        df_rows = df_rows.sort_values('timestamp_dt', ascending=False)
    # Seleccionar las n primeras
    df_recent = df_rows.head(n)
    return df_recent.to_dict(orient='records')