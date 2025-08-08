
# Tank Model (cuencas pequeñas) — Python

Modelo hidrológico modular tipo "tanques" (3 compartimentos + enrutamiento Nash) para series horarias o diarias.
- Unidades internas de **mm** (almacenamientos y flujos).
- Entrada mínima: fecha, **P_mm**, **PET_mm** (se puede poner PET_mm=0).
- Las columnas **P_mm** y **PET_mm** con valores `NaN` o negativos se reemplazan mediante interpolación lineal y se indica cuántos datos fueron corregidos.
- Salida: dataframe con caudales y particiones; conversión a m³/s vía `to_discharge()`.

Nuevas funcionalidades:
- Soporte ampliado de **PET** para conjuntos de datos con temperatura, radiación y métodos empíricos (p. ej. Hamon, Hargreaves).
- Módulos de calibración (`tank_model/calibration.py`) con búsqueda aleatoria y registro en `logs/`.
- Utilidades de **IO** (`tank_model/io.py`) para carga, partición y etiquetado de series.
- Interfaz gráfica (`scripts/gui_app.py`) basada en Tkinter/Matplotlib *(requiere `python3-tk`)*.
- Ejemplo de uso reproducible y calibración en `scripts/example_run.py`.

## Instalación

### Entorno Conda

Para crear y activar un entorno conda con las dependencias:

```bash
conda env create -f environment.yml
conda activate tank_model
```

Opcionalmente, instala el paquete en modo editable:

```bash
pip install -e .
```

## Estructura
```
tank_model/
  __init__.py
  parameters.py
  states.py
  model.py
  routing.py
  metrics.py
  calibration.py
scripts/
  example_run.py
data/
  example_forcing.csv
```

## Ecuaciones (discretas)
- Escorrentía superficial: \( Q_s = k_{qs} S_0^{\alpha} + \max(0, S_0 - S0_{max}) \)
- Infiltración: \( I = k_{inf} S_0 \)
- Evapotranspiración real: \( ET = \min(PET, f_{et0} S_0 + f_{et1} S_1) \)
- Pérdida por percolación: \( Perc = k_{perc} S_1^{\beta} \)
- Partición de percolación: fracción \(\phi\) a flujo sub-superficial rápido (tanque S2) y \(1-\phi\) al lento (S3).
- Descarga rápida: \( Q_f = k_{qf} S_2 \)
- Baseflow: \( Q_b = k_{bf} S_3 \)

Enrutamiento: **Nash** con parámetros `n_r` (entero ≥1) y `k_r` (horas o días, acorde a `dt_hours`).

## Supuestos
- Sin nieve/huelo.
- Homogeneidad lumped.
- Balance de masa explícito hacia adelante; paso `dt_hours` configurable.

## Uso rápido
```bash
python scripts/example_run.py
```
Este script carga los archivos de ejemplo ubicados en `data/`:

- `example_forcing.csv`: precipitación y evapotranspiración.
- `example_discharge.csv`: caudales observados.

Con ellos se ejecuta una simulación básica y se calibra el modelo con `calibration.random_search()`. Puedes reemplazar estos CSV por tus propias series para correr y calibrar el modelo con tus datos.


## PET — Cenicafé
[No verificado] Se incluye una forma empírica tipo Cenicafé: **PET = a · (Tmedia + b) · Rs**, con `a=0.0135` y `b=17.78` por defecto.
Ajusta `a`/`b` según calibración local y literatura específica para tu altitud y zona.

## GUI
La interfaz requiere [Tkinter](https://docs.python.org/3/library/tkinter.html), que no se instala vía `pip`.
Instálalo con el gestor de paquetes de tu sistema (p. ej. `sudo apt-get install python3-tk` en Debian/Ubuntu).

Ejecuta:
```bash
python scripts/gui_app.py
```
- Pestañas: (1) Datos & IO, (2) Parámetros, (3) Evapotranspiración, (4) Simulación, (5) Análisis seco/húmedo
- Gráfico integrado a la derecha que cambia según la pestaña activa.

## Notebook
`notebooks/calibracion_y_analisis.ipynb` con flujo reproducible: IO → PET → simulación → análisis seco/húmedo → métricas.
