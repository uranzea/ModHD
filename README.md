
# Tank Model (cuencas pequeñas) — Python

Modelo hidrológico modular tipo "tanques" (3 compartimentos + enrutamiento Nash) para series horarias o diarias.
- Unidades internas de **mm** (almacenamientos y flujos).
- Entrada mínima: fecha, **P_mm**, **PET_mm** (se puede poner PET_mm=0).
- Salida: dataframe con caudales y particiones; conversión a m³/s vía `to_discharge()`.

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
Edita parámetros en `example_run.py` o calibra con `calibration.random_search()`.


## PET — Cenicafé
[No verificado] Se incluye una forma empírica tipo Cenicafé: **PET = a · (Tmedia + b) · Rs**, con `a=0.0135` y `b=17.78` por defecto. 
Ajusta `a`/`b` según calibración local y literatura específica para tu altitud y zona. Si tienes la referencia oficial que usas en EPM, indícala y actualizo la fórmula exacta.

## GUI
Ejecuta:
```bash
python scripts/gui_app.py
```
- Pestañas: (1) Datos & IO, (2) Parámetros, (3) Evapotranspiración, (4) Simulación, (5) Análisis seco/húmedo
- Gráfico integrado a la derecha que se actualiza según la pestaña.

## Notebook
`notebooks/calibracion_y_analisis.ipynb` con flujo reproducible: IO → PET → simulación → análisis seco/húmedo → métricas.
