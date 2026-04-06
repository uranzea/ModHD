# Scripts de ModHD

Esta carpeta reúne los puntos de entrada vigentes para ejecutar, diagnosticar y explorar el modelo.

## Datos

- Insumos: `data/input/`
- Productos generados: `data/output/`
- Logs de calibración: `logs/`

## Scripts activos

- `launch_dashboard.py`
  Abre el dashboard local de ModHD con tabs para series, calibración, imágenes generadas y mapa/catálogo de capas.

- `run_calibration_validation.py`
  Ejecuta un flujo completo de lectura de datos, calibración, simulación, validación y guardado de resultados.

- `diagnose_calibration_validation.py`
  Corre un desk-check detallado para revisar resolución temporal, conversiones de unidades, balance hídrico y métricas.

- `launch_gui.py`
  Alias de compatibilidad que abre el dashboard actual.

## Legacy

La subcarpeta `legacy/` conserva pruebas exploratorias, notebooks y versiones integradas antiguas que ya no forman parte del flujo principal. Se mantienen como referencia, pero no deberían usarse como punto de entrada operativo.
