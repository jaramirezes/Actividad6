# Changelog

Todos los cambios notables en este proyecto serán documentados en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/) y este proyecto adhiere a [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [[1.1.0]] - 2026-10-25

### Actualizado
* **`datos_prep.py`**: Implementación completa del pipeline de preprocesamiento y limpieza de datos.
* **`train.py`**: Integración de la lógica de entrenamiento, validación con Scikit-Learn y seguimiento de experimentos con MLflow.
* **`entrena.ipynb`**: Notebook orquestador para la ejecución interactiva del pipeline completo, visualización de métricas y comparación de resultados.

---

## [[1.0.0]] - 2026-10-25

### Agregado
* Estructura de directorios bajo estándares MLOps (`datos` y `fuentes/`).
* Integración del dataset original y procesado en `datos/datos_ini/`.
* Integración del dataset procesado en `datos/datos_limp/`.
* Script `datos_prep.py` para manejo de valores monetarios, nulos y codificación.
* Script `train.py` con Pipeline de Scikit-Learn, comparación entre Logistic Regression vs. Random Forest.
* Notebook `entrena.ipynb` para ejecución interactiva.