# Actividad 6: Pipeline para comparar Modelos

Repositorio para la implementación de un pipeline de comparación entre Regresión Logistica (A) y Random Forest (B)

## Estructura del Repositorio
- `datos/datos_ini/`: Contiene el dataset original (`diabetes.csv`).
- `datos/datos_limp/`: Contiene el dataset procesado y limpio.
- `fuentes/`: 
  - `entrena.ipynb`: Notebook de desarrollo y evaluación.
  - `datos_prep.py`: Funciones de limpieza e imputación.
  - `train.py`: Script principal de entrenamiento, optimización y registro en MLflow.
 

## 🚀 Guía de Ejecución en Google Colab

Para ejecutar el proceso correctamente, debes seguir los pasos de manera consecutiva dentro de **Google Colab**:

### 📥 Requisito Previo: Carga del Dataset
Descarga el archivo `diabetes.csv` en tu computadora para utilizarlo durante la ejecución del notebook.

### 📋 Pasos para la Ejecución

1. **Abrir el Notebook:**
   Abre el script `entrena.ipynb` en Google Colab.

2. **Ejecución Secuencial:**
   Ejecuta las celdas de código de manera consecutiva de principio a fin.

3. **Carga de Datos:**
   Al llegar a la celda **`b) Carga y Preprocesamiento llamando a datos_prep.py`**, el sistema solicitará seleccionar el archivo de entrada.
   * Aparecerá el mensaje: *"Por favor, selecciona el dataset a procesar"*.
   * Haz clic en el botón **`Elegir archivos`** (Choose Files).
   * Selecciona el archivo `diabetes.csv` previamente descargado en tu equipo.
   
   > ⚠️ **Nota importante:** Si no cargas el archivo, el proceso permanecerá detenido esperando la selección.

4. **Finalización del Flujo:**
   Continúa ejecutando las celdas restantes hasta la última.

5. **Resultados y Métricas:**
   Al finalizar, el notebook generará las métricas de evaluación de los modelos y las gráficas comparativas.

6. **Rastreo con MLflow:**
   Abre la liga generada al final para acceder a la interfaz de **MLflow** y revisar el registro de los experimentos.
