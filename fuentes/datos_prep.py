import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from google.colab import files

def cargar_y_preprocesar_datos(target_col="Outcome", test_size=0.2, random_state=42):
    print("Por favor, selecciona el dataset Pima Indians Diabetes a procesar (CSV):")
    v_uploaded = files.upload()

    v_file_name = list(v_uploaded.keys())[0]
    v_df = pd.read_csv(v_file_name)
    print(f"\nDataset '{v_file_name}' cargado exitosamente.")

    # 1. Regla de Negocio / Limpieza Específica Pima (Ceros en variables clínicas)
    v_columnas_con_ceros_invalidos = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    for v_col in v_columnas_con_ceros_invalidos:
        if v_col in v_df.columns:
            # Reemplazar 0s por NaN para evitar sesgos médicos
            v_df[v_col] = v_df[v_col].replace(0, np.nan)
            # Imputación robusta con la mediana
            v_mediana_val = v_df[v_col].median()
            v_df[v_col] = v_df[v_col].fillna(v_mediana_val)
            print(f"Columna '{v_col}': 0s reemplazados por NaN e imputados con mediana ({v_mediana_val:.2f})")

    # 2. Definir target y separar X / y
    if target_col in v_df.columns:
        v_X = v_df.drop(columns=[target_col])
        v_y = v_df[target_col]
    else:
        v_X = v_df.iloc[:, :-1]
        v_y = v_df.iloc[:, -1]

    # 3. Escalado de datos (StandardScaler)
    v_scaler = StandardScaler()
    v_X_scaled = v_scaler.fit_transform(v_X)
    v_X = pd.DataFrame(v_X_scaled, columns=v_X.columns)

    # Guardar y descargar copia limpia en Colab
    v_df_limpio_exportar = v_X.copy()
    v_df_limpio_exportar[target_col] = v_y.values

    v_nombre_archivo_salida = "pima_diabetes_datos_limpios.csv"
    v_df_limpio_exportar.to_csv(v_nombre_archivo_salida, index=False)
    print(f"\n[INFO] Archivo limpio guardado. Descargando automáticamente...")
    files.download(v_nombre_archivo_salida)

    # 4. División en Train / Test (con estratificación para mantener la proporción de clases)
    v_X_train, v_X_test, v_y_train, v_y_test = train_test_split(
        v_X, v_y, test_size=test_size, random_state=random_state, stratify=v_y
    )

    print("\n¡Datos preprocesados, escalados y divididos exitosamente!")
    return v_X, v_y, v_X_train, v_X_test, v_y_train, v_y_test
