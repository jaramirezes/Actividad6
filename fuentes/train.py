import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def entrenar_y_registrar_modelos(v_X, v_y, v_X_train, v_y_train):
    """
    Configura el experimento en MLflow, entrena Logistic Regression y Random Forest
    usando validación cruzada de 5 pliegues (K=5) y registro de métricas y artefactos.
    """
    # Configurar el experimento en MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Pipeline_Diabetes_Pima_MLflow")

    # ==========================================
    # 1. MODELO 1: LOGISTIC REGRESSION (K=5)
    # ==========================================
    v_model_name_lr = "Logistic_Regression"

    with mlflow.start_run(run_name=v_model_name_lr):
        v_C_param = 1.0
        v_max_iter_param = 1000
        v_cv_folds = 5

        # Registrar parámetros
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C", v_C_param)
        mlflow.log_param("max_iter", v_max_iter_param)
        mlflow.log_param("cv_folds", v_cv_folds)

        # Definir modelo con prefijo v_
        v_clf = LogisticRegression(C=v_C_param, max_iter=v_max_iter_param, random_state=42)

        # Validación cruzada K=5 para AUC-ROC
        v_cv_scores_auc = cross_val_score(v_clf, v_X, v_y, cv=v_cv_folds, scoring='roc_auc')
        v_mean_auc = v_cv_scores_auc.mean()

        # Entrenar en train para artefacto
        v_clf.fit(v_X_train, v_y_train)

        # Registrar métricas en MLflow
        mlflow.log_metric("cv_roc_auc_mean", v_mean_auc)
        mlflow.log_metric("cv_roc_auc_std", v_cv_scores_auc.std())

        # Guardar artefacto
        mlflow.sklearn.log_model(v_clf, artifact_path="model")
        print(f"Logistic Regression - K=5 AUC-ROC Promedio: {v_mean_auc:.4f}")

    # ==========================================
    # 2. MODELO 2: RANDOM FOREST (K=5)
    # ==========================================
    v_model_name_rf = "Random_Forest"

    with mlflow.start_run(run_name=v_model_name_rf):
        v_n_estimators_param = 100
        v_max_depth_param = 10

        # Registrar parámetros
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", v_n_estimators_param)
        mlflow.log_param("max_depth", v_max_depth_param)
        mlflow.log_param("cv_folds", v_cv_folds)

        # Definir modelo con prefijo v_
        v_rf_clf = RandomForestClassifier(
            n_estimators=v_n_estimators_param,
            max_depth=v_max_depth_param,
            random_state=42
        )

        # Validación cruzada K=5 para AUC-ROC
        v_cv_scores_auc_rf = cross_val_score(v_rf_clf, v_X, v_y, cv=v_cv_folds, scoring='roc_auc')
        v_mean_auc_rf = v_cv_scores_auc_rf.mean()

        # Entrenar en train para artefacto
        v_rf_clf.fit(v_X_train, v_y_train)

        # Registrar métricas en MLflow
        mlflow.log_metric("cv_roc_auc_mean", v_mean_auc_rf)
        mlflow.log_metric("cv_roc_auc_std", v_cv_scores_auc_rf.std())

        # Guardar artefacto
        mlflow.sklearn.log_model(
            v_rf_clf,
            artifact_path="model",
            serialization_format="cloudpickle"
        )
        print(f"Random Forest - K=5 AUC-ROC Promedio: {v_mean_auc_rf:.4f}")

    print("\nModelos evaluados con K=5 Cross-Validation y registrados en MLflow exitosamente.")

    # Retorno con nomenclatura estricta para evitar errores de desempaquetado
    return v_clf, v_rf_clf
