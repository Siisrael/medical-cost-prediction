import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor



def main():


    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "medical_data")
    DB_USER = os.getenv("DB_USER", "sisrael")
    DB_PASS = os.getenv("DB_PASS", "pass123")

    # Creo el engine que después uso para leer/escribir tablas.

    engine = create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    REPO_ROOT = Path(__file__).resolve().parents[1]  # /app en Docker
    MODEL_PATH = Path(os.getenv("MODEL_PATH", str(REPO_ROOT / "modelo_xgboost_final.pkl")))
    METRICS_TXT = Path(os.getenv("METRICS_TXT", str(REPO_ROOT / "metrics.txt")))
    

    #La tabla training_dataset se crea en el load_training_dataset.py
    # Leo esa tabla

    data = pd.read_sql(
    "SELECT age, sex, bmi, children, smoker, region, charges FROM training_dataset;",
    engine
)
    # Separon en columnas numericas,categoricas y la variable que voy a querer predecir

    variables_numericas = ["age", "bmi", "children"]
    variables_categoricas = ["sex", "smoker", "region"]
    target = "charges"

    X = data[variables_numericas + variables_categoricas].copy()
    y = data[target].copy()   

    ## Separo los datos para entrenamiento

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    ## Preproccessing
    # Uso One-hot para las variables categóricas y las numercias no las toco.

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), variables_categoricas),
            ("num", "passthrough", variables_numericas),
        ],
        remainder="drop"
    )

    ## Armo el modelo y uso una escala logaritmica para la variable target para reducir el impacto de los outliers.
    xgb = XGBRegressor(
        objective="reg:absoluteerror",
        n_jobs=-1,
        random_state=42,
    )

    model = TransformedTargetRegressor(
        regressor=xgb,
        func=np.log1p,
        inverse_func=np.expm1
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

   ## Busqueda de hiperparametros
   # Busco los hipermarametros con GridSearch y cross-validation para obtener metricas mas confiables


    ## Legue a estos parametros luego de ir reduciendo el rango en el que el modelo funcionaba mejor
    param_grid = {
        "model__regressor__n_estimators": [150, 200, 250, 275, 300],
        "model__regressor__max_depth": [4, 5, 6],
        "model__regressor__learning_rate": [0.03, 0.05, 0.07, 0.08, 0.09],
    }

    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    # Obtengo las metricas del modelo

    best_idx = grid.best_index_
    cv_mae_mean = -grid.cv_results_["mean_test_score"][best_idx]
    cv_mae_std = grid.cv_results_["std_test_score"][best_idx]


    ## Me quedo con el mejor modelo de los hiperparaemtros probados
    best_model = grid.best_estimator_


    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)


    mae_train = mean_absolute_error(y_train, y_train_pred)
    ##En promedio por cuanto falla mi modelo:
    mae_test = mean_absolute_error(y_test, y_test_pred)
    rmse_test = mean_squared_error(y_test, y_test_pred) ** 0.5
    r2 = r2_score(y_test, y_test_pred)

    residuos = y_test - y_test_pred

    ## Analizando los residuos se ve claramente como el modelo cuanto  mas grande es el costo mas grande es el error del modelo(cosa que es dentro de todo esperable debido al sesgo)

    gap = mae_test - mae_train ## Cuanto mas bajo este valor mejor generaliza mi modelo

    ## Guardo el modelo

    joblib.dump(best_model, MODEL_PATH)


    # Guardo las metrticas

    txt = (
    "Metricas del modelo:\n"
    "================\n"
    f"CV (5-fold) MAE: {cv_mae_mean:.2f} ± {cv_mae_std:.2f}\n\n"
    f"Train MAE: {mae_train:.2f}\n"
    f"Test  MAE: {mae_test:.2f}\n"
    f"Gap (Test-Train): {gap:.2f}\n\n"
    f"Test RMSE: {rmse_test:.2f}\n"
    f"Test R2:   {r2:.4f}\n\n"
)
    METRICS_TXT.write_text(txt, encoding="utf-8")

    print("OK: training terminado.")

if __name__ == "__main__":
    main()