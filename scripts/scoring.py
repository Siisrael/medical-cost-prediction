import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import joblib

from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.compose import ColumnTransformer,TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error,make_scorer,r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression,Ridge
from xgboost import XGBRegressor

from pathlib import Path

import os
from sqlalchemy import create_engine

import numpy as np


def main():

    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "medical_data")
    DB_USER = os.getenv("DB_USER", "sisrael")
    DB_PASS = os.getenv("DB_PASS", "pass123")


    engine = create_engine(
            f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )

    REPO_ROOT = Path(__file__).resolve().parents[1]  # /app
    CSV_PATH = Path(os.getenv("CSV_PATH", str(REPO_ROOT / "dataset.csv")))
    MODEL_PATH = Path(os.getenv("MODEL_PATH", str(REPO_ROOT / "modelo_xgboost_final.pkl")))
    SCORING_TXT = Path(os.getenv("SCORING_TXT", str(REPO_ROOT / "scoring_metrics.txt")))


    data = pd.read_csv(CSV_PATH) 


    ## Creo un sample de 10 filas como me piden para hacer

    data_aleatoria = data.sample(n=10, random_state=42).reset_index(drop=True)

    sample_table = "sample_table"

    data_aleatoria.to_sql(sample_table, engine, if_exists="replace", index=False)

    scoring_df = pd.read_sql(f"SELECT * FROM {sample_table};", engine)

    ## Cargo el modelo entrenado y hago las prediciones sobre el sample 

    model = joblib.load(MODEL_PATH) 

    features = ["age","sex","bmi","children","smoker","region"]

    X = scoring_df[features].copy()

    y = scoring_df["charges"].copy() 

    y_pred = model.predict(X)

    ## Guardo los valores de las predicciones en una columna nueva en una nueva tabla

    scoring_df["predicted_charges"] = y_pred

    scoring_df.to_sql("predictions_table", engine, if_exists="replace", index=False) 


    ## Guardo meticas en un .txt

    mae_final = mean_absolute_error(y, y_pred)

    txt = (
        "Metricas sample de 10 filas\n"
        "==============================\n\n"
        "Estadisticas de la columna cherges del sample:\n"
        f"- Mean:   {float(np.mean(y)):.2f}\n"
        f"- Min:    {float(np.min(y)):.2f}\n"
        f"- Max:    {float(np.max(y)):.2f}\n"
        f"- Std:    {float(np.std(y)):.2f}\n\n"
        "Metricas:\n"
        f"- MAE:  {mae_final:.2f}\n"
        f"- RMSE: {mean_squared_error(y, y_pred) ** 0.5:.2f}\n\n"
        "Informacion extra sobre las predicciones:\n"
        f"- Pred mean: {float(np.mean(y_pred)):.2f}\n"
        f"- Pred min:  {float(np.min(y_pred)):.2f}\n"
        f"- Pred max:  {float(np.max(y_pred)):.2f}\n"
        f"- Pred std:  {float(np.std(y_pred)):.2f}\n"
    )

    SCORING_TXT.write_text(txt, encoding="utf-8")

    print("OK: Scoring terminado")

if __name__ == "__main__":
    main()