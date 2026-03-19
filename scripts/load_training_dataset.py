import os
import pandas as pd
from sqlalchemy import create_engine, text

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "medical_data")
DB_USER = os.getenv("DB_USER", "sisrael")
DB_PASS = os.getenv("DB_PASS", "pass123")

CSV_PATH = os.getenv("CSV_PATH", "dataset.csv")

def main():
        
    
    engine = create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    create_sql = """
    CREATE TABLE IF NOT EXISTS training_dataset (
    id SERIAL PRIMARY KEY,
    age INTEGER NOT NULL,
    sex TEXT NOT NULL,
    bmi DOUBLE PRECISION NOT NULL,
    children INTEGER NOT NULL,
    smoker TEXT NOT NULL,
    region TEXT NOT NULL,
    charges DOUBLE PRECISION NOT NULL
    );
    TRUNCATE TABLE training_dataset;
    """

    with engine.begin() as conn:
        conn.execute(text(create_sql))

    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip().lower() for c in df.columns]


    # Casts defensivos (evita que se inserte como texto)
    df["age"] = df["age"].astype(int)
    df["children"] = df["children"].astype(int)
    df["bmi"] = df["bmi"].astype(float)
    df["charges"] = df["charges"].astype(float)

    # Limpio antes de insertar
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE training_dataset;"))

    # Inserto
    df.to_sql("training_dataset", engine, if_exists="append", index=False, method="multi")

    # Verifico que se hayan insertado todas las filas.
    with engine.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM training_dataset;")).scalar_one()

    print(f"OK: training_dataset cargada con {n} filas.")

if __name__ == "__main__":
    main()
