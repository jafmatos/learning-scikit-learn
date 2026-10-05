from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "dataset" / "raw" / "housing.csv"
MODEL_PATH = ROOT_DIR / "models" / "essencial-do-scikit-learn.joblib"

NUMERICAL_FEATURES = [
    "latitude",
    "longitude",
    "mediana_idade_casas",
    "total_comodos",
    "total_quartos",
    "total_populacao",
    "total_casas",
    "mediana_renda_familiar",
]

CATEGORICAL_FEATURES = [
    "proximidade_oceano",
]

TARGET_FEATURE = "mediana_valor_casa"
