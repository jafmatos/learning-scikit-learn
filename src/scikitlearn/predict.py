from scikitlearn.config import MODEL_PATH
from scikitlearn.config import TARGET_FEATURE
from sklearn.pipeline import Pipeline
import pandas
import joblib
from pathlib import Path
from functools import lru_cache

REQUIRED_COLUMNS = [
    "latitude",
    "longitude",
    "mediana_idade_casas",
    "total_comodos",
    "total_quartos",
    "total_populacao",
    "total_casas",
    "mediana_renda_familiar",
    "proximidade_oceano",
]
TARGET_COLUMN = TARGET_FEATURE

@lru_cache(maxsize=1)
def load_model(model_path: Path | str | None = None):
    path = Path(model_path) if model_path else MODEL_PATH

    if not Path(path).exists():
        raise FileNotFoundError(f"Modelo não encontrado em '{path}'")

    return joblib.load(path)

def predict_single(
    model: Pipeline,
    latitude: float, 
    longitude: float, 
    mediana_idade_casas: int, 
    total_comodos: int,
    total_quartos: int, 
    total_populacao: int, 
    total_casas: int, 
    mediana_renda_familiar: float,
    proximidade_oceano: str
    ):
    input_dataframe = pandas.DataFrame(
        {
            "latitude": [latitude], 
            "longitude": [longitude], 
            "mediana_idade_casas": [mediana_idade_casas], 
            "total_comodos": [total_comodos], 
            "total_quartos": [total_quartos], 
            "total_populacao": [total_populacao], 
            "total_casas": [total_casas], 
            "mediana_renda_familiar": [mediana_renda_familiar],
            "proximidade_oceano": [proximidade_oceano],
        }
    )
    prediction = model.predict(input_dataframe)

    return float(prediction[0])


def predict_batch(model: Pipeline, data: list[dict]):
    dataframe = pandas.DataFrame(data)
    missing_columns = [
        column for column in REQUIRED_COLUMNS 
        if column not in dataframe.columns
    ]

    if (missing_columns):
        raise ValueError(f"Faltam colunas obrigatórias: {missing_columns}")

    predictions = model.predict(dataframe[REQUIRED_COLUMNS])

    for index in range(0, len(data)):
        data[index][TARGET_COLUMN] = float(predictions[index])

    return data

