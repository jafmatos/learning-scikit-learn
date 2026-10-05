from scikitlearn.config import TARGET_FEATURE
from scikitlearn.config import CATEGORICAL_FEATURES
from scikitlearn.config import NUMERICAL_FEATURES
from scikitlearn.config import DATA_PATH
from sklearn.model_selection import train_test_split
import pandas

dataframe = pandas.read_csv(DATA_PATH).rename(
    columns={
        "housing_median_age": "mediana_idade_casas",
        "total_rooms": "total_comodos",
        "total_bedrooms": "total_quartos",
        "population": "total_populacao",
        "households": "total_casas",
        "median_income": "mediana_renda_familiar",
        "median_house_value": "mediana_valor_casa",
        "ocean_proximity": "proximidade_oceano",
    }
)

X = dataframe[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
y = dataframe[TARGET_FEATURE]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
