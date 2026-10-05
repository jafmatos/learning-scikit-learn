from scikitlearn.config import NUMERICAL_FEATURES
from scikitlearn.config import CATEGORICAL_FEATURES
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
import pandas

NUMERICAL_FEATURES_ENGINEERED = NUMERICAL_FEATURES.copy()
NUMERICAL_FEATURES_ENGINEERED.append("quartos_por_comodo")
NUMERICAL_FEATURES_ENGINEERED.append("comodos_por_casa")
NUMERICAL_FEATURES_ENGINEERED.append("quartos_por_casa")
NUMERICAL_FEATURES_ENGINEERED.remove("total_quartos")
NUMERICAL_FEATURES_ENGINEERED.remove("total_comodos")


def feature_engineering(dataframe: pandas.DataFrame) -> pandas.DataFrame:
    dataframe = dataframe.copy()

    dataframe["quartos_por_comodo"] = (
        dataframe["total_quartos"] / dataframe["total_comodos"]
    )
    dataframe["comodos_por_casa"] = (
        dataframe["total_comodos"] / dataframe["total_casas"]
    )
    dataframe["quartos_por_casa"] = (
        dataframe["total_quartos"] / dataframe["total_casas"]
    )
    dataframe["mediana_renda_familiar"] = dataframe["mediana_renda_familiar"] * 10000

    return dataframe


FeatureTransformer = FunctionTransformer(feature_engineering)

numerical_pipeline = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", MinMaxScaler())]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numerical_pipeline, NUMERICAL_FEATURES_ENGINEERED),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ]
)

full_pipeline = Pipeline(
    steps=[
        ("feature_engineering", FeatureTransformer),
        ("preprocessing", preprocessor),
        ("model", LinearRegression()),
    ]
)


