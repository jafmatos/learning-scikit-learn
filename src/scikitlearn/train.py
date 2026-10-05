from scikitlearn.data import y_train
from scikitlearn.data import X_train
from scikitlearn.config import MODEL_PATH
from scikitlearn.pipeline import full_pipeline
import joblib

def train_and_save_model():
    full_pipeline.fit(X_train, y_train)

    joblib.dump(full_pipeline, MODEL_PATH)

if __name__ == "__main__":
    train_and_save_model()