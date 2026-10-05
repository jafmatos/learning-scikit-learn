from scikitlearn.pipeline import full_pipeline
from scikitlearn.data import X_train, y_train
from sklearn.model_selection import cross_val_score

print(
    f"Average R2 Score: {cross_val_score(full_pipeline, X_train, y_train, cv=5, scoring="r2").mean()*100:.3f}%"
)
print(
    f"Standard Deviation: {cross_val_score(full_pipeline, X_train, y_train, cv=5, scoring="r2").std()*100:.3f}%"
)
