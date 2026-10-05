from pathlib import Path

import joblib

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data.loader import load_dataset, split_features_target
from src.features.engineering import TelcoFeatureEngineer
from src.features.preprocessing import build_preprocessor


RANDOM_STATE = 42


def build_model() -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("feature_engineering", TelcoFeatureEngineer()),
            ("preprocessor", build_preprocessor()),
            ("classifier", build_model()),
        ]
    )


def train_model(
    data_path: str,
    model_path: str,
) -> Pipeline:
    df = load_dataset(data_path)

    X, y = split_features_target(
        df,
        target_column="Churn",
        id_columns=["CustomerID"],
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = build_pipeline()

    parameter_distributions = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [3, 5, 7],
        "classifier__learning_rate": [0.01, 0.05, 0.1],
        "classifier__subsample": [0.8, 1.0],
        "classifier__colsample_bytree": [0.8, 1.0],
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_distributions,
        n_iter=10,
        scoring="roc_auc",
        cv=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)

    print("\nBest Parameters")
    print("-" * 30)
    print(search.best_params_)

    print("\nBest Cross-Validation ROC-AUC")
    print("-" * 30)
    print(f"{search.best_score_:.4f}")

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(search.best_estimator_, model_path)

    print(f"\nModel saved to: {model_path}")

    return search.best_estimator_


if __name__ == "__main__":
    train_model(
        data_path="data/raw/customer_churn_dataset-testing-master.csv",
        model_path="models/artifacts/churn_pipeline.joblib",
    )