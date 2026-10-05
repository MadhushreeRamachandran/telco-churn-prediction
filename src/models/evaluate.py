from pathlib import Path
import json

import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from data.loader import load_dataset, split_features_target


RANDOM_STATE = 42


def evaluate_model(
    data_path: str,
    model_path: str,
    metrics_path: str,
) -> dict:
    df = load_dataset(data_path)

    X, y = split_features_target(
        df,
        target_column="Churn",
        id_columns=["CustomerID"],
    )

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = joblib.load(Path(model_path))

    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_probability),
    }

    print("\nEvaluation Metrics")
    print("-" * 30)

    for metric, value in metrics.items():
        print(f"{metric.upper():<10}: {value:.4f}")

    print("\nClassification Report")
    print("-" * 30)
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix")
    print("-" * 30)
    print(confusion_matrix(y_test, y_pred))

    metrics_file = Path(metrics_path)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    with metrics_file.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"\nMetrics saved to: {metrics_file}")

    return metrics


if __name__ == "__main__":
    evaluate_model(
        data_path="data/raw/customer_churn_dataset-testing-master.csv",
        model_path="models/artifacts/churn_pipeline.joblib",
        metrics_path="models/artifacts/metrics.json",
    )