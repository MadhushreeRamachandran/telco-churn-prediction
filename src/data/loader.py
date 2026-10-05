from pathlib import Path

import pandas as pd


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Dataset is empty.")

    return df


def split_features_target(
    df: pd.DataFrame,
    target_column: str,
    id_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    id_columns = id_columns or []

    required_columns = [target_column, *id_columns]
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    X = df.drop(columns=[target_column, *id_columns])
    y = df[target_column]

    if y.isnull().any():
        raise ValueError("Target column contains missing values.")

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(
            f"Target column must contain only 0 and 1. "
            f"Found: {sorted(y.unique())}"
        )

    return X, y