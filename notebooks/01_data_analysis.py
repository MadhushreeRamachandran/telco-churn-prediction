import pandas as pd

from data.loader import load_dataset


DATA_PATH = "data/raw/customer_churn_dataset-testing-master.csv"


df = load_dataset(DATA_PATH)

print("Dataset shape:")
print(df.shape)

print("\nData types:")
print(df.dtypes)

print("\nUnique values:")
for column in df.columns:
    print(f"\n{column}:")
    print(df[column].nunique())

print("\nNumerical correlation with Churn:")
numeric_df = df.select_dtypes(include="number")
print(
    numeric_df.corr()["Churn"]
    .sort_values(ascending=False)
)

print("\nChurn rate by categorical features:")

categorical_columns = [
    "Gender",
    "Subscription Type",
    "Contract Length",
]

for column in categorical_columns:
    print(f"\n--- {column} ---")
    print(
        df.groupby(column)["Churn"]
        .agg(["count", "mean"])
        .sort_values("mean", ascending=False)
    )