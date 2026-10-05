import pandas as pd

from data.loader import load_dataset


DATA_PATH = "data/raw/customer_churn_dataset-testing-master.csv"

df = load_dataset(DATA_PATH)

features = [
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Subscription Type",
    "Contract Length",
    "Total Spend",
    "Last Interaction",
]

print("Churn rate:")
print(df["Churn"].value_counts(normalize=True).sort_index())

print("\nChurn rate by Payment Delay:")
print(
    df.groupby("Payment Delay")["Churn"]
    .agg(["count", "mean"])
    .round(4)
)

print("\nChurn rate by Support Calls:")
print(
    df.groupby("Support Calls")["Churn"]
    .agg(["count", "mean"])
    .round(4)
)

print("\nChurn rate by Tenure:")
print(
    df.groupby("Tenure")["Churn"]
    .agg(["count", "mean"])
    .round(4)
)

print("\nDuplicate feature rows with conflicting targets:")

duplicate_features = (
    df.groupby(features)["Churn"]
    .nunique()
)

conflicting = duplicate_features[duplicate_features > 1]

print(f"Number of feature combinations with conflicting targets: {len(conflicting)}")

print("\nExact duplicate rows:")
print(df.duplicated().sum())