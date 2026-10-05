from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/artifacts/churn_pipeline.joblib")

VALID_GENDERS = {"Male", "Female"}
VALID_SUBSCRIPTIONS = {"Basic", "Standard", "Premium"}
VALID_CONTRACTS = {"Monthly", "Quarterly", "Annual"}


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def get_integer_input(prompt: str, minimum: int = 0) -> int:
    while True:
        try:
            value = int(input(prompt))

            if value < minimum:
                print(
                    f"Please enter a value greater than or equal to {minimum}."
                )
                continue

            return value

        except ValueError:
            print("Please enter a valid integer.")


def get_float_input(prompt: str, minimum: float = 0) -> float:
    while True:
        try:
            value = float(input(prompt))

            if value < minimum:
                print(
                    f"Please enter a value greater than or equal to {minimum}."
                )
                continue

            return value

        except ValueError:
            print("Please enter a valid number.")


def get_choice(prompt: str, valid_values: set[str]) -> str:
    while True:
        value = input(prompt).strip().title()

        if value in valid_values:
            return value

        print(
            f"Invalid choice. Please choose from: "
            f"{', '.join(sorted(valid_values))}"
        )


def get_customer_input() -> pd.DataFrame:
    print("\nEnter customer information")
    print("-" * 30)

    customer = {
        "Age": get_integer_input("Age: ", minimum=0),
        "Gender": get_choice(
            "Gender (Male/Female): ",
            VALID_GENDERS,
        ),
        "Tenure": get_integer_input("Tenure: ", minimum=0),
        "Usage Frequency": get_integer_input(
            "Usage Frequency: ",
            minimum=0,
        ),
        "Support Calls": get_integer_input(
            "Support Calls: ",
            minimum=0,
        ),
        "Payment Delay": get_integer_input(
            "Payment Delay: ",
            minimum=0,
        ),
        "Subscription Type": get_choice(
            "Subscription Type (Basic/Standard/Premium): ",
            VALID_SUBSCRIPTIONS,
        ),
        "Contract Length": get_choice(
            "Contract Length (Monthly/Quarterly/Annual): ",
            VALID_CONTRACTS,
        ),
        "Total Spend": get_float_input(
            "Total Spend: ",
            minimum=0,
        ),
        "Last Interaction": get_integer_input(
            "Last Interaction: ",
            minimum=0,
        ),
    }

    return pd.DataFrame([customer])


def predict_churn(
    model,
    customer: pd.DataFrame,
) -> None:
    prediction = model.predict(customer)[0]
    probability = model.predict_proba(customer)[0][1]

    result = "Churn" if prediction == 1 else "No Churn"

    print("\nPrediction Result")
    print("-" * 30)
    print(f"Prediction: {result}")
    print(f"Churn Probability: {probability:.2%}")


def main():
    model = load_model()
    customer = get_customer_input()
    predict_churn(model, customer)


if __name__ == "__main__":
    main()