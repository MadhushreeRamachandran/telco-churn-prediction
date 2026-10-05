from pathlib import Path

import gradio as gr
import joblib
import pandas as pd


MODEL_PATH = Path("models/artifacts/churn_pipeline.joblib")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


model = load_model()


def predict_churn(
    age,
    gender,
    tenure,
    usage_frequency,
    support_calls,
    payment_delay,
    subscription_type,
    contract_length,
    total_spend,
    last_interaction,
):
    customer = pd.DataFrame(
        [
            {
                "Age": age,
                "Gender": gender,
                "Tenure": tenure,
                "Usage Frequency": usage_frequency,
                "Support Calls": support_calls,
                "Payment Delay": payment_delay,
                "Subscription Type": subscription_type,
                "Contract Length": contract_length,
                "Total Spend": total_spend,
                "Last Interaction": last_interaction,
            }
        ]
    )

    prediction = model.predict(customer)[0]
    probability = model.predict_proba(customer)[0][1]

    result = "Churn" if prediction == 1 else "No Churn"

    return (
        result,
        f"{probability:.2%}",
    )


with gr.Blocks(title="Telco Churn Prediction") as demo:
    gr.Markdown(
        """
        # Telco Customer Churn Prediction

        Enter customer information to predict the likelihood of churn.
        """
    )

    with gr.Row():
        with gr.Column():
            age = gr.Number(
                label="Age",
                minimum=0,
                value=30,
            )

            gender = gr.Dropdown(
                choices=["Male", "Female"],
                label="Gender",
                value="Male",
            )

            tenure = gr.Number(
                label="Tenure",
                minimum=0,
                value=12,
            )

            usage_frequency = gr.Number(
                label="Usage Frequency",
                minimum=0,
                value=10,
            )

            support_calls = gr.Number(
                label="Support Calls",
                minimum=0,
                value=2,
            )

        with gr.Column():
            payment_delay = gr.Number(
                label="Payment Delay",
                minimum=0,
                value=5,
            )

            subscription_type = gr.Dropdown(
                choices=[
                    "Basic",
                    "Standard",
                    "Premium",
                ],
                label="Subscription Type",
                value="Standard",
            )

            contract_length = gr.Dropdown(
                choices=[
                    "Monthly",
                    "Quarterly",
                    "Annual",
                ],
                label="Contract Length",
                value="Annual",
            )

            total_spend = gr.Number(
                label="Total Spend",
                minimum=0,
                value=500,
            )

            last_interaction = gr.Number(
                label="Last Interaction",
                minimum=0,
                value=10,
            )

    predict_button = gr.Button("Predict Churn")

    prediction_output = gr.Textbox(
        label="Prediction",
    )

    probability_output = gr.Textbox(
        label="Churn Probability",
    )

    predict_button.click(
        fn=predict_churn,
        inputs=[
            age,
            gender,
            tenure,
            usage_frequency,
            support_calls,
            payment_delay,
            subscription_type,
            contract_length,
            total_spend,
            last_interaction,
        ],
        outputs=[
            prediction_output,
            probability_output,
        ],
    )


if __name__ == "__main__":
    demo.launch()