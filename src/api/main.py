from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


MODEL_PATH = Path("models/artifacts/churn_pipeline.joblib")


app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="API for predicting customer churn probability.",
    version="1.0.0",
)


class CustomerInput(BaseModel):
    Age: int = Field(ge=0)
    Gender: str
    Tenure: int = Field(ge=0)
    Usage_Frequency: int = Field(ge=0)
    Support_Calls: int = Field(ge=0)
    Payment_Delay: int = Field(ge=0)
    Subscription_Type: str
    Contract_Length: str
    Total_Spend: float = Field(ge=0)
    Last_Interaction: int = Field(ge=0)


class PredictionResponse(BaseModel):
    prediction: str
    churn_probability: float


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


model = load_model()


@app.get("/")
def root():
    return {
        "message": "Telco Customer Churn Prediction API",
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(customer: CustomerInput):
    input_data = {
        "Age": customer.Age,
        "Gender": customer.Gender,
        "Tenure": customer.Tenure,
        "Usage Frequency": customer.Usage_Frequency,
        "Support Calls": customer.Support_Calls,
        "Payment Delay": customer.Payment_Delay,
        "Subscription Type": customer.Subscription_Type,
        "Contract Length": customer.Contract_Length,
        "Total Spend": customer.Total_Spend,
        "Last Interaction": customer.Last_Interaction,
    }

    customer_df = pd.DataFrame([input_data])

    prediction = model.predict(customer_df)[0]
    probability = model.predict_proba(customer_df)[0][1]

    result = "Churn" if prediction == 1 else "No Churn"

    return {
        "prediction": result,
        "churn_probability": round(float(probability), 4),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )