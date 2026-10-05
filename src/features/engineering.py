import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TelcoFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create additional customer-level features for churn prediction."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()

        tenure = data["Tenure"].clip(lower=1)

        data["support_call_rate"] = (
            data["Support Calls"] / tenure
        )

        data["payment_delay_rate"] = (
            data["Payment Delay"] / tenure
        )

        data["usage_per_tenure"] = (
            data["Usage Frequency"] / tenure
        )

        data["spend_per_tenure"] = (
            data["Total Spend"] / tenure
        )

        return data