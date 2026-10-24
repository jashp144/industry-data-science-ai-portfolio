import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.time_series_features import RetailTimeSeriesData

class DemandForecaster:
    def __init__(self, model_path: str = None):
        if model_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(project_root, "data", "forecaster_model.pkl")
        self.model_path = model_path
        self.model = None
        self.feature_cols = [
            "unit_price", "is_promo", "is_holiday", "day_of_week",
            "day_of_month", "month", "is_weekend",
            "lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_std_7", "rolling_mean_14"
        ]
        self.metrics = {}

    def train(self) -> Dict[str, Any]:
        ts_data = RetailTimeSeriesData()
        raw_df = ts_data.generate_synthetic_retail_data()
        df = ts_data.create_lag_features(raw_df)

        # Time-based Train-Test Split (Last 60 days as test)
        max_date = df["date"].max()
        cutoff_date = max_date - pd.Timedelta(days=60)

        train_df = df[df["date"] <= cutoff_date]
        test_df = df[df["date"] > cutoff_date]

        X_train = train_df[self.feature_cols]
        y_train = train_df["units_sold"]

        X_test = test_df[self.feature_cols]
        y_test = test_df["units_sold"]

        self.model = GradientBoostingRegressor(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=5,
            random_state=42
        )
        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        # WAPE: Sum(|Actual - Forecast|) / Sum(Actual)
        wape = np.sum(np.abs(y_test - preds)) / np.sum(y_test)
        mape = np.mean(np.abs((y_test - preds) / np.maximum(y_test, 1))) * 100

        self.metrics = {
            "mae": round(float(mae), 2),
            "rmse": round(float(rmse), 2),
            "wape": round(float(wape), 4),
            "mape_pct": round(float(mape), 2)
        }

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "metrics": self.metrics,
                "feature_cols": self.feature_cols
            }, f)

        return self.metrics

    def load(self):
        if self.model is None:
            if not os.path.exists(self.model_path):
                self.train()
            else:
                with open(self.model_path, "rb") as f:
                    data = pickle.load(f)
                    self.model = data["model"]
                    self.metrics = data["metrics"]
                    self.feature_cols = data["feature_cols"]

    def forecast_next_period(self, feature_row: dict) -> float:
        self.load()
        df = pd.DataFrame([feature_row])[self.feature_cols]
        pred = float(self.model.predict(df)[0])
        return max(0.0, round(pred, 1))
