import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class RetailTimeSeriesData:
    def __init__(self, data_path: str = None):
        if data_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(project_root, "data", "daily_retail_sales.csv")
        self.data_path = data_path

    def generate_synthetic_retail_data(self, n_days: int = 730, random_state: int = 42) -> pd.DataFrame:
        """
        Generates 2 years of daily multi-store, multi-product retail demand.
        Features include weekly seasonality, annual trend, promotions, and holidays.
        """
        np.random.seed(random_state)
        start_date = datetime(2024, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(n_days)]

        stores = ["Store_Downtown", "Store_Suburban"]
        items = ["SKU_Laptop_Accessories", "SKU_Ergonomic_Chair"]

        records = []
        for d in dates:
            day_of_week = d.weekday()  # 0=Monday, 6=Sunday
            is_weekend = 1 if day_of_week in [5, 6] else 0
            is_holiday = 1 if d.month == 12 and d.day in [24, 25, 31] else (1 if d.month == 11 and d.day in [28, 29] else 0)

            for store in stores:
                store_multiplier = 1.3 if store == "Store_Downtown" else 1.0
                for item in items:
                    base_demand = 80 if "Laptop" in item else 35
                    
                    # Seasonality & promotions
                    weekly_factor = 1.35 if is_weekend else 0.95
                    promo = np.random.choice([0, 1], p=[0.85, 0.15])
                    promo_boost = 1.4 if promo == 1 else 1.0
                    holiday_boost = 1.6 if is_holiday else 1.0

                    noise = np.random.normal(0, base_demand * 0.1)
                    demand = int(max(5, (base_demand * weekly_factor * promo_boost * holiday_boost * store_multiplier) + noise))
                    unit_price = 29.99 if "Laptop" in item else 189.99

                    records.append({
                        "date": d.strftime("%Y-%m-%d"),
                        "store_id": store,
                        "item_id": item,
                        "unit_price": unit_price,
                        "is_promo": promo,
                        "is_holiday": is_holiday,
                        "units_sold": demand
                    })

        df = pd.DataFrame(records)
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        df.to_csv(self.data_path, index=False)
        return df

    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineers time-series lag features (t-1, t-7, t-14), rolling window statistics, and calendar attributes.
        """
        df = df.sort_values(by=["store_id", "item_id", "date"]).reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])

        # Calendar features
        df["day_of_week"] = df["date"].dt.dayofweek
        df["day_of_month"] = df["date"].dt.day
        df["month"] = df["date"].dt.month
        df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x in [5, 6] else 0)

        # Lag features grouped by store and item
        grouped = df.groupby(["store_id", "item_id"])["units_sold"]
        df["lag_1"] = grouped.shift(1)
        df["lag_7"] = grouped.shift(7)
        df["lag_14"] = grouped.shift(14)

        # Rolling statistics (using shift(1) to avoid target leakage)
        df["rolling_mean_7"] = grouped.shift(1).rolling(window=7).mean().round(2)
        df["rolling_std_7"] = grouped.shift(1).rolling(window=7).std().round(2)
        df["rolling_mean_14"] = grouped.shift(1).rolling(window=14).mean().round(2)

        df = df.dropna().reset_index(drop=True)
        return df
