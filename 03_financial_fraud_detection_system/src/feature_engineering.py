import os
import numpy as np
import pandas as pd
from typing import Tuple

class FinancialFeatureEngineering:
    def __init__(self, data_path: str = None):
        if data_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(project_root, "data", "transactions.csv")
        self.data_path = data_path

    def generate_synthetic_transactions(self, n_samples: int = 10000, fraud_ratio: float = 0.02, random_state: int = 42) -> pd.DataFrame:
        """
        Generates realistic financial transaction streams with extreme class imbalance (e.g. 2% fraud).
        """
        np.random.seed(random_state)
        n_fraud = int(n_samples * fraud_ratio)
        n_legit = n_samples - n_fraud

        # Legitimate transactions
        legit_amounts = np.random.exponential(scale=45, size=n_legit).clip(2, 800).round(2)
        legit_velocity = np.random.poisson(lam=1.2, size=n_legit).clip(1, 6)
        legit_device_risk = np.random.beta(a=1, b=8, size=n_legit).round(3)
        legit_foreign_flag = np.random.choice([0, 1], size=n_legit, p=[0.95, 0.05])
        hour_probs = np.array([
            0.01, 0.01, 0.01, 0.01, 0.01, 0.02,
            0.03, 0.05, 0.06, 0.07, 0.08, 0.08,
            0.08, 0.08, 0.07, 0.07, 0.06, 0.06,
            0.05, 0.04, 0.03, 0.03, 0.02, 0.01
        ])
        hour_probs = hour_probs / np.sum(hour_probs)
        legit_hour = np.random.choice(range(24), size=n_legit, p=hour_probs)

        # Fraudulent transactions (high velocity, abnormal hours, high amounts, high device risk)
        fraud_amounts = np.random.exponential(scale=350, size=n_fraud).clip(150, 4000).round(2)
        fraud_velocity = np.random.poisson(lam=5.5, size=n_fraud).clip(3, 15)
        fraud_device_risk = np.random.beta(a=6, b=2, size=n_fraud).round(3)
        fraud_foreign_flag = np.random.choice([0, 1], size=n_fraud, p=[0.40, 0.60])
        fraud_hour = np.random.choice(range(24), size=n_fraud)  # Uniform across day/night

        legit_df = pd.DataFrame({
            "amount": legit_amounts,
            "velocity_1h": legit_velocity,
            "device_risk_score": legit_device_risk,
            "is_foreign_ip": legit_foreign_flag,
            "transaction_hour": legit_hour,
            "is_fraud": 0
        })

        fraud_df = pd.DataFrame({
            "amount": fraud_amounts,
            "velocity_1h": fraud_velocity,
            "device_risk_score": fraud_device_risk,
            "is_foreign_ip": fraud_foreign_flag,
            "transaction_hour": fraud_hour,
            "is_fraud": 1
        })

        df = pd.concat([legit_df, fraud_df], ignore_index=True)
        # Add engineered features: amount-to-velocity ratio and night transaction flag
        df["amount_per_velocity"] = (df["amount"] / (df["velocity_1h"] + 1)).round(2)
        df["is_night_transaction"] = df["transaction_hour"].apply(lambda h: 1 if h in [0, 1, 2, 3, 4] else 0)

        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        df.to_csv(self.data_path, index=False)
        return df

    def get_train_test_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        if not os.path.exists(self.data_path):
            self.generate_synthetic_transactions()
        df = pd.read_csv(self.data_path)

        train_size = int(len(df) * 0.8)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]

        X_train = train_df.drop(columns=["is_fraud"])
        y_train = train_df["is_fraud"]
        X_test = test_df.drop(columns=["is_fraud"])
        y_test = test_df["is_fraud"]

        return X_train, X_test, y_train, y_test
