import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.logger import logger

@dataclass
class DataIngestionConfig:
    project_root: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_data_path: str = os.path.join(project_root, "data", "raw_customers.csv")
    train_data_path: str = os.path.join(project_root, "artifacts", "train.csv")
    test_data_path: str = os.path.join(project_root, "artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def generate_synthetic_enterprise_data(self, n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
        """
        Generates realistic E-Commerce customer retention and transaction data.
        Features mimic actual subscription/e-commerce platforms.
        """
        np.random.seed(random_state)
        
        customer_ids = [f"CUST_{10000 + i}" for i in range(n_samples)]
        tenure_months = np.random.exponential(scale=18, size=n_samples).clip(1, 72).astype(int)
        contract = np.random.choice(["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.55, 0.25, 0.20])
        payment_method = np.random.choice(["Credit Card", "Bank Transfer", "Electronic Check", "UPI"], size=n_samples)
        
        monthly_charges = np.random.normal(loc=65, scale=25, size=n_samples).clip(15, 150).round(2)
        total_charges = (monthly_charges * tenure_months + np.random.normal(0, 15, size=n_samples)).clip(15, None).round(2)
        
        support_tickets = np.random.poisson(lam=1.5, size=n_samples).clip(0, 10)
        app_logins_per_month = np.random.negative_binomial(n=5, p=0.25, size=n_samples).clip(0, 60)
        cart_abandonment_rate = np.random.beta(a=2, b=5, size=n_samples).round(3)
        discount_applied = np.random.choice([0, 1], size=n_samples, p=[0.65, 0.35])

        # Latent churn probability computation
        log_odds = (
            -1.8
            - 0.05 * tenure_months
            + 0.015 * monthly_charges
            + 0.45 * support_tickets
            - 0.04 * app_logins_per_month
            + 1.2 * (contract == "Month-to-month")
            - 0.8 * (contract == "Two year")
            + 0.5 * cart_abandonment_rate
            + np.random.normal(0, 0.5, size=n_samples)
        )
        churn_prob = 1 / (1 + np.exp(-log_odds))
        churn = (np.random.rand(n_samples) < churn_prob).astype(int)

        # Customer Lifetime Value (CLV in USD)
        # Higher tenure, higher monthly charges, and lower churn risk yield higher LTV
        expected_remaining_months = np.where(churn == 1, 0, np.maximum(3, 48 - tenure_months * 0.5 + np.random.normal(0, 5, n_samples)))
        clv = (total_charges + (monthly_charges * expected_remaining_months * 0.8)).clip(20, 12000).round(2)

        df = pd.DataFrame({
            "customer_id": customer_ids,
            "tenure_months": tenure_months,
            "contract": contract,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "support_tickets": support_tickets,
            "app_logins_per_month": app_logins_per_month,
            "cart_abandonment_rate": cart_abandonment_rate,
            "discount_applied": discount_applied,
            "clv": clv,
            "churn": churn
        })

        return df

    def initiate_data_ingestion(self):
        logger.info("Initiating Data Ingestion component...")
        os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

        df = self.generate_synthetic_enterprise_data()
        df.to_csv(self.ingestion_config.raw_data_path, index=False)
        logger.info(f"Raw data saved to {self.ingestion_config.raw_data_path} with shape {df.shape}")

        # Train-Test Split (80/20 stratified split by churn)
        shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        train_idx = int(len(shuffled) * 0.8)
        train_df = shuffled.iloc[:train_idx]
        test_df = shuffled.iloc[train_idx:]

        train_df.to_csv(self.ingestion_config.train_data_path, index=False)
        test_df.to_csv(self.ingestion_config.test_data_path, index=False)
        logger.info(f"Train dataset ({train_df.shape}) and Test dataset ({test_df.shape}) saved.")

        return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.initiate_data_ingestion()
