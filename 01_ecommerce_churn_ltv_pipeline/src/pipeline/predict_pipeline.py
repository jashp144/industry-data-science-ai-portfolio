import os
import pandas as pd
import numpy as np
from src.logger import logger
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.preprocessor_path = os.path.join(project_root, "artifacts", "preprocessor.pkl")
        self.churn_model_path = os.path.join(project_root, "artifacts", "churn_model.pkl")
        self.ltv_model_path = os.path.join(project_root, "artifacts", "ltv_model.pkl")

        self.preprocessor = None
        self.churn_model = None
        self.ltv_model = None

    def _load_artifacts(self):
        if self.preprocessor is None:
            self.preprocessor = load_object(self.preprocessor_path)
            self.churn_model = load_object(self.churn_model_path)
            self.ltv_model = load_object(self.ltv_model_path)

    def predict(self, features: pd.DataFrame) -> list[dict]:
        try:
            self._load_artifacts()
            transformed_features = self.preprocessor.transform(features)

            # Predict Churn probability & label
            churn_probs = self.churn_model.predict_proba(transformed_features)[:, 1] if hasattr(self.churn_model, "predict_proba") else self.churn_model.predict(transformed_features)
            churn_preds = (churn_probs >= 0.5).astype(int)

            # Predict Lifetime Value
            ltv_preds = self.ltv_model.predict(transformed_features).round(2)

            results = []
            for i in range(len(features)):
                prob = float(churn_probs[i])
                risk_tier = "High" if prob >= 0.7 else ("Moderate" if prob >= 0.35 else "Low")
                results.append({
                    "churn_probability": round(prob, 4),
                    "will_churn": bool(churn_preds[i]),
                    "churn_risk_tier": risk_tier,
                    "estimated_ltv_usd": float(ltv_preds[i]),
                    "recommended_action": (
                        "Immediate retention outreach & VIP promo discount" if risk_tier == "High"
                        else ("Targeted re-engagement email campaign" if risk_tier == "Moderate"
                              else "Standard loyalty loyalty cross-sell")
                    )
                })

            return results
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise e

class CustomerData:
    def __init__(
        self,
        tenure_months: int,
        contract: str,
        payment_method: str,
        monthly_charges: float,
        total_charges: float,
        support_tickets: int,
        app_logins_per_month: int,
        cart_abandonment_rate: float,
        discount_applied: int
    ):
        self.tenure_months = tenure_months
        self.contract = contract
        self.payment_method = payment_method
        self.monthly_charges = monthly_charges
        self.total_charges = total_charges
        self.support_tickets = support_tickets
        self.app_logins_per_month = app_logins_per_month
        self.cart_abandonment_rate = cart_abandonment_rate
        self.discount_applied = discount_applied

    def get_data_as_data_frame(self) -> pd.DataFrame:
        data_dict = {
            "tenure_months": [self.tenure_months],
            "contract": [self.contract],
            "payment_method": [self.payment_method],
            "monthly_charges": [self.monthly_charges],
            "total_charges": [self.total_charges],
            "support_tickets": [self.support_tickets],
            "app_logins_per_month": [self.app_logins_per_month],
            "cart_abandonment_rate": [self.cart_abandonment_rate],
            "discount_applied": [self.discount_applied]
        }
        return pd.DataFrame(data_dict)
