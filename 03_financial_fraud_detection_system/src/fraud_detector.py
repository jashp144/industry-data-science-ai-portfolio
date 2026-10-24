import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, f1_score, confusion_matrix
from src.feature_engineering import FinancialFeatureEngineering

class FraudDetector:
    def __init__(self, model_path: str = None):
        if model_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(project_root, "data", "fraud_model.pkl")
        self.model_path = model_path
        self.model = None
        self.optimal_threshold = 0.5
        self.metrics = {}

    def train_and_optimize(self) -> Dict[str, Any]:
        fe = FinancialFeatureEngineering()
        X_train, X_test, y_train, y_test = fe.get_train_test_split()

        # Balanced Random Forest for high class imbalance
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=7,
            class_weight="balanced",
            random_state=42
        )
        self.model.fit(X_train, y_train)

        y_proba = self.model.predict_proba(X_test)[:, 1]

        # Calculate PR-AUC and ROC-AUC
        pr_auc = average_precision_score(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)

        # Optimize Decision Threshold based on Business Cost Matrix
        # Cost of False Negative (FN) = $400 (Direct chargeback loss)
        # Cost of False Positive (FP) = $15 (Manual verification/customer support friction)
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

        best_cost = float("inf")
        best_thresh = 0.5

        test_amounts = X_test["amount"].values

        for thresh in np.linspace(0.1, 0.9, 81):
            y_pred = (y_proba >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            
            # Monetary Cost: FN * amount + FP * $15
            fn_mask = (y_test.values == 1) & (y_pred == 0)
            fp_count = fp
            monetary_cost = np.sum(test_amounts[fn_mask]) + (fp_count * 15.0)

            if monetary_cost < best_cost:
                best_cost = monetary_cost
                best_thresh = thresh

        self.optimal_threshold = round(float(best_thresh), 3)

        # Calculate baseline cost with naive 0.5 threshold
        y_pred_naive = (y_proba >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_naive).ravel()
        fn_mask_naive = (y_test.values == 1) & (y_pred_naive == 0)
        naive_cost = np.sum(test_amounts[fn_mask_naive]) + (fp * 15.0)
        cost_savings = max(0.0, naive_cost - best_cost)

        self.metrics = {
            "pr_auc": round(float(pr_auc), 4),
            "roc_auc": round(float(roc_auc), 4),
            "optimal_threshold": self.optimal_threshold,
            "cost_with_optimized_threshold": round(float(best_cost), 2),
            "cost_with_default_threshold": round(float(naive_cost), 2),
            "estimated_dollars_saved": round(float(cost_savings), 2)
        }

        # Save model and parameters
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "threshold": self.optimal_threshold,
                "metrics": self.metrics
            }, f)

        return self.metrics

    def load(self):
        if self.model is None:
            if not os.path.exists(self.model_path):
                self.train_and_optimize()
            else:
                with open(self.model_path, "rb") as f:
                    data = pickle.load(f)
                    self.model = data["model"]
                    self.optimal_threshold = data["threshold"]
                    self.metrics = data["metrics"]

    def score_transaction(self, tx_dict: dict) -> dict:
        self.load()
        # Compute engineered features
        amount = float(tx_dict["amount"])
        velocity = float(tx_dict["velocity_1h"])
        hour = int(tx_dict["transaction_hour"])

        feature_dict = {
            "amount": amount,
            "velocity_1h": velocity,
            "device_risk_score": float(tx_dict["device_risk_score"]),
            "is_foreign_ip": int(tx_dict["is_foreign_ip"]),
            "transaction_hour": hour,
            "amount_per_velocity": round(amount / (velocity + 1), 2),
            "is_night_transaction": 1 if hour in [0, 1, 2, 3, 4] else 0
        }

        df = pd.DataFrame([feature_dict])
        fraud_prob = float(self.model.predict_proba(df)[0, 1])
        is_flagged = fraud_prob >= self.optimal_threshold

        # Tri-tier decisioning
        if fraud_prob >= 0.75:
            decision = "IMMEDIATE_DECLINE"
            action = "Block transaction immediately and notify cardholder of suspicious activity."
        elif fraud_prob >= self.optimal_threshold:
            decision = "STEP_UP_AUTHENTICATION"
            action = "Trigger mandatory SMS/Biometric OTP step-up challenge."
        else:
            decision = "APPROVE"
            action = "Transaction cleared for instant settlement."

        return {
            "fraud_probability": round(fraud_prob, 4),
            "optimal_threshold": self.optimal_threshold,
            "decision": decision,
            "recommended_action": action,
            "risk_factors": {
                "high_velocity": velocity >= 4,
                "high_amount": amount >= 500,
                "suspicious_device": float(tx_dict["device_risk_score"]) >= 0.5,
                "night_hour": hour in [0, 1, 2, 3, 4]
            }
        }
