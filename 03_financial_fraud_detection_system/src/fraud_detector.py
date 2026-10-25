import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, confusion_matrix
from src.feature_engineering import FinancialFeatureEngineering

class HybridFraudDetector:
    def __init__(self, model_path: str = None):
        if model_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(project_root, "data", "fraud_model.pkl")
        self.model_path = model_path
        self.supervised_model = None
        self.unsupervised_model = None
        self.optimal_threshold = 0.5
        self.metrics = {}

    def train_and_optimize(self) -> Dict[str, Any]:
        fe = FinancialFeatureEngineering()
        X_train, X_test, y_train, y_test = fe.get_train_test_split()

        # 1. Supervised Learning: Balanced Random Forest for Known Fraud Patterns
        self.supervised_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=7,
            class_weight="balanced",
            random_state=42
        )
        self.supervised_model.fit(X_train, y_train)

        # 2. Unsupervised Learning: Isolation Forest for Novel / Zero-Day Anomalies
        # Fit on legitimate transactions to model standard baseline behavior
        legit_train = X_train[y_train == 0]
        self.unsupervised_model = IsolationForest(
            n_estimators=100,
            contamination=0.025,
            random_state=42
        )
        self.unsupervised_model.fit(legit_train)

        # Evaluate Supervised & Unsupervised on Test Set
        y_sup_proba = self.supervised_model.predict_proba(X_test)[:, 1]
        # Invert decision function so higher score = more anomalous (0 to 1 range)
        raw_anomaly = -self.unsupervised_model.score_samples(X_test)
        norm_anomaly = (raw_anomaly - raw_anomaly.min()) / (raw_anomaly.max() - raw_anomaly.min() + 1e-6)

        # Hybrid Composite Risk Score: 70% Supervised + 30% Unsupervised Anomaly
        composite_score = 0.70 * y_sup_proba + 0.30 * norm_anomaly

        pr_auc = average_precision_score(y_test, composite_score)
        roc_auc = roc_auc_score(y_test, composite_score)

        # Optimize Decision Threshold based on Business Cost Matrix
        # Cost of False Negative (FN) = $400 (Direct chargeback loss)
        # Cost of False Positive (FP) = $15 (Manual verification/customer friction)
        best_cost = float("inf")
        best_thresh = 0.5
        test_amounts = X_test["amount"].values

        for thresh in np.linspace(0.1, 0.9, 81):
            y_pred = (composite_score >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            fn_mask = (y_test.values == 1) & (y_pred == 0)
            monetary_cost = np.sum(test_amounts[fn_mask]) + (fp * 15.0)

            if monetary_cost < best_cost:
                best_cost = monetary_cost
                best_thresh = thresh

        self.optimal_threshold = round(float(best_thresh), 3)

        # Baseline cost with naive 0.5 cutoff
        y_pred_naive = (composite_score >= 0.5).astype(int)
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
            "estimated_dollars_saved": round(float(cost_savings), 2),
            "architecture": "Hybrid Supervised (Balanced RF) + Unsupervised (Isolation Forest)"
        }

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump({
                "supervised_model": self.supervised_model,
                "unsupervised_model": self.unsupervised_model,
                "threshold": self.optimal_threshold,
                "metrics": self.metrics
            }, f)

        return self.metrics

    def load(self):
        if self.supervised_model is None:
            if not os.path.exists(self.model_path):
                self.train_and_optimize()
            else:
                with open(self.model_path, "rb") as f:
                    data = pickle.load(f)
                    self.supervised_model = data["supervised_model"]
                    self.unsupervised_model = data["unsupervised_model"]
                    self.optimal_threshold = data["threshold"]
                    self.metrics = data["metrics"]

    def score_transaction(self, tx_dict: dict) -> dict:
        self.load()
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
        sup_prob = float(self.supervised_model.predict_proba(df)[0, 1])
        raw_anomaly = float(-self.unsupervised_model.score_samples(df)[0])
        # Approximate normalization mapping for isolation score
        norm_anomaly = min(1.0, max(0.0, (raw_anomaly - 0.4) / 0.4))

        composite_risk = round(0.70 * sup_prob + 0.30 * norm_anomaly, 4)

        if composite_risk >= 0.75:
            decision = "IMMEDIATE_DECLINE"
            action = "Block transaction immediately and freeze payment card."
        elif composite_risk >= self.optimal_threshold:
            decision = "STEP_UP_AUTHENTICATION"
            action = "Trigger mandatory SMS/Biometric 2FA challenge."
        else:
            decision = "APPROVE"
            action = "Transaction cleared for instant settlement."

        return {
            "composite_risk_score": composite_risk,
            "supervised_fraud_probability": round(sup_prob, 4),
            "unsupervised_anomaly_score": round(norm_anomaly, 4),
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
# Keep backward-compatible alias
FraudDetector = HybridFraudDetector
