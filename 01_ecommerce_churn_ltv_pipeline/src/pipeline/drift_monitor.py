import os
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from src.logger import logger
from src.utils import calculate_psi

class DataDriftMonitor:
    def __init__(self, baseline_train_path: str = None):
        if baseline_train_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            baseline_train_path = os.path.join(project_root, "artifacts", "train.csv")
        
        self.baseline_train_path = baseline_train_path
        self.baseline_df = None
        self.numerical_cols = [
            "tenure_months",
            "monthly_charges",
            "total_charges",
            "support_tickets",
            "app_logins_per_month",
            "cart_abandonment_rate"
        ]

    def _load_baseline(self):
        if self.baseline_df is None and os.path.exists(self.baseline_train_path):
            self.baseline_df = pd.read_csv(self.baseline_train_path)

    def evaluate_drift(self, current_df: pd.DataFrame) -> dict:
        self._load_baseline()
        if self.baseline_df is None:
            return {"status": "error", "message": "Baseline training data not available."}

        drift_report = {}
        drift_detected_count = 0
        max_psi = 0.0

        for col in self.numerical_cols:
            if col in current_df.columns and col in self.baseline_df.columns:
                base_vals = self.baseline_df[col].dropna().values
                curr_vals = current_df[col].dropna().values

                # 1. KS-Test
                ks_stat, p_val = ks_2samp(base_vals, curr_vals)
                
                # 2. PSI
                psi = calculate_psi(base_vals, curr_vals)
                if psi > max_psi:
                    max_psi = psi

                has_drift = bool(p_val < 0.05 and psi >= 0.1)
                if has_drift:
                    drift_detected_count += 1

                drift_report[col] = {
                    "ks_statistic": round(float(ks_stat), 4),
                    "p_value": round(float(p_val), 5),
                    "psi": round(float(psi), 4),
                    "is_drifted": has_drift
                }

        # Overall health assessment
        status = "HEALTHY"
        recommendation = "Model is operating within statistical distribution boundaries."
        if max_psi >= 0.25 or drift_detected_count >= 2:
            status = "CRITICAL_DRIFT"
            recommendation = "Severe distribution shift detected! Trigger automated retraining pipeline."
        elif max_psi >= 0.1 or drift_detected_count >= 1:
            status = "MODERATE_DRIFT"
            recommendation = "Moderate drift observed. Monitor upcoming batches closely."

        logger.info(f"Drift evaluation completed: {status} (Max PSI: {max_psi:.4f}, Drifted Features: {drift_detected_count})")

        return {
            "status": status,
            "max_psi": round(float(max_psi), 4),
            "drifted_features_count": drift_detected_count,
            "recommendation": recommendation,
            "feature_details": drift_report
        }
