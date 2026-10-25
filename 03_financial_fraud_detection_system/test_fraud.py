import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.fraud_detector import HybridFraudDetector

class TestFraudDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n=== Training & Optimizing Hybrid Fraud Detection System ===")
        cls.detector = HybridFraudDetector()
        cls.metrics = cls.detector.train_and_optimize()
        print("Model Performance & Cost Metrics:", cls.metrics)
        assert cls.metrics["pr_auc"] > 0.60, "PR-AUC below threshold"
        assert cls.metrics["roc_auc"] > 0.80, "ROC-AUC below threshold"

    def test_legitimate_transaction_approval(self):
        print("\n=== Testing Legitimate Transaction ===")
        legit_tx = {
            "amount": 28.50,
            "velocity_1h": 1,
            "device_risk_score": 0.05,
            "is_foreign_ip": 0,
            "transaction_hour": 14
        }
        res = self.detector.score_transaction(legit_tx)
        print("Legit Transaction Result:", res)
        self.assertEqual(res["decision"], "APPROVE")
        self.assertLess(res["composite_risk_score"], self.detector.optimal_threshold)

    def test_fraudulent_transaction_intervention(self):
        print("\n=== Testing High-Risk Fraudulent Transaction ===")
        fraud_tx = {
            "amount": 1850.00,
            "velocity_1h": 8,
            "device_risk_score": 0.95,
            "is_foreign_ip": 1,
            "transaction_hour": 3
        }
        res = self.detector.score_transaction(fraud_tx)
        print("Fraud Transaction Result:", res)
        self.assertIn(res["decision"], ["IMMEDIATE_DECLINE", "STEP_UP_AUTHENTICATION"])
        self.assertGreaterEqual(res["composite_risk_score"], self.detector.optimal_threshold)

    def test_cost_savings(self):
        print("\n=== Testing Business Cost Savings Metric ===")
        print(f"Cost with optimized threshold: ${self.metrics['cost_with_optimized_threshold']}")
        print(f"Estimated dollars saved: ${self.metrics['estimated_dollars_saved']}")
        self.assertGreaterEqual(self.metrics["cost_with_optimized_threshold"], 0)

if __name__ == "__main__":
    unittest.main()
