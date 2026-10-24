import os
import sys
import unittest
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.pipeline.predict_pipeline import PredictPipeline, CustomerData
from src.pipeline.drift_monitor import DataDriftMonitor

class TestMLPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n=== STEP 1: Running Data Ingestion ===")
        ingestion = DataIngestion()
        cls.train_path, cls.test_path = ingestion.initiate_data_ingestion()
        assert os.path.exists(cls.train_path), "Train data file missing"
        assert os.path.exists(cls.test_path), "Test data file missing"

        print("\n=== STEP 2: Running Data Transformation ===")
        transformation = DataTransformation()
        cls.train_arr, cls.test_arr, cls.preprocessor_path = transformation.initiate_data_transformation(
            cls.train_path, cls.test_path
        )
        assert os.path.exists(cls.preprocessor_path), "Preprocessor pickle missing"

        print("\n=== STEP 3: Running Model Training & Evaluation ===")
        trainer = ModelTrainer()
        cls.metrics = trainer.initiate_model_trainer(cls.train_arr, cls.test_arr)
        print("Training Metrics:", cls.metrics)
        assert os.path.exists(trainer.config.churn_model_path), "Churn model pickle missing"
        assert os.path.exists(trainer.config.ltv_model_path), "LTV model pickle missing"

    def test_single_prediction(self):
        print("\n=== STEP 4: Verifying Single Prediction ===")
        pipeline = PredictPipeline()
        cust = CustomerData(
            tenure_months=6,
            contract="Month-to-month",
            payment_method="Electronic Check",
            monthly_charges=85.0,
            total_charges=510.0,
            support_tickets=4,
            app_logins_per_month=3,
            cart_abandonment_rate=0.75,
            discount_applied=0
        )
        df = cust.get_data_as_data_frame()
        res = pipeline.predict(df)[0]
        print("Prediction result for high-risk customer:", res)
        self.assertIn("churn_probability", res)
        self.assertIn("estimated_ltv_usd", res)
        self.assertGreaterEqual(res["churn_probability"], 0.0)
        self.assertLessEqual(res["churn_probability"], 1.0)

    def test_batch_prediction_and_drift_detection(self):
        print("\n=== STEP 5: Verifying Batch Prediction & Drift Detection ===")
        # 1. Normal batch (sampled from test)
        test_df = pd.read_csv(self.test_path).drop(columns=["customer_id", "churn", "clv"], errors="ignore")
        normal_batch = test_df.sample(n=200, random_state=42)

        monitor = DataDriftMonitor()
        normal_drift_report = monitor.evaluate_drift(normal_batch)
        print("Normal batch drift status:", normal_drift_report["status"], "Max PSI:", normal_drift_report["max_psi"])
        self.assertEqual(normal_drift_report["status"], "HEALTHY")

        # 2. Artificially drifted batch (severe shift in support tickets and charges)
        drifted_batch = normal_batch.copy()
        drifted_batch["support_tickets"] = np.random.poisson(lam=8.0, size=len(drifted_batch))
        drifted_batch["monthly_charges"] = drifted_batch["monthly_charges"] * 2.5

        drifted_report = monitor.evaluate_drift(drifted_batch)
        print("Drifted batch status:", drifted_report["status"], "Max PSI:", drifted_report["max_psi"])
        self.assertIn(drifted_report["status"], ["MODERATE_DRIFT", "CRITICAL_DRIFT"])
        self.assertGreater(drifted_report["max_psi"], 0.1)

if __name__ == "__main__":
    unittest.main()
