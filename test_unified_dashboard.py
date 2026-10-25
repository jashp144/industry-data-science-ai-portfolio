import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
from unified_dashboard import app

class TestUnifiedDashboard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_dashboard_index_and_timing_header(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("Enterprise AI & Machine Learning Systems Suite", res.text)
        self.assertIn("7/7 Microservices Active", res.text)
        self.assertIn("chart.umd.min.js", res.text)
        self.assertIn("x-process-time-ms", res.headers)

    def test_health_endpoint(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "HEALTHY")
        self.assertEqual(data["active_microservices"], 7)
        self.assertIn("uptime_seconds", data)
        self.assertIn("memory_used_mb", data)
        self.assertEqual(data["systems_status"]["01_churn_ltv"], "ONLINE")
        self.assertEqual(data["systems_status"]["02_hybrid_rag"], "ONLINE")

    def test_metrics_telemetry_endpoint(self):
        res = self.client.get("/api/metrics")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("mean_inference_latency_ms", data)
        self.assertIn("data_drift_psi", data)
        self.assertIn("fraud_loss_avoidance_monthly_usd", data)
        self.assertEqual(data["ci_cd_status"], "ALL_8_SUITES_PASSING")

    def test_sql_schema_endpoint(self):
        res = self.client.get("/api/sql/schema")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("customers", data)
        self.assertIn("products", data)
        self.assertIn("orders", data)
        self.assertIn("order_items", data)
        self.assertGreater(data["customers"]["row_count"], 50)

    def test_churn_endpoint(self):
        payload = {
            "tenure_months": 8,
            "monthly_charges": 80.0,
            "support_tickets": 3,
            "contract": "Month-to-month",
            "payment_method": "Credit Card",
            "total_charges": 640.0,
            "app_logins_per_month": 10,
            "cart_abandonment_rate": 0.4,
            "discount_applied": 0
        }
        res = self.client.post("/api/churn", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("will_churn", data)
        self.assertIn("churn_probability", data)
        self.assertIn("estimated_ltv_usd", data)
        self.assertIn("top_risk_drivers", data)

    def test_rag_endpoint(self):
        res = self.client.post("/api/rag", json={"query": "What is the refund window for subscriptions?"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("answer", data)
        self.assertIn("retrieval_method", data)
        self.assertIn("top_ranking_metrics", data)
        self.assertIn("sources", data)

    def test_fraud_endpoint(self):
        payload = {
            "amount": 1200.0,
            "velocity_1h": 8,
            "device_risk_score": 0.92,
            "is_foreign_ip": 1,
            "transaction_hour": 3
        }
        res = self.client.post("/api/fraud", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("decision", data)
        self.assertIn("composite_risk_score", data)
        self.assertIn("supervised_fraud_probability", data)
        self.assertIn("unsupervised_anomaly_score", data)

    def test_demand_forecast_endpoint(self):
        payload = {
            "unit_price": 29.99,
            "is_promo": 1,
            "is_holiday": 0,
            "day_of_week": 5,
            "day_of_month": 20,
            "month": 10,
            "is_weekend": 1,
            "lag_1": 110.0,
            "lag_7": 115.0,
            "lag_14": 108.0,
            "rolling_mean_7": 105.0,
            "rolling_std_7": 12.5,
            "rolling_mean_14": 102.0,
            "current_inventory": 250,
            "lead_time_days": 5
        }
        res = self.client.post("/api/forecast", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("predicted_demand", data)
        self.assertIn("plan", data)
        self.assertIn("historical_series", data)
        self.assertIn("monte_carlo_simulation", data["plan"])
        self.assertIn("histogram_labels", data["plan"]["monte_carlo_simulation"])

    def test_sql_agent_endpoint(self):
        res = self.client.post("/api/sql", json={"question": "Segment customers into RFM quartiles"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("query", data)
        self.assertIn("insights", data)

    def test_real_estate_endpoint(self):
        payload = {
            "RM": 7.2,
            "LSTAT": 4.5,
            "CRIM": 0.08,
            "PTRATIO": 15.0,
            "ZN": 20.0,
            "INDUS": 5.5,
            "CHAS": 1,
            "NOX": 0.45,
            "AGE": 25.0,
            "DIS": 4.5,
            "RAD": 4,
            "TAX": 280,
            "B": 390.0,
            "interest_rate_pct": 6.5,
            "down_payment_pct": 0.20
        }
        res = self.client.post("/api/realestate", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("valuation_and_roi", data)
        self.assertIn("stress_test_matrix", data)

    def test_talent_endpoint(self):
        payload = {
            "city": "Bangalore",
            "experience_years": 3.0,
            "primary_stack": "Python/AI",
            "has_cloud_skill": 0,
            "has_genai_skill": 0,
            "company_tier": "Product/Unicorn"
        }
        res = self.client.post("/api/talent", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("predicted_salary_lpa", data)
        self.assertIn("projected_salary_with_both_skills_lpa", data)
        self.assertIn("strategic_skill_recommendations", data)

if __name__ == "__main__":
    unittest.main()
