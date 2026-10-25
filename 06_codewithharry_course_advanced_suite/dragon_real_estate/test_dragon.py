import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model_pipeline import DragonRealEstateEngine

class TestDragonRealEstate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n=== Training Refined Dragon Real Estate Pipeline ===")
        cls.engine = DragonRealEstateEngine()
        cls.metrics = cls.engine.train_and_evaluate()
        print("Model Performance Metrics:", cls.metrics)
        assert cls.metrics["r2_score"] > 0.65, f"R² score too low: {cls.metrics['r2_score']}"
        assert cls.metrics["mape_pct"] < 20.0, f"MAPE too high: {cls.metrics['mape_pct']}"

    def test_property_valuation_and_roi(self):
        print("\n=== Testing Real Estate Valuation & Investment ROI ===")
        sample_prop = {
            "CRIM": 0.08,
            "ZN": 20.0,
            "INDUS": 5.5,
            "CHAS": 1,
            "NOX": 0.45,
            "RM": 7.2,
            "AGE": 25.0,
            "DIS": 4.5,
            "RAD": 4,
            "TAX": 280,
            "PTRATIO": 15.0,
            "B": 390.0,
            "LSTAT": 4.5
        }
        res = self.engine.predict_valuation_and_roi(sample_prop)
        print("Valuation & ROI Result:", res)
        self.assertGreater(res["estimated_property_value_usd"], 25000.0)
        self.assertIn("cap_rate_pct", res)
        self.assertIn("investment_grade", res)
        self.assertIn("debt_service_coverage_ratio_dscr", res)
        self.assertIn(res["investment_grade"], ["TIER_A_STRONG_BUY", "TIER_B_MODERATE_HOLD", "TIER_C_OVERVALUED_LOW_YIELD"])

    def test_interest_rate_sensitivity(self):
        print("\n=== Testing Mortgage Interest Rate Sensitivity Matrix ===")
        sample_prop = {
            "CRIM": 0.05,
            "ZN": 25.0,
            "INDUS": 4.0,
            "CHAS": 0,
            "NOX": 0.42,
            "RM": 6.8,
            "AGE": 15.0,
            "DIS": 5.2,
            "RAD": 3,
            "TAX": 240,
            "PTRATIO": 14.5,
            "B": 395.0,
            "LSTAT": 5.0
        }
        rates = [5.0, 6.0, 7.0, 8.0]
        matrix = self.engine.calculate_interest_rate_sensitivity(sample_prop, rates=rates)
        self.assertEqual(len(matrix), 4)
        print("Sensitivity Matrix Sample:", matrix[0])
        # As interest rates rise, monthly mortgage payment must strictly increase
        self.assertGreater(matrix[3]["monthly_mortgage_usd"], matrix[0]["monthly_mortgage_usd"])
        # As debt service rises, net cash flow should decrease
        self.assertLess(matrix[3]["net_annual_cash_flow_usd"], matrix[0]["net_annual_cash_flow_usd"])

    def test_top_feature_drivers(self):
        print("\n=== Testing Feature Importance Drivers ===")
        self.assertIn("top_drivers", self.metrics)
        self.assertGreater(len(self.metrics["top_drivers"]), 5)
        print("Top 3 Valuation Drivers:", list(self.metrics["top_drivers"].items())[:3])

if __name__ == "__main__":
    unittest.main()
