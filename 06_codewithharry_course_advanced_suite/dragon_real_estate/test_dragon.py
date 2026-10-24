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
        self.assertIn(res["investment_grade"], ["TIER_A_STRONG_BUY", "TIER_B_MODERATE_HOLD", "TIER_C_OVERVALUED_LOW_YIELD"])

if __name__ == "__main__":
    unittest.main()
