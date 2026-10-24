import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.forecaster import DemandForecaster
from src.inventory_optimizer import InventoryOptimizer

class TestDemandForecasting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n=== Training Demand Forecasting Engine ===")
        cls.forecaster = DemandForecaster()
        cls.metrics = cls.forecaster.train()
        print("Forecasting Accuracy Metrics:", cls.metrics)
        assert cls.metrics["wape"] < 0.25, f"WAPE error too high: {cls.metrics['wape']}"
        assert cls.metrics["mae"] < 25.0, f"MAE error too high: {cls.metrics['mae']}"

    def test_forecast_prediction(self):
        print("\n=== Testing Single Period Forecast ===")
        sample_features = {
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
            "rolling_mean_14": 102.0
        }
        pred = self.forecaster.forecast_next_period(sample_features)
        print("Predicted Daily Demand (Units):", pred)
        self.assertGreater(pred, 50.0)
        self.assertLess(pred, 300.0)

    def test_inventory_replenishment_logic(self):
        print("\n=== Testing Safety Stock & Reorder Point Logic ===")
        optimizer = InventoryOptimizer(service_level=0.95)
        
        # Case A: Low inventory -> should trigger reorder
        plan_reorder = optimizer.calculate_replenishment_plan(
            predicted_daily_demand=100.0,
            demand_std_dev=15.0,
            current_on_hand_inventory=300,
            lead_time_days=5
        )
        print("Reorder Needed Plan:", plan_reorder)
        self.assertTrue(plan_reorder["needs_reorder"])
        self.assertGreater(plan_reorder["recommended_order_quantity"], 0)

        # Case B: High inventory -> should NOT trigger reorder
        plan_healthy = optimizer.calculate_replenishment_plan(
            predicted_daily_demand=100.0,
            demand_std_dev=15.0,
            current_on_hand_inventory=1200,
            lead_time_days=5
        )
        print("Healthy Inventory Plan:", plan_healthy)
        self.assertFalse(plan_healthy["needs_reorder"])
        self.assertEqual(plan_healthy["recommended_order_quantity"], 0)

if __name__ == "__main__":
    unittest.main()
