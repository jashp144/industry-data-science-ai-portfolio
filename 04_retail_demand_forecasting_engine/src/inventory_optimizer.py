import math
from typing import Dict, Any

class InventoryOptimizer:
    def __init__(self, service_level: float = 0.95):
        """
        Z-score mappings for target service level (probability of not stocking out):
        90% -> 1.28
        95% -> 1.645
        98% -> 2.05
        99% -> 2.33
        """
        self.service_level = service_level
        z_scores = {0.90: 1.28, 0.95: 1.645, 0.98: 2.05, 0.99: 2.33}
        self.z = z_scores.get(service_level, 1.645)

    def calculate_replenishment_plan(
        self,
        predicted_daily_demand: float,
        demand_std_dev: float,
        current_on_hand_inventory: int,
        lead_time_days: int = 5,
        ordering_cost_fixed: float = 50.0,
        holding_cost_annual_pct: float = 0.20,
        unit_cost: float = 30.0
    ) -> Dict[str, Any]:
        """
        Calculates optimal safety stock, reorder point (ROP), and automated restocking purchase quantity.
        """
        # Safety Stock formula: SS = Z * sigma_demand * sqrt(lead_time)
        safety_stock = math.ceil(self.z * demand_std_dev * math.sqrt(lead_time_days))
        
        # Lead time demand
        lead_time_demand = math.ceil(predicted_daily_demand * lead_time_days)
        
        # Reorder Point (ROP): ROP = Lead Time Demand + Safety Stock
        reorder_point = lead_time_demand + safety_stock

        # Annual demand estimate (365 days)
        annual_demand = predicted_daily_demand * 365
        annual_holding_cost_per_unit = unit_cost * holding_cost_annual_pct

        # Economic Order Quantity (EOQ): sqrt((2 * D * S) / H)
        if annual_holding_cost_per_unit > 0:
            eoq = math.ceil(math.sqrt((2 * annual_demand * ordering_cost_fixed) / annual_holding_cost_per_unit))
        else:
            eoq = math.ceil(lead_time_demand * 2)

        # Restock Decision
        needs_reorder = current_on_hand_inventory <= reorder_point
        recommended_order_quantity = eoq if needs_reorder else 0

        # Estimated stockout probability given current inventory
        if current_on_hand_inventory >= reorder_point:
            stockout_risk_level = "LOW (< 5%)"
        elif current_on_hand_inventory > lead_time_demand:
            stockout_risk_level = "MODERATE (5% - 20%)"
        else:
            stockout_risk_level = "CRITICAL (> 50% imminent stockout)"

        return {
            "predicted_daily_demand": round(predicted_daily_demand, 1),
            "lead_time_days": lead_time_days,
            "current_inventory": current_on_hand_inventory,
            "safety_stock": safety_stock,
            "reorder_point": reorder_point,
            "needs_reorder": needs_reorder,
            "recommended_order_quantity": recommended_order_quantity,
            "economic_order_quantity_eoq": eoq,
            "stockout_risk": stockout_risk_level,
            "action_item": (
                f"Trigger PO for {recommended_order_quantity} units immediately!"
                if needs_reorder else "Inventory levels sufficient; no action needed."
            )
        }
