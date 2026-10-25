import math
import numpy as np
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

    def run_monte_carlo_simulation(
        self,
        predicted_daily_demand: float,
        demand_std_dev: float,
        current_inventory: int,
        lead_time_days: int = 5,
        n_simulations: int = 1000,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Runs Monte Carlo empirical simulations across lead-time demand cycles
        to quantify exact probabilistic stockout distributions.
        """
        np.random.seed(random_state)
        # Simulate daily demands across lead time
        simulated_daily = np.random.normal(
            loc=predicted_daily_demand,
            scale=max(1.0, demand_std_dev),
            size=(n_simulations, lead_time_days)
        ).clip(0, None)

        total_lead_time_demands = np.sum(simulated_daily, axis=1)
        stockouts = total_lead_time_demands > current_inventory
        empirical_stockout_prob = float(np.mean(stockouts))

        p95_demand = float(np.percentile(total_lead_time_demands, 95))
        p99_demand = float(np.percentile(total_lead_time_demands, 99))

        # Generate 10-bin histogram for dashboard visualization
        counts, bin_edges = np.histogram(total_lead_time_demands, bins=10)
        bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(counts))]

        return {
            "empirical_stockout_probability": round(empirical_stockout_prob, 4),
            "p95_lead_time_demand": round(p95_demand, 1),
            "p99_lead_time_demand": round(p99_demand, 1),
            "mean_simulated_demand": round(float(np.mean(total_lead_time_demands)), 1),
            "std_simulated_demand": round(float(np.std(total_lead_time_demands)), 1),
            "histogram_labels": bin_labels,
            "histogram_counts": [int(c) for c in counts],
            "percentiles": {
                "p10": round(float(np.percentile(total_lead_time_demands, 10)), 1),
                "p25": round(float(np.percentile(total_lead_time_demands, 25)), 1),
                "p50_median": round(float(np.percentile(total_lead_time_demands, 50)), 1),
                "p75": round(float(np.percentile(total_lead_time_demands, 75)), 1),
                "p90": round(float(np.percentile(total_lead_time_demands, 90)), 1),
                "p95": round(p95_demand, 1),
                "p99": round(p99_demand, 1)
            }
        }

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
        Calculates analytical safety stock, reorder point (ROP), economic order quantity (EOQ),
        and runs Monte Carlo empirical risk simulations.
        """
        safety_stock = math.ceil(self.z * demand_std_dev * math.sqrt(lead_time_days))
        lead_time_demand = math.ceil(predicted_daily_demand * lead_time_days)
        reorder_point = lead_time_demand + safety_stock

        annual_demand = predicted_daily_demand * 365
        annual_holding_cost_per_unit = unit_cost * holding_cost_annual_pct

        if annual_holding_cost_per_unit > 0:
            eoq = math.ceil(math.sqrt((2 * annual_demand * ordering_cost_fixed) / annual_holding_cost_per_unit))
        else:
            eoq = math.ceil(lead_time_demand * 2)

        needs_reorder = current_on_hand_inventory <= reorder_point
        recommended_order_quantity = eoq if needs_reorder else 0

        # Monte Carlo Risk Quantification
        mc_results = self.run_monte_carlo_simulation(
            predicted_daily_demand=predicted_daily_demand,
            demand_std_dev=demand_std_dev,
            current_inventory=current_on_hand_inventory,
            lead_time_days=lead_time_days
        )

        stockout_prob = mc_results["empirical_stockout_probability"]
        if stockout_prob < 0.05:
            stockout_risk_level = "LOW (< 5%)"
        elif stockout_prob < 0.25:
            stockout_risk_level = "MODERATE (5% - 25%)"
        else:
            stockout_risk_level = "CRITICAL (High Risk of Stockout)"

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
            "monte_carlo_simulation": mc_results,
            "action_item": (
                f"Trigger PO for {recommended_order_quantity} units immediately!"
                if needs_reorder else "Inventory levels sufficient; no action needed."
            )
        }
