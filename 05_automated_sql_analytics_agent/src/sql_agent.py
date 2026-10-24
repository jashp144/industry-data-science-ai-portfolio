import re
from typing import Dict, Any, Tuple
import pandas as pd
from src.db_manager import DatabaseManager

class SQLAnalyticsAgent:
    def __init__(self, db_manager: DatabaseManager = None):
        if db_manager is None:
            db_manager = DatabaseManager()
        self.db_manager = db_manager

    def validate_sql_safety(self, sql_query: str) -> Tuple[bool, str]:
        """Strict guardrail ensuring only read-only safe SELECT queries execute."""
        forbidden_keywords = ["DROP", "DELETE", "TRUNCATE", "UPDATE", "INSERT", "ALTER", "CREATE", "REPLACE", "EXEC"]
        # Match standalone forbidden words
        for kw in forbidden_keywords:
            if re.search(rf"\b{kw}\b", sql_query, re.IGNORECASE):
                return False, f"Query rejected: Mutating keyword '{kw}' detected. Only read-only queries are permitted."

        clean_sql = sql_query.strip().upper()
        if not (clean_sql.startswith("SELECT") or clean_sql.startswith("WITH")):
            return False, "Query rejected: Query must start with SELECT or WITH."

        return True, "Valid safe query."

    def translate_question_to_sql(self, natural_language_query: str) -> str:
        """
        Translates business natural language questions to advanced SQL with Window functions and CTEs.
        """
        q = natural_language_query.lower()

        # 1. MoM Revenue Growth with LAG() Window Function and CTE
        if "month" in q or "growth" in q or "trend" in q:
            return """
                WITH MonthlySales AS (
                    SELECT 
                        strftime('%Y-%m', o.order_date) AS sales_month,
                        ROUND(SUM(oi.quantity * oi.unit_price * (1 - oi.discount)), 2) AS total_revenue
                    FROM orders o
                    JOIN order_items oi ON o.order_id = oi.order_id
                    WHERE o.status != 'Cancelled'
                    GROUP BY sales_month
                )
                SELECT 
                    sales_month,
                    total_revenue,
                    LAG(total_revenue, 1) OVER (ORDER BY sales_month) AS prev_month_revenue,
                    ROUND(
                        ((total_revenue - LAG(total_revenue, 1) OVER (ORDER BY sales_month)) / 
                        LAG(total_revenue, 1) OVER (ORDER BY sales_month)) * 100.0, 2
                    ) AS mom_growth_pct
                FROM MonthlySales
                ORDER BY sales_month ASC;
            """

        # 2. Top Spending Customers with DENSE_RANK() Window Function
        elif "top customer" in q or "spender" in q or "highest spending" in q or "customer rank" in q:
            return """
                WITH CustomerSpend AS (
                    SELECT 
                        c.customer_id,
                        c.name,
                        c.region,
                        c.segment,
                        ROUND(SUM(oi.quantity * oi.unit_price * (1 - oi.discount)), 2) AS total_spend,
                        COUNT(DISTINCT o.order_id) AS total_orders
                    FROM customers c
                    JOIN orders o ON c.customer_id = o.customer_id
                    JOIN order_items oi ON o.order_id = oi.order_id
                    WHERE o.status != 'Cancelled'
                    GROUP BY c.customer_id, c.name, c.region, c.segment
                )
                SELECT 
                    customer_id,
                    name,
                    region,
                    segment,
                    total_spend,
                    total_orders,
                    DENSE_RANK() OVER (ORDER BY total_spend DESC) AS spend_rank
                FROM CustomerSpend
                LIMIT 10;
            """

        # 3. Product Category Profitability & Contribution
        elif "product" in q or "category" in q or "profit" in q or "margin" in q:
            return """
                SELECT 
                    p.category,
                    COUNT(DISTINCT p.product_id) AS product_count,
                    SUM(oi.quantity) AS total_units_sold,
                    ROUND(SUM(oi.quantity * oi.unit_price * (1 - oi.discount)), 2) AS total_revenue,
                    ROUND(SUM(oi.quantity * (oi.unit_price * (1 - oi.discount) - p.cost_price)), 2) AS total_gross_profit,
                    ROUND(
                        (SUM(oi.quantity * (oi.unit_price * (1 - oi.discount) - p.cost_price)) / 
                        SUM(oi.quantity * oi.unit_price * (1 - oi.discount))) * 100.0, 2
                    ) AS profit_margin_pct
                FROM products p
                JOIN order_items oi ON p.product_id = oi.product_id
                JOIN orders o ON oi.order_id = o.order_id
                WHERE o.status != 'Cancelled'
                GROUP BY p.category
                ORDER BY total_gross_profit DESC;
            """

        # Default: Regional Performance Summary
        else:
            return """
                SELECT 
                    c.region,
                    c.segment,
                    COUNT(DISTINCT o.order_id) AS order_volume,
                    ROUND(SUM(oi.quantity * oi.unit_price * (1 - oi.discount)), 2) AS regional_revenue
                FROM customers c
                JOIN orders o ON c.customer_id = o.customer_id
                JOIN order_items oi ON o.order_id = oi.order_id
                WHERE o.status != 'Cancelled'
                GROUP BY c.region, c.segment
                ORDER BY regional_revenue DESC;
            """

    def process_query(self, user_question: str) -> Dict[str, Any]:
        sql_query = self.translate_question_to_sql(user_question)
        is_safe, msg = self.validate_sql_safety(sql_query)
        if not is_safe:
            return {
                "question": user_question,
                "status": "REJECTED",
                "error": msg,
                "sql": sql_query
            }

        df = self.db_manager.execute_query(sql_query)
        return {
            "question": user_question,
            "status": "SUCCESS",
            "sql_query": sql_query.strip(),
            "row_count": len(df),
            "data": df.to_dict(orient="records"),
            "dataframe": df
        }
