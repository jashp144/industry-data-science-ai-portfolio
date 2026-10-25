import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.db_manager import DatabaseManager
from src.sql_agent import SQLAnalyticsAgent
from src.insights_generator import ExecutiveInsightsGenerator

class TestSQLAnalyticsAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n=== Initializing & Seeding Enterprise SQL Database ===")
        cls.db = DatabaseManager()
        cls.db.initialize_and_seed_database()
        cls.agent = SQLAnalyticsAgent(cls.db)
        cls.insights_gen = ExecutiveInsightsGenerator()

    def test_mom_revenue_growth_query(self):
        print("\n=== Testing MoM Revenue Growth with LAG() Window Function ===")
        q = "Show me monthly revenue growth trend"
        res = self.agent.process_query(q)
        print("Generated SQL:\n", res["sql_query"])
        print(f"Execution Latency: {res['execution_latency_ms']} ms")
        self.assertEqual(res["status"], "SUCCESS")
        self.assertGreater(res["row_count"], 0)
        
        insights = self.insights_gen.generate_brief(res)
        print("Executive Key Findings:", insights["key_findings"])
        self.assertGreater(len(insights["key_findings"]), 0)

    def test_rfm_customer_segmentation_query(self):
        print("\n=== Testing RFM Segmentation with NTILE() Window Functions ===")
        q = "Run RFM customer segmentation and loyalty analysis"
        res = self.agent.process_query(q)
        print("Generated SQL:\n", res["sql_query"])
        self.assertEqual(res["status"], "SUCCESS")
        self.assertGreater(res["row_count"], 0)
        
        insights = self.insights_gen.generate_brief(res)
        print("RFM Executive Brief:", insights["key_findings"])
        self.assertTrue(any("RFM Segmentation" in f for f in insights["key_findings"]))

    def test_top_customer_dense_rank_query(self):
        print("\n=== Testing Customer Ranking with DENSE_RANK() Window Function ===")
        q = "Who are the highest spending top customers?"
        res = self.agent.process_query(q)
        print("Generated SQL:\n", res["sql_query"])
        self.assertEqual(res["status"], "SUCCESS")
        self.assertLessEqual(res["row_count"], 10)

    def test_sql_injection_rejection_guardrail(self):
        print("\n=== Testing SQL Injection Guardrail ===")
        malicious_sql = "DROP TABLE customers; SELECT * FROM products;"
        is_safe, msg = self.agent.validate_sql_safety(malicious_sql)
        print("Guardrail Safety Result:", is_safe, "| Message:", msg)
        self.assertFalse(is_safe)
        self.assertIn("Mutating keyword", msg)

if __name__ == "__main__":
    unittest.main()
