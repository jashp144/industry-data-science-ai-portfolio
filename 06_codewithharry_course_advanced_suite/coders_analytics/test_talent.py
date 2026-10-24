import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.talent_analytics import CodersLaborMarketEngine

class TestCodersAnalytics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n=== Training Coders of Delhi & Bangalore Labor Market Engine ===")
        cls.engine = CodersLaborMarketEngine()
        cls.metrics = cls.engine.train()
        print("Model Performance Metrics:", cls.metrics)
        assert cls.metrics["r2_score"] > 0.70, f"R² score too low: {cls.metrics['r2_score']}"

    def test_profile_analysis_and_uplift(self):
        print("\n=== Testing Candidate Salary & Skill Uplift Recommendation ===")
        candidate = {
            "city": "Bangalore",
            "experience_years": 2.5,
            "primary_stack": "Python/AI",
            "has_cloud_skill": 0,
            "has_genai_skill": 0,
            "company_tier": "Product/Unicorn"
        }
        res = self.engine.analyze_profile_and_recommend(candidate)
        print("Candidate Analysis Result:", res)
        self.assertGreater(res["predicted_salary_lpa"], 5.0)
        self.assertGreater(res["total_potential_uplift_pct"], 15.0)
        self.assertGreater(len(res["strategic_skill_recommendations"]), 0)

if __name__ == "__main__":
    unittest.main()
