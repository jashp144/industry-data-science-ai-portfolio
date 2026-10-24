import subprocess
import sys
import os

projects = [
    ("01 E-Commerce Churn & LTV Pipeline", "01_ecommerce_churn_ltv_pipeline/test_pipeline.py"),
    ("02 Enterprise RAG Knowledge Engine", "02_enterprise_rag_knowledge_engine/test_rag.py"),
    ("03 Financial Fraud Detection System", "03_financial_fraud_detection_system/test_fraud.py"),
    ("04 Retail Demand Forecasting Engine", "04_retail_demand_forecasting_engine/test_forecasting.py"),
    ("05 Automated SQL Analytics Agent", "05_automated_sql_analytics_agent/test_sql_agent.py"),
    ("06A CWH Advanced Dragon Real Estate", "06_codewithharry_course_advanced_suite/dragon_real_estate/test_dragon.py"),
    ("06B CWH Advanced Coders Labor Market", "06_codewithharry_course_advanced_suite/coders_analytics/test_talent.py"),
]

base_dir = os.path.dirname(os.path.abspath(__file__))

print("==========================================================")
print("  EXECUTING ALL DATA SCIENCE & AI PORTFOLIO TEST SUITES")
print("==========================================================")

all_passed = True
results = []

for name, script_rel_path in projects:
    script_full_path = os.path.join(base_dir, script_rel_path)
    print(f"\n---> Running: {name}")
    res = subprocess.run([sys.executable, script_full_path], cwd=os.path.dirname(script_full_path), capture_output=True, text=True)
    
    if res.returncode == 0:
        print(f" [PASS] {name}")
        results.append((name, "PASSED"))
    else:
        print(f" [FAIL] {name}")
        print(res.stderr)
        all_passed = False
        results.append((name, "FAILED"))

print("\n==========================================================")
print("  PORTFOLIO VERIFICATION SUMMARY")
print("==========================================================")
for name, status in results:
    print(f"{name:.<45} {status}")

if all_passed:
    print("\nALL 7 SYSTEMS VERIFIED AND 100% OPERATIONAL!")
else:
    print("\nSome tests failed. Review logs above.")
