import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag_engine import RAGEngine

class TestEnterpriseRAG(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n=== Initializing RAG Knowledge Base ===")
        cls.rag = RAGEngine()
        n_docs, n_chunks = cls.rag.initialize_knowledge_base()
        print(f"Indexed {n_docs} documents across {n_chunks} chunks.")
        assert n_docs > 0, "No documents indexed."
        assert n_chunks > 0, "No chunks created."

    def test_refund_policy_retrieval(self):
        print("\n=== Testing In-Domain Query: Refund Policy ===")
        query = "What is the return window and how many days to get refund?"
        res = self.rag.query(query)
        print("Query:", query)
        print("Answer:", res["answer"])
        print("Confidence Score:", res["confidence_score"])
        print("Sources:", res["sources"])
        self.assertEqual(res["guardrail_status"], "VERIFIED_ACCURATE")
        self.assertGreater(res["confidence_score"], 0.15)
        self.assertTrue(any("refund_policy.md" in s for s in res["sources"]))

    def test_sla_uptime_retrieval(self):
        print("\n=== Testing In-Domain Query: SLA & Outage ===")
        query = "What is the monthly uptime percentage SLA guarantee and credit?"
        res = self.rag.query(query)
        print("Answer:", res["answer"])
        self.assertEqual(res["guardrail_status"], "VERIFIED_ACCURATE")
        self.assertTrue(any("sla_agreement.md" in s for s in res["sources"]))

    def test_security_standards_retrieval(self):
        print("\n=== Testing In-Domain Query: Encryption & Security ===")
        query = "What encryption standard is used for data at rest and transit?"
        res = self.rag.query(query)
        print("Answer:", res["answer"])
        self.assertEqual(res["guardrail_status"], "VERIFIED_ACCURATE")
        self.assertTrue(any("security_compliance.md" in s for s in res["sources"]))

    def test_hallucination_guardrail_refusal(self):
        print("\n=== Testing Out-of-Domain Query: Hallucination Guardrail ===")
        irrelevant_query = "What is the best recipe to make Italian lasagna and bake pasta?"
        res = self.rag.query(irrelevant_query)
        print("Query:", irrelevant_query)
        print("Status:", res["guardrail_status"])
        print("Answer:", res["answer"])
        self.assertEqual(res["guardrail_status"], "REFUSAL_UNVERIFIED_CONTEXT")
        self.assertIn("does not contain sufficient verified context", res["answer"])

if __name__ == "__main__":
    unittest.main()
