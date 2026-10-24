import os
import glob
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Document:
    page_content: str
    metadata: Dict[str, str]

class DocumentLoader:
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(project_root, "data")
        self.data_dir = data_dir

    def load_directory(self) -> List[Document]:
        documents = []
        file_patterns = ["*.txt", "*.md"]
        
        for pattern in file_patterns:
            for file_path in glob.glob(os.path.join(self.data_dir, pattern)):
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        documents.append(
                            Document(
                                page_content=content,
                                metadata={
                                    "source": os.path.basename(file_path),
                                    "file_path": file_path
                                }
                            )
                        )
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        return documents

    def create_sample_knowledge_base(self):
        """Generates realistic enterprise policy and product documentation for demo testing."""
        os.makedirs(self.data_dir, exist_ok=True)

        docs = {
            "refund_policy.md": (
                "# CloudCommerce Global Refund & Return Policy\n\n"
                "Customers are eligible for a full refund within 30 days of purchase for all digital and physical products.\n"
                "To initiate a return, customers must submit their receipt and order ID via the support portal.\n"
                "Subscriptions cancelled within 14 days of renewal qualify for a prorated refund.\n"
                "Hardware products must be returned in original packaging with all accessories.\n"
                "Refund processing takes 3 to 5 business days to reflect on the original payment method."
            ),
            "sla_agreement.md": (
                "# CloudCommerce Service Level Agreement (SLA)\n\n"
                "We guarantee 99.95% monthly uptime for all Tier-1 enterprise cloud services.\n"
                "Scheduled maintenance occurs on Sundays between 02:00 UTC and 04:00 UTC and is excluded from uptime calculations.\n"
                "If monthly uptime drops between 99.0% and 99.95%, enterprise customers receive a 10% billing credit.\n"
                "If uptime falls below 99.0%, a 25% credit is automatically applied to the subsequent invoice.\n"
                "Incident response time for Severity-1 outages is strictly guaranteed within 15 minutes."
            ),
            "security_compliance.md": (
                "# CloudCommerce Security & Data Protection Standards\n\n"
                "All customer data at rest is encrypted using AES-256 GCM encryption.\n"
                "All data in transit is protected using TLS 1.3 with automated certificate rotation.\n"
                "The platform maintains SOC 2 Type II, ISO 27001, and GDPR compliance certifications.\n"
                "Access to production databases requires hardware MFA and just-in-time (JIT) role elevation.\n"
                "Automated vulnerability scans run daily, and third-party penetration testing is conducted biannually."
            )
        }

        for filename, content in docs.items():
            path = os.path.join(self.data_dir, filename)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        print(f"Created sample knowledge base in {self.data_dir}")
