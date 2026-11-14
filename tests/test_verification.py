"""
Unit tests for claim extraction, fact checking, and evidence grounding verification.
"""

import unittest

from src.verification.claim_extractor import ClaimExtractor
from src.verification.fact_checker import FactChecker
from src.verification.grounding_pipeline import GroundingPipeline


class TestVerification(unittest.TestCase):
    """Tests verification and factual grounding modules."""

    def test_claim_extraction(self):
        text = "The user wants complaint that the bottle cap is loose and water leaks out. The plastic smells like chemicals."
        claims = ClaimExtractor.extract_claims(text)

        self.assertGreaterEqual(len(claims), 2)
        # Verify conversational boilerplate removed
        for c in claims:
            self.assertFalse(c.lower().startswith("the user wants complaint that"))

    def test_fact_checking_visual_and_unsupported(self):
        checker = FactChecker()

        visual_obs = [
            {"timestamp": 3.2, "observation": "water leaks out from the bottle cap", "confidence": 0.95}
        ]
        customer_text = "Not recommended, product is poor quality."

        # Verified visual claim
        res_visual = checker.check_claim(
            claim="water leaks out",
            customer_text=customer_text,
            visual_evidence=visual_obs,
        )
        self.assertEqual(res_visual["status"], "VERIFIED_VISUAL")
        self.assertEqual(res_visual["timestamp"], 3.2)

        # Unsupported hallucinated claim
        res_unsupported = checker.check_claim(
            claim="the screen exploded into flames",
            customer_text=customer_text,
            visual_evidence=visual_obs,
        )
        self.assertEqual(res_unsupported["status"], "UNSUPPORTED")

    def test_grounding_pipeline_score_calculation(self):
        pipeline = GroundingPipeline()

        visual_obs = [
            {"timestamp": 1.5, "observation": "the bottle cap is cracked and loose", "confidence": 0.9}
        ]
        complaint = "The bottle cap is cracked. The phone display is completely shattered."

        res = pipeline.verify_complaint(
            generated_complaint=complaint,
            customer_text="Broken cap",
            visual_evidence=visual_obs,
        )

        self.assertEqual(res["total_claims"], 2)
        self.assertEqual(res["verified_count"], 1)
        self.assertEqual(res["unsupported_count"], 1)
        self.assertEqual(res["evidence_grounding_score"], 0.5)
        self.assertIn("The bottle cap is cracked", res["revised_complaint"])
        self.assertNotIn("The phone display is completely shattered", res["revised_complaint"])


if __name__ == "__main__":
    unittest.main()
