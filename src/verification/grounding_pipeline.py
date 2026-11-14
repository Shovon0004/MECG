"""
Grounding verification pipeline for MECG.
Coordinates claim extraction, evidence matching, fact checking,
and calculation of the defined Evidence Grounding Score.
"""

from typing import Any, Dict, List, Optional

from src.verification.claim_extractor import ClaimExtractor
from src.verification.fact_checker import FactChecker


class GroundingPipeline:
    """Verifies generated complaints against evidence and calculates Evidence Grounding Score."""

    def __init__(
        self,
        claim_extractor: Optional[ClaimExtractor] = None,
        fact_checker: Optional[FactChecker] = None,
    ):
        self.claim_extractor = claim_extractor or ClaimExtractor()
        self.fact_checker = fact_checker or FactChecker()

    def verify_complaint(
        self,
        generated_complaint: str,
        customer_text: str = "",
        transcript: str = "",
        visual_evidence: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Decomposes generated complaint into claims and checks each against available evidence.
        Evidence Grounding Score is defined as:
            Score = (Number of Verified Claims) / (Total Number of Extracted Claims)
        """
        claims = self.claim_extractor.extract_claims(generated_complaint)
        if not claims:
            return {
                "evidence_grounding_score": 0.0 if not generated_complaint.strip() else 1.0,
                "total_claims": 0,
                "verified_count": 0,
                "unsupported_count": 0,
                "verified_claims": [],
                "unsupported_claims": [],
                "revised_complaint": generated_complaint,
            }

        verified_claims = []
        unsupported_claims = []

        for c in claims:
            res = self.fact_checker.check_claim(
                claim=c,
                customer_text=customer_text,
                transcript=transcript,
                visual_evidence=visual_evidence,
            )

            status = res["status"]
            if status in ("VERIFIED_VISUAL", "VERIFIED_TEXTUAL"):
                verified_claims.append(res)
            else:
                unsupported_claims.append(res)

        total = len(claims)
        v_count = len(verified_claims)
        grounding_score = round(v_count / total, 3) if total > 0 else 0.0

        # Revised complaint: keep only verified claims
        revised_complaint = ". ".join(v["claim"] for v in verified_claims)
        if revised_complaint:
            revised_complaint += "."

        return {
            "evidence_grounding_score": grounding_score,
            "total_claims": total,
            "verified_count": v_count,
            "unsupported_count": len(unsupported_claims),
            "verified_claims": verified_claims,
            "unsupported_claims": unsupported_claims,
            "revised_complaint": revised_complaint,
        }
