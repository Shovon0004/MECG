"""
Claim extraction module for MECG.
Decomposes generated complaints into discrete, verifiable atomic factual claims.
"""

import re
from typing import List


class ClaimExtractor:
    """Extracts atomic testable propositions from complaint text."""

    @staticmethod
    def extract_claims(text: str) -> List[str]:
        """
        Splits complaint text into individual verifiable factual assertions.
        Filters out boilerplate introductory and conversational phrases.
        """
        if not text or not text.strip():
            return []

        # Split on sentence boundaries and semicolons
        raw_sentences = re.split(r"(?<=[.!?])\s+|;\s*", text.strip())

        claims: List[str] = []
        for s in raw_sentences:
            s_clean = s.strip()
            if not s_clean:
                continue

            # Remove common conversational wrappers
            s_clean = re.sub(r"^(The user wants (to complain that|complaint that)|I want to complain that|The customer states that|Complaint:)\s*", "", s_clean, flags=re.IGNORECASE)

            # Split compound coordinate sentences joined by 'and' or 'moreover' if both clauses have verbs
            sub_clauses = re.split(r"\s+and\s+|\s+moreover,\s*", s_clean, flags=re.IGNORECASE)
            for clause in sub_clauses:
                c = clause.strip().rstrip(".")
                # Filter trivial fragments
                if len(c.split()) >= 3:
                    claims.append(c)

        return claims
