"""
Unit tests for evaluation metrics:
- Composite Complaint Retention (CR) score: CR = (S_VADER^N + ES + AS) / 3
- Normalized VADER sentiment
- Emotion score
- Aspect score
- Grounding score & Temporal alignment
"""

import unittest
from evaluation.metrics import (
    calculate_aspect_score,
    calculate_composite_cr,
    calculate_cr,
    calculate_emotion_score,
    calculate_evidence_grounding_score,
    calculate_temporal_alignment_score,
    calculate_text_overlap,
    calculate_vader_compound,
    normalize_vader_sentiment,
)


class TestEvaluationMetrics(unittest.TestCase):

    def test_vader_compound_negative_and_normalized(self):
        # Strongly negative complaint text
        complaint_text = "The product is broken, defective, and damaged. Terrible quality!"
        compound = calculate_vader_compound(complaint_text)
        self.assertLess(compound, 0.0)

        # S_VADER^N = (1 - compound) / 2 should be > 0.5 for complaints
        s_norm = normalize_vader_sentiment(compound)
        self.assertGreater(s_norm, 0.5)
        self.assertLessEqual(s_norm, 1.0)

    def test_emotion_score_exact_and_related(self):
        # Exact match
        self.assertEqual(calculate_emotion_score("anger", "anger"), 1.0)

        # Semantically related negative complaint emotions
        self.assertEqual(calculate_emotion_score("frustration", "anger"), 0.7)
        self.assertEqual(calculate_emotion_score("sadness", "disappointment"), 0.7)

        # Mismatch
        self.assertEqual(calculate_emotion_score("joy", "anger"), 0.0)

    def test_aspect_score_calculation(self):
        gt_aspects = ["display", "battery", "speaker"]
        pred_aspects = ["display", "battery"]
        as_score = calculate_aspect_score(pred_aspects, gt_aspects)
        self.assertAlmostEqual(as_score, 2 / 3, places=2)

    def test_composite_cr_formula(self):
        # CR = (S_VADER^N + ES + AS) / 3
        text = "Extremely disappointed. The screen arrived completely cracked and scratched."
        gt_aspects = ["screen", "material"]
        pred_aspects = ["screen"]  # AS = 0.5
        gt_emotion = "disappointment"
        pred_emotion = "disappointment"  # ES = 1.0

        res = calculate_composite_cr(
            text=text,
            target_aspects=gt_aspects,
            predicted_aspects=pred_aspects,
            target_emotion=gt_emotion,
            predicted_emotion=pred_emotion,
        )

        self.assertIn("cr_score", res)
        self.assertIn("s_vader_norm", res)
        self.assertIn("emotion_score", res)
        self.assertIn("aspect_score", res)

        expected_cr = round((res["s_vader_norm"] + res["emotion_score"] + res["aspect_score"]) / 3.0, 4)
        self.assertAlmostEqual(res["cr_score"], expected_cr, places=4)
        self.assertEqual(res["emotion_score"], 1.0)
        self.assertEqual(res["aspect_score"], 0.5)

    def test_backward_compatibility_calculate_cr(self):
        # Without text/emotion, calculate_cr behaves as aspect score
        gt_aspects = ["camera", "zoom"]
        pred_aspects = ["camera", "zoom"]
        self.assertEqual(calculate_cr(pred_aspects, gt_aspects), 1.0)

        # With text/emotion, calculate_cr returns composite CR
        cr_comp = calculate_cr(
            predicted_aspects=["camera"],
            ground_truth_aspects=["camera", "zoom"],
            text="Horrible quality, defective lens.",
            predicted_emotion="anger",
            ground_truth_emotion="anger",
        )
        self.assertIsInstance(cr_comp, float)
        self.assertGreater(cr_comp, 0.0)


if __name__ == "__main__":
    unittest.main()
