"""
Unit tests for AspectClassifier and EmotionClassifier.
Tests label vectorization, decoding, loss metrics, and taxonomy mapping persistence.
"""

import tempfile
import unittest
from pathlib import Path

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

from src.classification.aspect_classifier import DEFAULT_ASPECT_TAXONOMY, AspectClassifier
from src.classification.emotion_classifier import DEFAULT_EMOTIONS, EmotionClassifier

if HAS_TORCH:
    from training.train_aspect_classifier import compute_metrics
    from training.train_emotion_classifier import compute_multiclass_metrics


class TestClassification(unittest.TestCase):
    """Tests aspect and emotion classification utilities."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_path = Path(cls.temp_dir.name)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_aspect_encoding_and_decoding(self):
        classifier = AspectClassifier(checkpoint_dir=self.temp_path)
        vec = classifier.encode_labels(["Quality", "Usability"])
        self.assertEqual(len(vec), len(classifier.taxonomy))

        q_idx = classifier.aspect2id["Quality"]
        u_idx = classifier.aspect2id["Usability"]
        self.assertEqual(vec[q_idx], 1.0)
        self.assertEqual(vec[u_idx], 1.0)

        # Decode simulated probabilities
        probs = [0.1] * len(classifier.taxonomy)
        probs[q_idx] = 0.92
        probs[u_idx] = 0.85
        preds = classifier.decode_predictions(probs, threshold=0.5)

        aspect_names = [p["aspect"] for p in preds]
        self.assertIn("Quality", aspect_names)
        self.assertIn("Usability", aspect_names)
        self.assertEqual(len(preds), 2)

    @unittest.skipUnless(HAS_TORCH, "PyTorch required for tensor metric testing")
    def test_aspect_compute_metrics(self):
        # 2 samples, 3 classes
        preds = torch.tensor([[0.9, 0.1, 0.8], [0.2, 0.7, 0.1]])
        targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        metrics = compute_metrics(preds, targets, threshold=0.5)

        self.assertIn("f1_micro", metrics)
        self.assertIn("f1_macro", metrics)
        self.assertAlmostEqual(metrics["f1_micro"], 1.0, places=2)
        self.assertAlmostEqual(metrics["f1_macro"], 1.0, places=2)

    @unittest.skipUnless(HAS_TORCH, "PyTorch required for tensor metric testing")
    def test_emotion_compute_metrics(self):
        # 4 samples, 4 classes
        preds = torch.tensor([0, 1, 2, 3])
        targets = torch.tensor([0, 1, 2, 0])  # last one incorrect
        metrics = compute_multiclass_metrics(preds, targets, num_classes=4)

        self.assertIn("accuracy", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertEqual(metrics["accuracy"], 0.75)

    def test_save_mappings(self):
        aspect_clf = AspectClassifier(checkpoint_dir=self.temp_path / "aspect")
        asp_map = aspect_clf.save_mapping()
        self.assertTrue(asp_map.exists())

        emotion_clf = EmotionClassifier(checkpoint_dir=self.temp_path / "emotion")
        emo_map = emotion_clf.save_mapping()
        self.assertTrue(emo_map.exists())


if __name__ == "__main__":
    unittest.main()
