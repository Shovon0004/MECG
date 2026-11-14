"""
Unit tests for LanguageDetector and NLLBTranslator.
Validates script detection, language code conversions, no-op translation, and caching.
"""

import tempfile
import unittest
from pathlib import Path

from src.multilingual.language_detector import LanguageDetector
from src.multilingual.translator import NLLBTranslator


class TestMultilingual(unittest.TestCase):
    """Tests language detection and translator caching mechanisms."""

    @classmethod
    def setUpClass(cls):
        cls.detector = LanguageDetector()
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.cache_file = Path(cls.temp_dir.name) / "test_cache.json"

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_language_detection_scripts(self):
        # English
        en_res = self.detector.detect_language("The plastic bottle has a strong chemical smell.")
        self.assertEqual(en_res["iso_code"], "en")
        self.assertEqual(en_res["nllb_code"], "eng_Latn")

        # Bengali
        bn_res = self.detector.detect_language("প্লাস্টিকের বোতল থেকে তীব্র রাসায়নিক গন্ধ বের হচ্ছে।")
        self.assertEqual(bn_res["iso_code"], "bn")
        self.assertEqual(bn_res["nllb_code"], "ben_Beng")

        # Hindi
        hi_res = self.detector.detect_language("स्क्रीन काम नहीं कर रही है और बार-बार झिलमिला रही है।")
        self.assertEqual(hi_res["iso_code"], "hi")
        self.assertEqual(hi_res["nllb_code"], "hin_Deva")

        # Tamil
        ta_res = self.detector.detect_language("இந்த தயாரிப்பு மிகவும் மோசமாக உள்ளது.")
        self.assertEqual(ta_res["iso_code"], "ta")
        self.assertEqual(ta_res["nllb_code"], "tam_Taml")

        # Telugu
        te_res = self.detector.detect_language("ఈ ఉత్పత్తి సరిగ్గా పని చేయడం లేదు.")
        self.assertEqual(te_res["iso_code"], "te")
        self.assertEqual(te_res["nllb_code"], "tel_Telu")

    def test_iso_to_nllb_mapping(self):
        self.assertEqual(self.detector.iso_to_nllb("en"), "eng_Latn")
        self.assertEqual(self.detector.iso_to_nllb("bn"), "ben_Beng")
        self.assertEqual(self.detector.iso_to_nllb("hi"), "hin_Deva")
        self.assertEqual(self.detector.nllb_to_iso("ben_Beng"), "bn")
        self.assertEqual(self.detector.nllb_to_iso("hin_Deva"), "hi")

    def test_translator_no_op_same_language(self):
        # When source == target, should return original text without loading model
        translator = NLLBTranslator(
            default_target_lang="eng_Latn",
            cache_path=self.cache_file,
        )
        text = "This complaint is already in English."
        res = translator.translate(text, src_lang="en", tgt_lang="en")

        self.assertEqual(res["original_text"], text)
        self.assertEqual(res["translated_text"], text)
        self.assertFalse(res["cached"])
        # Ensure model was not instantiated
        self.assertIsNone(translator.model)

    def test_translator_caching_mechanism(self):
        translator = NLLBTranslator(
            default_target_lang="eng_Latn",
            cache_path=self.cache_file,
        )
        # Pre-seed cache
        key = translator._make_cache_key("বোতল ভাঙা", "ben_Beng", "eng_Latn")
        translator.cache[key] = "Broken bottle"
        translator._save_cache()

        res = translator.translate("বোতল ভাঙা", src_lang="bn", tgt_lang="en")
        self.assertTrue(res["cached"])
        self.assertEqual(res["translated_text"], "Broken bottle")


if __name__ == "__main__":
    unittest.main()
