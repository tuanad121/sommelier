import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / "podcast-pipeline"


class AudioPreprocessingTests(unittest.TestCase):
    def test_stage_audio_preprocessing_clamps_gain_to_configurable_db(self):
        text = (PIPELINE_DIR / "utils" / "audio_preprocessing.py").read_text(encoding="utf-8")

        self.assertIn("AUDIO_GAIN_CLAMP_DB", text)
        self.assertIn("gain_clamp_db = abs(float(", text)
        self.assertIn("min(max(gain, -gain_clamp_db), gain_clamp_db)", text)
        self.assertNotIn("min(max(gain, -3), 3)", text)
        self.assertNotIn("min(max(gain, -6), 6)", text)

    def test_legacy_main_audio_preprocessing_clamps_gain_to_six_db(self):
        text = (PIPELINE_DIR / "main_original_ASR_MoE.py").read_text(encoding="utf-8")

        self.assertIn("min(max(gain, -6), 6)", text)
        self.assertNotIn("min(max(gain, -3), 3)", text)


if __name__ == "__main__":
    unittest.main()
