import unittest
from ui.presets import PRESET_SAMPLES, get_preset_choices, get_preset_text, get_preset_metadata


class TestUiPresets(unittest.TestCase):
    def test_all_six_presets_available(self):
        """Verifies exactly 6 benchmark presets exist in PRESET_SAMPLES."""
        self.assertEqual(len(PRESET_SAMPLES), 6)
        expected_keys = [
            "1. Authentic Kaspi Consumer Review (Human)",
            "2. Authentic Formal News Article (Human)",
            "3. Authentic Academic Wikipedia Article (Human)",
            "4. Sherkala-7B Generated Article (AI)",
            "5. Qwen-2.5-7B-Instruct Wild Sample (AI)",
            "6. Injected Hybrid Essay (Partially AI)",
        ]
        for key in expected_keys:
            self.assertIn(key, PRESET_SAMPLES)
            item = PRESET_SAMPLES[key]
            self.assertIn("text", item)
            self.assertIn("domain", item)
            self.assertIn("expected_verdict", item)
            self.assertIn("generator", item)
            # Must be authentic Kazakh text > 20 characters
            self.assertGreater(len(item["text"].strip()), 20)
            self.assertGreater(len(item["domain"].strip()), 0)
            self.assertGreater(len(item["expected_verdict"].strip()), 0)

    def test_preset_choices_order(self):
        """Verifies get_preset_choices returns all 6 keys in exact sequential order."""
        choices = get_preset_choices()
        self.assertEqual(len(choices), 6)
        self.assertTrue(choices[0].startswith("1."))
        self.assertTrue(choices[1].startswith("2."))
        self.assertTrue(choices[2].startswith("3."))
        self.assertTrue(choices[3].startswith("4."))
        self.assertTrue(choices[4].startswith("5."))
        self.assertTrue(choices[5].startswith("6."))

    def test_preset_text_helper(self):
        """Verifies get_preset_text retrieves authentic text for valid and invalid keys."""
        choices = get_preset_choices()
        for choice in choices:
            text = get_preset_text(choice)
            self.assertIsInstance(text, str)
            self.assertGreater(len(text), 20)

        # Nonexistent preset
        self.assertEqual(get_preset_text("Nonexistent Preset Title"), "")
        self.assertEqual(get_preset_text(None), "")

    def test_preset_metadata_helper(self):
        """Verifies get_preset_metadata retrieves metadata dictionary."""
        choices = get_preset_choices()
        for choice in choices:
            meta = get_preset_metadata(choice)
            self.assertIsInstance(meta, dict)
            self.assertIn("domain", meta)
            self.assertIn("expected_verdict", meta)
            self.assertIn("generator", meta)

        # Nonexistent preset
        self.assertEqual(get_preset_metadata("Nonexistent"), {})
        self.assertEqual(get_preset_metadata(None), {})

    def test_expected_verdicts_coverage(self):
        """Verifies that the 6 presets span all three verdict categories."""
        verdicts = {PRESET_SAMPLES[k]["expected_verdict"] for k in PRESET_SAMPLES}
        self.assertIn("Authentic Human", verdicts)
        self.assertIn("Machine-Generated", verdicts)
        self.assertIn("Partially AI / Hybrid", verdicts)

        # Specifically check types: 3 human, 2 AI, 1 hybrid
        human_count = sum(1 for k in PRESET_SAMPLES if PRESET_SAMPLES[k]["expected_verdict"] == "Authentic Human")
        ai_count = sum(1 for k in PRESET_SAMPLES if PRESET_SAMPLES[k]["expected_verdict"] == "Machine-Generated")
        hybrid_count = sum(1 for k in PRESET_SAMPLES if PRESET_SAMPLES[k]["expected_verdict"] == "Partially AI / Hybrid")

        self.assertEqual(human_count, 3)
        self.assertEqual(ai_count, 2)
        self.assertEqual(hybrid_count, 1)

    def test_distinct_sample_texts(self):
        """Verifies that all 6 presets have unique, distinct Kazakh texts."""
        texts = [PRESET_SAMPLES[k]["text"] for k in PRESET_SAMPLES]
        self.assertEqual(len(set(texts)), 6)


if __name__ == "__main__":
    unittest.main()
