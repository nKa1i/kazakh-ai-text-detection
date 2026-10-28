"""Unit tests for DynamicFSTOverlay, longest-match sorting, morphophonological alternation, and JSON persistence."""

import os
import unittest
import tempfile
from src.auto_fst.overlay import DynamicFSTOverlay

class TestDynamicFSTOverlay(unittest.TestCase):
    def setUp(self):
        self.overlay = DynamicFSTOverlay()

    def test_longest_match_first_sorting(self):
        # Prevent "чат" from shadowing "чатбот"
        self.overlay.register_root("чат")
        self.overlay.register_root("чатбот")
        segmented = self.overlay.analyze_and_segment("чатботтың жұмысы")
        self.assertIn("чатбот -тың", segmented)
        self.assertNotIn("чат -боттың", segmented)

    def test_dynamic_slang_registration(self):
        raw = "Біз кеше қатты вайбтадық"
        before = self.overlay.analyze_and_segment(raw)
        self.assertNotIn("вайб -та -дық", before)

        self.overlay.register_root("вайб")
        after = self.overlay.analyze_and_segment(raw)
        self.assertIn("вайб -та -дық", after)

    def test_russian_layout_particle_handling(self):
        raw = "Ол әлі жұмыста турсынгой"
        segmented = self.overlay.analyze_and_segment(raw)
        self.assertTrue("тур -сын -гой" in segmented or "тұр -сың -ғой" in segmented)

    def test_json_persistence(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name
        try:
            self.overlay.register_root("донат", {"harmony": "hard"})
            self.overlay.save_to_json(temp_path)

            new_overlay = DynamicFSTOverlay()
            count = new_overlay.load_from_json(temp_path)
            self.assertEqual(count, 1)
            self.assertIn("донат", new_overlay.loanword_roots)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_mixed_script_hyphenated_segmentation(self):
        raw = "Біз жаңа AI-дың жүйесін қолданып, ChatGPT-мен сөйлестік."
        segmented = self.overlay.analyze_and_segment(raw)
        self.assertIn("AI -дың", segmented)
        self.assertIn("ChatGPT -мен", segmented)

    def test_native_voicing_alternation_in_overlay(self):
        raw = "Оқушының кітабы үстелде жатыр, ал оның жүрегі тыныш."
        segmented = self.overlay.analyze_and_segment(raw)
        self.assertIn("кітап -ы", segmented)
        self.assertIn("жүрек -і", segmented)

    def test_vowel_syncope_alternation_in_overlay(self):
        raw = "Қазақстан халқы бейбітшілікті қалайды."
        segmented = self.overlay.analyze_and_segment(raw)
        self.assertIn("халық -ы", segmented)

    def test_registered_loanword_voicing_alternation(self):
        self.overlay.register_root("стартап")
        raw = "Бұл жаңа стартабы тез өсті."
        segmented = self.overlay.analyze_and_segment(raw)
        self.assertIn("стартап -ы", segmented)

    def test_loanword_nominal_suffixes_not_corrupted_by_verbalizer(self):
        # каспиден -> каспи -ден (NOT каспи -де -н)
        # бонустар -> бонус -тар (NOT бонус -та -р)
        # видеолар -> видео -лар (NOT видео -ла -р)
        # файлдарды -> файл -дар -ды (NOT файл -да -рды)
        raw = "каспиден бонустар видеолар файлдарды чектерді курьерден"
        segmented = self.overlay.analyze_and_segment(raw)
        self.assertIn("каспи -ден", segmented)
        self.assertIn("бонус -тар", segmented)
        self.assertIn("видео -лар", segmented)
        self.assertIn("файл -дар -ды", segmented)
        self.assertIn("чек -тер -ді", segmented)
        self.assertIn("курьер -ден", segmented)

    def test_two_letter_roots_and_uninflected_words(self):
        # үйлердің -> үй -лер -дің; ұлдардың -> ұл -дар -дың; дәптер -> дәптер (unsplit)
        raw = "үйлердің ұлдардың дәптер жұмысың"
        segmented = self.overlay.analyze_and_segment(raw)
        self.assertIn("үй -лер -дің", segmented)
        self.assertIn("ұл -дар -дың", segmented)
        self.assertIn("дәптер", segmented)
        self.assertNotIn("дәп -тер", segmented)
        self.assertIn("жұмыс -ың", segmented)


if __name__ == '__main__':
    unittest.main()
