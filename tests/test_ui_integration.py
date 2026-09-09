# -*- coding: utf-8 -*-
"""
Tests for ui/app.py:
Interactive Gradio Explainability Dashboard Integration.
"""

import unittest
import tempfile
import os
import gradio as gr

from ui.app import (
    create_app,
    handle_analyze_document,
    handle_sentence_select,
    handle_preset_change,
    handle_file_upload,
    OfflineHeuristicDetector
)
from ui.presets import get_preset_choices, get_preset_text


class TestUiIntegration(unittest.TestCase):

    def test_gradio_app_structure_and_components(self):
        demo = create_app(load_model=False)
        self.assertIsNotNone(demo)
        self.assertIsInstance(demo, gr.Blocks)
        self.assertEqual(demo.title, "Kazakh AI-Text Detector & Explainability UI")

    def test_offline_heuristic_detector(self):
        detector = OfflineHeuristicDetector()
        # Human text
        human_text = "Бұл менің сүйікті дүкенімнен сатып алған тауарым. Сапасы өте керемет, жеткізу жылдам болды."
        res_human = detector.predict_document(human_text)
        self.assertEqual(res_human.verdict, "Authentic Human")
        self.assertLess(res_human.document_ai_probability, 0.40)
        self.assertEqual(res_human.ai_content_ratio, 0.0)

        # AI text
        ai_text = "Қорытындылай келе, бұл құбылыс заманауи қоғам үшін ерекше маңызды рөл атқарады. Осыған орай, жүйелі түрде талдау жасау қажет."
        res_ai = detector.predict_document(ai_text)
        self.assertGreaterEqual(res_ai.document_ai_probability, 0.40)

    def test_handle_analyze_document_empty(self):
        detector = OfflineHeuristicDetector()
        (
            verdict_badge,
            prob_text,
            ratio_text,
            stats_text,
            heatmap_html,
            dropdown_update,
            sentence_text,
            gate_html,
            morpheme_html,
            raw_sents_state
        ) = handle_analyze_document("", detector)

        self.assertIn("Мәтін енгізілмеді", heatmap_html)
        self.assertEqual(raw_sents_state, [])

    def test_handle_analyze_document_with_text(self):
        detector = OfflineHeuristicDetector()
        text = "Бұл бірінші қазақша сөйлем. Ал бұл екінші сөйлем болып табылады."
        (
            verdict_badge,
            prob_text,
            ratio_text,
            stats_text,
            heatmap_html,
            dropdown_update,
            sentence_text,
            gate_html,
            morpheme_html,
            raw_sents_state
        ) = handle_analyze_document(text, detector)

        self.assertIn("kaz-sentence", heatmap_html)
        self.assertEqual(len(raw_sents_state), 2)
        self.assertIn("Сөздер", stats_text)
        self.assertIn("Сөйлемдер: 2", stats_text)

    def test_handle_sentence_select(self):
        detector = OfflineHeuristicDetector()
        text = "Қазақстанның астанасы — Астана қаласы."
        *_, raw_sents_state = handle_analyze_document(text, detector)

        sent_text, gate_html, morph_html = handle_sentence_select(0, raw_sents_state)
        self.assertIn("Қазақстанның", sent_text)
        self.assertIn("gate-bar-container", gate_html)
        self.assertIn("fst-table", morph_html)

    def test_handle_preset_change(self):
        choices = get_preset_choices()
        self.assertGreater(len(choices), 0)
        first_choice = choices[0]
        text_out, meta_out = handle_preset_change(first_choice)
        expected_text = get_preset_text(first_choice)
        self.assertEqual(text_out, expected_text)
        self.assertIn("Домен:", meta_out)

    def test_handle_file_upload(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as f:
            f.write("Жүктелген файлдағы қазақша мәтін үлгісі.")
            tmp_path = f.name
        try:
            text_out, status_out = handle_file_upload(tmp_path)
            self.assertEqual(text_out, "Жүктелген файлдағы қазақша мәтін үлгісі.")
            self.assertIn("сәтті жүктелді", status_out)
        finally:
            os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
