# -*- coding: utf-8 -*-
"""
Tests for ui/app.py:
Academic 3-Tabbed Gradio Explainability Dashboard Integration.

Verifies:
- 3 Dedicated academic tabs (Detection & Explainability, Morphological FST Lab, Methodology)
- 6 Quick-sample pills horizontally placed and correctly wired
- Horizontal confidence progress meter with percentage display
- Dynamic linguistic reasoning bullets
- Tab 2 Morphological FST Lab parsing and 8-column decomposition table
- Tab 3 Academic methodology (Kaz-MAGE 2x2 matrix, empirical results, mathematical formulation, credits)
- Bilingual toggle updating all components cleanly
- Mandatory Zero Emoji presence across all UI text, badges, tabs, and HTML
- Strict XSS sanitization
"""

import unittest
import tempfile
import os
import re
import gradio as gr

from ui.app import (
    create_app,
    handle_analyze_document,
    handle_sentence_select,
    handle_fst_parse,
    handle_preset_change,
    handle_file_upload,
    switch_ui_language,
    render_hero_html,
    render_legend_html,
    render_methodology_markdown,
    extract_detailed_morphemes,
    OfflineHeuristicDetector,
    QUICK_SAMPLES,
    I18N
)
from ui.highlighting import render_confidence_meter, render_fst_decomposition_table
from ui.presets import get_preset_choices, get_preset_text


class TestUiIntegration(unittest.TestCase):

    def test_gradio_app_structure_and_components(self):
        """Verifies Gradio Blocks app initializes with title and expected structure."""
        demo = create_app(load_model=False)
        self.assertIsNotNone(demo)
        self.assertIsInstance(demo, gr.Blocks)
        self.assertEqual(demo.title, "Kazakh AI-Text Detector & Explainability UI")

    def test_offline_heuristic_detector(self):
        """Verifies defensive offline heuristic detector on human and AI text."""
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
        """Verifies empty input handling returns awaiting badges, empty confidence meter, and clean heatmap."""
        detector = OfflineHeuristicDetector()
        # Default English
        (
            exec_card_html,
            bullets_md,
            heatmap_html,
            dropdown_update,
            sentence_text,
            gate_html,
            morpheme_html,
            raw_sents_state
        ) = handle_analyze_document("", detector)

        self.assertIn("executive-summary-card", exec_card_html)
        self.assertIn("[AWAITING INPUT]", exec_card_html)
        self.assertIn("0.0%", exec_card_html)
        self.assertEqual(bullets_md, "")
        self.assertIn("No text entered or document is empty.", heatmap_html)
        self.assertEqual(raw_sents_state, [])

        # Explicit Kazakh
        *_, kz_heatmap, _, _, _, _, _ = handle_analyze_document("", detector, lang_choice="kz")
        self.assertIn("Мәтін енгізілмеді", kz_heatmap)

    def test_handle_analyze_document_with_human_text(self):
        """Verifies analysis of authentic human text produces authentic status badge, confidence meter, and bullets."""
        detector = OfflineHeuristicDetector()
        text = "Бұл бірінші қазақша сөйлем. Ал бұл екінші сөйлем болып табылады."
        # Default English
        (
            exec_card_html,
            bullets_md,
            heatmap_html,
            dropdown_update,
            sentence_text,
            gate_html,
            morpheme_html,
            raw_sents_state
        ) = handle_analyze_document(text, detector)

        self.assertIn("executive-summary-card", exec_card_html)
        self.assertIn("badge-human", exec_card_html)
        self.assertNotIn("[======", exec_card_html)
        self.assertIn("kaz-sentence", heatmap_html)
        self.assertEqual(len(raw_sents_state), 2)
        self.assertIn("words", exec_card_html.lower())
        self.assertIn("sentences", exec_card_html.lower())
        # Linguistic reasoning bullets present
        self.assertTrue(len(bullets_md) > 0)
        self.assertTrue(bullets_md.startswith("- "))

        # Explicit Kazakh
        kz_exec, kz_bullets, _, _, _, _, _, _ = handle_analyze_document(text, detector, lang_choice="kz")
        self.assertIn("сөз", kz_exec)
        self.assertIn("сөйлем", kz_exec)

    def test_handle_analyze_document_with_ai_text(self):
        """Verifies analysis of AI text produces [MACHINE-GENERATED] badge and high confidence meter."""
        detector = OfflineHeuristicDetector()
        ai_sample = QUICK_SAMPLES["pill_3"]["text"]
        # Default English
        (
            exec_card_html,
            bullets_md,
            heatmap_html,
            dropdown_update,
            sentence_text,
            gate_html,
            morpheme_html,
            raw_sents_state
        ) = handle_analyze_document(ai_sample, detector)

        self.assertIn("badge-ai", exec_card_html)
        self.assertIn("meter-ai", exec_card_html)
        self.assertTrue("99.9%" in exec_card_html or "100.0%" in exec_card_html)
        self.assertTrue(len(bullets_md) > 0)
        self.assertIn("formulaic", bullets_md.lower())

        # Explicit Kazakh
        _, kz_bullets, _, _, _, _, _, _ = handle_analyze_document(ai_sample, detector, lang_choice="kz")
        self.assertIn("формулалық", kz_bullets.lower())

    def test_handle_sentence_select(self):
        """Verifies selecting a sentence updates text, gate bar, and morpheme table."""
        detector = OfflineHeuristicDetector()
        text = "Қазақстанның астанасы — Астана қаласы."
        *_, raw_sents_state = handle_analyze_document(text, detector)

        sent_text, gate_html, morph_html = handle_sentence_select(0, raw_sents_state)
        self.assertIn("Қазақстанның", sent_text)
        self.assertIn("gate-bar-container", gate_html)
        self.assertIn("fst-table", morph_html)

    def test_quick_sample_pills_wired(self):
        """Verifies all 6 quick-sample pills are defined with valid Kazakh text and expected verdicts."""
        self.assertEqual(len(QUICK_SAMPLES), 6)
        expected_pills = ["pill_1", "pill_2", "pill_3", "pill_4", "pill_5", "pill_6"]
        detector = OfflineHeuristicDetector()

        for pill_id in expected_pills:
            self.assertIn(pill_id, QUICK_SAMPLES)
            sample = QUICK_SAMPLES[pill_id]
            self.assertIn("title_kz", sample)
            self.assertIn("title_en", sample)
            self.assertIn("text", sample)
            self.assertIn("expected_verdict", sample)
            self.assertGreater(len(sample["text"].strip()), 25)

            # Test classification matches expected verdict category
            res = detector.predict_document(sample["text"])
            expected = sample["expected_verdict"]
            if expected == "Authentic Human":
                self.assertEqual(res.verdict, "Authentic Human")
            elif expected == "Machine-Generated":
                self.assertEqual(res.verdict, "Machine-Generated")
            elif "Partially" in expected or "Hybrid" in expected:
                self.assertIn(res.verdict, ["Partially AI", "Machine-Generated"])

    def test_handle_fst_lab_parse(self):
        """Verifies Tab 2 Morphological FST Lab parsing and 8-column decomposition table."""
        detector = OfflineHeuristicDetector()
        text = "Қазақстанның қалаларында тұратын адамдармен сөйлестім."
        gate_html, table_html = handle_fst_parse(text, detector=detector, lang_choice="kz")

        self.assertIn("gate-bar-container", gate_html)
        self.assertIn("fst-table", table_html)
        self.assertIn("Сөз (Word)", table_html)
        self.assertIn("Түбір (Root)", table_html)
        self.assertIn("Сөз табы (POS)", table_html)
        self.assertIn("Септік (Case)", table_html)
        self.assertIn("Көптік (Plural)", table_html)
        self.assertIn("Шақ (Tense)", table_html)
        self.assertIn("Жұрнақ тізбегі (Suffix Chain)", table_html)
        self.assertIn("Қазақстанның", table_html)

        # Verify accurate deep root extraction for complex verbalized word: қанағаттанбаған -> қанағат
        _, qan_table = handle_fst_parse("қанағаттанбаған", detector=detector, lang_choice="en")
        self.assertIn("қанағат", qan_table)
        self.assertIn("-тан (VERB.DERIV)", qan_table)
        self.assertIn("-ба (NEG)", qan_table)
        self.assertIn("-ған (PAST.PART)", qan_table)

        # Verify empty input handling
        empty_gate, empty_table = handle_fst_parse("", detector=detector, lang_choice="kz")
        self.assertIn("gate-bar-container", empty_gate)
        self.assertIn("Морфологиялық талдау деректері жоқ", empty_table)

    def test_extract_detailed_morphemes_xss(self):
        """Verifies XSS sanitization in detailed morpheme extraction and table rendering."""
        malicious = "<script>alert('fst_xss')</script> <img src=x onerror=1>"
        breakdowns = extract_detailed_morphemes(malicious)
        table_html = render_fst_decomposition_table(breakdowns, lang="kz")

        self.assertNotIn("<script>", table_html)
        self.assertNotIn("<img", table_html)
        self.assertNotIn("alert('fst_xss')", table_html)

        # Direct table rendering with malicious tags across all columns
        raw_xss_data = [{
            "word": "<script>evil</script>",
            "root": "<b>root</b>",
            "pos": "<i>pos</i>",
            "case": "<span class='case'>-дан</span>",
            "plural": "<a href='evil'>-лар</a>",
            "tense": "<script>-ған</script>",
            "person": "<img src=x>-мын",
            "suffix_chain": "<iframe src='bad'></iframe>"
        }]
        safe_html = render_fst_decomposition_table(raw_xss_data, lang="kz")
        self.assertNotIn("<script>", safe_html)
        self.assertNotIn("<b>", safe_html)
        self.assertNotIn("<i>", safe_html)
        self.assertNotIn("<span class='case'>", safe_html)
        self.assertNotIn("<a ", safe_html)
        self.assertNotIn("<img", safe_html)
        self.assertNotIn("<iframe", safe_html)
        self.assertIn("&lt;script&gt;evil&lt;/script&gt;", safe_html)
        self.assertIn("&lt;b&gt;root&lt;/b&gt;", safe_html)
        self.assertIn("&lt;iframe src=&#x27;bad&#x27;&gt;&lt;/iframe&gt;", safe_html)

    def test_confidence_meter_rendering(self):
        """Verifies publication-grade horizontal confidence meter rendering with zero emojis."""
        # 1. High AI
        meter_ai = render_confidence_meter(0.9985, "[MACHINE-GENERATED]", lang="kz")
        self.assertIn("99.9%", meter_ai)
        self.assertIn("meter-ai", meter_ai)
        self.assertNotIn("[======", meter_ai)
        self.assertIn("confidence-track", meter_ai)
        self.assertIn("[MACHINE-GENERATED]", meter_ai)

        # 2. Borderline / Amber
        meter_amber = render_confidence_meter(0.552, "[PARTIALLY AI / HYBRID]", lang="en")
        self.assertIn("55.2%", meter_amber)
        self.assertIn("meter-amber", meter_amber)
        self.assertIn("AI Confidence Level", meter_amber)

        # 3. Authentic Human
        meter_human = render_confidence_meter(0.042, "[AUTHENTIC HUMAN]", lang="kz")
        self.assertIn("4.2%", meter_human)
        self.assertIn("meter-human", meter_human)
        self.assertIn("AI Сенімділік Деңгейі", meter_human)

        # 4. None / Clamping
        meter_none = render_confidence_meter(None, lang="kz")
        self.assertIn("0.0%", meter_none)

        meter_over = render_confidence_meter(1.5, lang="kz")
        self.assertIn("100.0%", meter_over)

        # 5. XSS escaping on verdict tag
        meter_xss = render_confidence_meter(0.5, "<script>alert(1)</script>", lang="kz")
        self.assertNotIn("<script>", meter_xss)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", meter_xss)

    def test_methodology_tab_content(self):
        """Verifies Tab 3 contains Kaz-MAGE 2x2 matrix, empirical table, gated formula, and credits."""
        for lang in ["kz", "en"]:
            md = render_methodology_markdown(lang)
            # Kaz-MAGE 2x2 Matrix
            self.assertIn("Kaz-MAGE", md)
            self.assertIn("Q1", md)
            self.assertIn("Q2", md)
            self.assertIn("Q3", md)
            self.assertIn("Q4", md)
            # Empirical results
            self.assertIn("0.9996", md)
            self.assertIn("0.9997", md)
            self.assertIn("0.7812", md)
            self.assertIn("0.9253", md)
            self.assertIn("0.5762", md)
            self.assertIn("0.9980", md)
            self.assertIn("0.6231", md)
            self.assertIn("1.0000", md)
            # Mathematical formulation
            self.assertIn(r"\mathbf{h}", md)
            self.assertIn(r"\mathbf{g}", md)
            self.assertIn(r"\mathbf{h}_{\text{sem}}", md)
            self.assertIn(r"\mathbf{h}_{\text{morph}}", md)
            self.assertIn(r"\sigma", md)
            # Section 4 (Credits) removed as requested
            self.assertNotIn("Project Credits", md)
            self.assertNotIn("Жоба Авторлары", md)
            self.assertNotIn("Da Lei", md)

    def test_language_switch(self):
        """Verifies switching language updates all pills, labels, methodology, and placeholders."""
        en_outputs = switch_ui_language("English")
        self.assertIsInstance(en_outputs, tuple)
        self.assertEqual(len(en_outputs), 35)

        # Verify English strings in key positions
        quick_samples_en = en_outputs[0]
        self.assertIn("Quick Benchmark Samples", quick_samples_en)

        exec_card_en = en_outputs[13]
        self.assertIn("Document Analysis Executive Summary", exec_card_en)

        methodology_en = en_outputs[-1]
        self.assertIn("Benchmark & Academic Methodology", methodology_en)
        self.assertIn("Scientific Significance", methodology_en)

        # Switch back to Kazakh
        kz_outputs = switch_ui_language("Қазақша")
        quick_samples_kz = kz_outputs[0]
        self.assertIn("Жылдам сынақ үлгілері", quick_samples_kz)

        exec_card_kz = kz_outputs[13]
        self.assertIn("Құжатты Сараптаудың Қорытындысы", exec_card_kz)

    def test_zero_emoji_constraint_across_ui(self):
        """
        MANDATORY ZERO EMOJI AUDIT:
        Asserts that NO decorative emojis exist anywhere in tab titles, hero banners,
        badges, legend, methodology, pills, I18N dictionary, or confidence meter.
        """
        # Regex matching common unicode emojis (pictographs, symbols, flags, supplemental)
        emoji_pattern = re.compile(
            r'[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf\ufe0f]',
            flags=re.UNICODE
        )

        # 1. Audit I18N dictionaries
        for lang_key, d in I18N.items():
            for key, val in d.items():
                if isinstance(val, str):
                    matches = emoji_pattern.findall(val)
                    self.assertEqual(
                        matches, [],
                        f"Found emoji {matches} in I18N[{lang_key}][{key}]: {val}"
                    )

        # 2. Audit rendered HTML strings
        for lang in ["kz", "en"]:
            hero = render_hero_html(lang)
            self.assertEqual(emoji_pattern.findall(hero), [], f"Emoji found in render_hero_html({lang})")

            legend = render_legend_html(lang)
            self.assertEqual(emoji_pattern.findall(legend), [], f"Emoji found in render_legend_html({lang})")

            method = render_methodology_markdown(lang)
            self.assertEqual(emoji_pattern.findall(method), [], f"Emoji found in render_methodology_markdown({lang})")

            meter = render_confidence_meter(0.9985, "[MACHINE-GENERATED]", lang)
            self.assertEqual(emoji_pattern.findall(meter), [], f"Emoji found in render_confidence_meter({lang})")

        # 3. Audit quick sample pill labels
        for pill_id, sample in QUICK_SAMPLES.items():
            self.assertEqual(emoji_pattern.findall(sample["title_kz"]), [])
            self.assertEqual(emoji_pattern.findall(sample["title_en"]), [])

        # 4. Audit handle_analyze_document outputs
        detector = OfflineHeuristicDetector()
        res_tuple = handle_analyze_document("Сынақ мәтіні", detector, "kz")
        for item in res_tuple:
            if isinstance(item, str):
                self.assertEqual(
                    emoji_pattern.findall(item), [],
                    f"Emoji found in handle_analyze_document output: {item[:60]}"
                )

    def test_handle_preset_change(self):
        """Verifies preset sample selection populates text and metadata."""
        choices = get_preset_choices()
        self.assertGreater(len(choices), 0)
        first_choice = choices[0]
        # Default English
        text_out, meta_out = handle_preset_change(first_choice)
        expected_text = get_preset_text(first_choice)
        self.assertEqual(text_out, expected_text)
        self.assertIn("Domain:", meta_out)

        # Explicit Kazakh
        _, meta_kz = handle_preset_change(first_choice, lang_choice="kz")
        self.assertIn("Домен:", meta_kz)

    def test_handle_file_upload(self):
        """Verifies file ingestion returns extracted text and clean status without emojis."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as f:
            f.write("Жүктелген файлдағы қазақша мәтін үлгісі.")
            tmp_path = f.name
        try:
            # Default English
            text_out, status_out = handle_file_upload(tmp_path)
            self.assertEqual(text_out, "Жүктелген файлдағы қазақша мәтін үлгісі.")
            self.assertIn("File loaded successfully", status_out)
            self.assertNotIn("✅", status_out)

            # Explicit Kazakh
            _, status_kz = handle_file_upload(tmp_path, lang_choice="kz")
            self.assertIn("сәтті жүктелді", status_kz)
            self.assertNotIn("✅", status_kz)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
