import unittest
import html
from ui.highlighting import (
    render_document_heatmap,
    render_dynamic_gate_bar,
    render_morpheme_table,
    HEATMAP_CSS,
)

class TestUiHighlighting(unittest.TestCase):
    def test_xss_escaping_in_heatmap(self):
        malicious_input = [{"index": 0, "text": "<script>alert('xss')</script>", "ai_probability": 0.05}]
        rendered = render_document_heatmap(malicious_input)
        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;", rendered)

    def test_xss_attribute_injection_in_heatmap(self):
        malicious_input = [{"index": 1, "text": '"><img src=x onerror=alert(1)>', "ai_probability": 0.95}]
        rendered = render_document_heatmap(malicious_input)
        self.assertNotIn("<img src=x", rendered)
        self.assertIn("&quot;&gt;&lt;img src=x onerror=alert(1)&gt;", rendered)

    def test_color_scale_classification(self):
        sents = [
            {"index": 0, "text": "Адам мәтіні.", "ai_probability": 0.05},
            {"index": 1, "text": "Аралық мәтін.", "ai_probability": 0.50},
            {"index": 2, "text": "Жасанды мәтін.", "ai_probability": 0.9995}
        ]
        rendered = render_document_heatmap(sents, calibrated_threshold=0.9980)
        self.assertIn("lvl-human", rendered)
        self.assertIn("lvl-amber", rendered)
        self.assertIn("lvl-ai", rendered)
        self.assertIn("kaz-heatmap-container", rendered)

    def test_threshold_boundaries(self):
        # Human < 0.40
        res_human = render_document_heatmap([{"index": 0, "text": "T", "ai_probability": 0.3999}])
        self.assertIn("lvl-human", res_human)
        self.assertNotIn("lvl-amber", res_human)

        # Amber: 0.40 <= prob < calibrated_threshold
        res_amber = render_document_heatmap([{"index": 0, "text": "T", "ai_probability": 0.4000}])
        self.assertIn("lvl-amber", res_amber)

        res_amber_edge = render_document_heatmap([{"index": 0, "text": "T", "ai_probability": 0.9979}])
        self.assertIn("lvl-amber", res_amber_edge)

        # AI: prob >= calibrated_threshold
        res_ai = render_document_heatmap([{"index": 0, "text": "T", "ai_probability": 0.9980}])
        self.assertIn("lvl-ai", res_ai)

    def test_empty_document_heatmap(self):
        rendered = render_document_heatmap([])
        self.assertIn("empty-doc-prompt", rendered)
        self.assertIn("Мәтін енгізілмеді немесе бос.", rendered)

    def test_dynamic_gate_bar_rendering(self):
        bar_html = render_dynamic_gate_bar(0.523)
        self.assertIn("52.3%", bar_html)
        self.assertIn("47.7%", bar_html)
        self.assertIn("gate-bar-container", bar_html)
        self.assertIn("gate-stream-bert", bar_html)
        self.assertIn("gate-stream-fst", bar_html)
        self.assertIn("Семантикалық Контекст (BERT)", bar_html)
        self.assertIn("Морфологиялық FST (Тіл Құрылымы)", bar_html)

    def test_dynamic_gate_bar_clamping(self):
        # Clamped to 0.0 - 1.0
        bar_low = render_dynamic_gate_bar(-0.1)
        self.assertIn("0.0%", bar_low)
        self.assertIn("100.0%", bar_low)

        bar_high = render_dynamic_gate_bar(1.5)
        self.assertIn("100.0%", bar_high)
        self.assertIn("0.0%", bar_high)

    def test_morpheme_table_rendering(self):
        breakdowns = [
            {"word": "жасанды", "root": "жаса", "pos": "VERB", "affixes": ["-н (PASS)", "-ды (ADJ)"]},
            {"word": "кітап", "root": "кітап", "pos": "NOUN", "affixes": []}
        ]
        table_html = render_morpheme_table(breakdowns)
        self.assertIn("жасанды", table_html)
        self.assertIn("жаса", table_html)
        self.assertIn("-н (PASS)", table_html)
        self.assertIn("fst-table", table_html)
        self.assertIn("Сөз (Word)", table_html)
        self.assertIn("Түбір (Stem)", table_html)
        self.assertIn("Сөз табы (POS)", table_html)
        self.assertIn("Жұрнақ / Жалғаулар (Affixes)", table_html)

    def test_morpheme_table_xss_escaping(self):
        breakdowns = [
            {
                "word": "<script>evil</script>",
                "root": "<b>root</b>",
                "pos": "<span 'test'>",
                "affixes": ["<img src=x onerror=1>", "' OR '1'='1"]
            }
        ]
        table_html = render_morpheme_table(breakdowns)
        self.assertNotIn("<script>", table_html)
        self.assertNotIn("<b>root</b>", table_html)
        self.assertNotIn("<img src=x", table_html)
        self.assertIn("&lt;script&gt;evil&lt;/script&gt;", table_html)
        self.assertIn("&lt;b&gt;root&lt;/b&gt;", table_html)
        self.assertIn("&lt;img src=x onerror=1&gt;", table_html)

    def test_morpheme_table_empty(self):
        rendered = render_morpheme_table([])
        self.assertIn("Морфологиялық талдау деректері жоқ.", rendered)
        self.assertIn("text-muted", rendered)

    def test_css_rules_present(self):
        self.assertIn(".kaz-sentence", HEATMAP_CSS)
        self.assertIn(".lvl-human", HEATMAP_CSS)
        self.assertIn(".lvl-amber", HEATMAP_CSS)
        self.assertIn(".lvl-ai", HEATMAP_CSS)
        self.assertIn(".gate-bar-container", HEATMAP_CSS)
        self.assertIn(".fst-table", HEATMAP_CSS)

if __name__ == "__main__":
    unittest.main()
