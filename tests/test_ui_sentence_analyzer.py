# -*- coding: utf-8 -*-
"""
Tests for ui/sentence_analyzer.py:
Sentence-level explainability scoring and FST morphological breakdown engine.
"""

import unittest
from kaz_mage.document import DocumentAnalysisResult, DocumentChunk
from fst_analyzer import AdvancedKazakhFSTAnalyzer


class TestUiSentenceAnalyzer(unittest.TestCase):

    def test_sentence_morphemes_breakdown(self):
        from ui.sentence_analyzer import analyze_sentence_morphemes
        breakdowns = analyze_sentence_morphemes("Қазақстанның болашағы жарқын.")
        self.assertGreater(len(breakdowns), 0)
        self.assertEqual(breakdowns[0]["word"], "Қазақстанның")
        self.assertIn("root", breakdowns[0])
        self.assertIn("affixes", breakdowns[0])
        self.assertEqual(breakdowns[0]["pos"], "NOUN")
        self.assertIn("-ның (GEN)", breakdowns[0]["affixes"])

    def test_noun_cases_and_plurals(self):
        from ui.sentence_analyzer import analyze_sentence_morphemes
        # қалалардың -> root: қала, pos: NOUN, affixes: [-лар (PLUR), -дың (GEN)]
        breakdowns = analyze_sentence_morphemes("қалалардың")
        self.assertEqual(len(breakdowns), 1)
        item = breakdowns[0]
        self.assertEqual(item["word"], "қалалардың")
        self.assertEqual(item["root"], "қала")
        self.assertEqual(item["pos"], "NOUN")
        self.assertEqual(item["affixes"], ["-лар (PLUR)", "-дың (GEN)"])

        # балаларға -> root: бала, pos: NOUN, affixes: [-лар (PLUR), -ға (DAT)]
        breakdowns_dat = analyze_sentence_morphemes("балаларға")
        self.assertEqual(len(breakdowns_dat), 1)
        item_dat = breakdowns_dat[0]
        self.assertEqual(item_dat["root"], "бала")
        self.assertEqual(item_dat["pos"], "NOUN")
        self.assertEqual(item_dat["affixes"], ["-лар (PLUR)", "-ға (DAT)"])

    def test_verb_tenses_and_persons(self):
        from ui.sentence_analyzer import analyze_sentence_morphemes
        # келгенмін -> root: кел, pos: VERB, affixes: [-ген (PAST.PART), -мін (PERS.1SG)]
        breakdowns = analyze_sentence_morphemes("келгенмін")
        self.assertEqual(len(breakdowns), 1)
        item = breakdowns[0]
        self.assertEqual(item["word"], "келгенмін")
        self.assertEqual(item["root"], "кел")
        self.assertEqual(item["pos"], "VERB")
        self.assertEqual(item["affixes"], ["-ген (PAST.PART)", "-мін (PERS.1SG)"])

        # жасалғандықтан -> root: жасал, pos: VERB, affixes: [-ғандықтан (CVB.REASON)]
        breakdowns_cvb = analyze_sentence_morphemes("жасалғандықтан")
        self.assertEqual(len(breakdowns_cvb), 1)
        item_cvb = breakdowns_cvb[0]
        self.assertEqual(item_cvb["pos"], "VERB")
        self.assertEqual(item_cvb["affixes"], ["-ғандықтан (CVB.REASON)"])

        # қанағаттанбаған -> root: қанағат, pos: VERB, affixes: [-тан (VERB.DERIV), -ба (NEG), -ған (PAST.PART)]
        breakdowns_qan = analyze_sentence_morphemes("қанағаттанбаған")
        self.assertEqual(len(breakdowns_qan), 1)
        item_qan = breakdowns_qan[0]
        self.assertEqual(item_qan["word"], "қанағаттанбаған")
        self.assertEqual(item_qan["root"], "қанағат")
        self.assertEqual(item_qan["pos"], "VERB")
        self.assertEqual(item_qan["affixes"], ["-тан (VERB.DERIV)", "-ба (NEG)", "-ған (PAST.PART)"])

    def test_loanword_breakdown(self):
        from ui.sentence_analyzer import analyze_sentence_morphemes
        # доставкасы -> root: доставка, pos: LOANWORD/NOUN, affixes: [-сы]
        breakdowns = analyze_sentence_morphemes("доставкасы")
        self.assertEqual(len(breakdowns), 1)
        item = breakdowns[0]
        self.assertEqual(item["word"], "доставкасы")
        self.assertEqual(item["root"], "доставка")
        self.assertEqual(item["pos"], "LOANWORD/NOUN")
        self.assertEqual(item["affixes"], ["-сы"])

        # каспиден -> root: каспи, pos: LOANWORD/NOUN, affixes: [-ден]
        breakdowns_kaspi = analyze_sentence_morphemes("каспиден")
        self.assertEqual(len(breakdowns_kaspi), 1)
        self.assertEqual(breakdowns_kaspi[0]["root"], "каспи")
        self.assertEqual(breakdowns_kaspi[0]["pos"], "LOANWORD/NOUN")
        self.assertEqual(breakdowns_kaspi[0]["affixes"], ["-ден"])

        # Standalone loanword root: банк -> root: банк, pos: LOANWORD/NOUN, affixes: []
        breakdowns_bank = analyze_sentence_morphemes("банк")
        self.assertEqual(len(breakdowns_bank), 1)
        self.assertEqual(breakdowns_bank[0]["root"], "банк")
        self.assertEqual(breakdowns_bank[0]["pos"], "LOANWORD/NOUN")
        self.assertEqual(breakdowns_bank[0]["affixes"], [])

    def test_short_and_numeric_tokens(self):
        from ui.sentence_analyzer import analyze_sentence_morphemes
        # Length <= 2
        b_short = analyze_sentence_morphemes("ак")
        self.assertEqual(len(b_short), 1)
        self.assertEqual(b_short[0]["root"], "ак")
        self.assertEqual(b_short[0]["pos"], "OTHER")
        self.assertEqual(b_short[0]["affixes"], [])

        # Pure numeric
        b_num = analyze_sentence_morphemes("2024")
        self.assertEqual(len(b_num), 1)
        self.assertEqual(b_num[0]["root"], "2024")
        self.assertEqual(b_num[0]["pos"], "OTHER")
        self.assertEqual(b_num[0]["affixes"], [])

    def test_punctuation_and_quotes(self):
        from ui.sentence_analyzer import analyze_sentence_morphemes
        text = "«Қалалардың», үйлері! Астана — бас қала."
        breakdowns = analyze_sentence_morphemes(text)
        words = [b["word"] for b in breakdowns]
        self.assertIn("Қалалардың", words)
        self.assertIn("үйлері", words)
        self.assertIn("Астана", words)
        self.assertIn("бас", words)
        self.assertIn("қала", words)
        # Verify punctuation characters themselves are not treated as words
        self.assertNotIn("—", words)
        self.assertNotIn("«", words)
        self.assertNotIn("»", words)

    def test_empty_and_whitespace_inputs(self):
        from ui.sentence_analyzer import analyze_sentence_morphemes, analyze_document_sentences
        self.assertEqual(analyze_sentence_morphemes(""), [])
        self.assertEqual(analyze_sentence_morphemes("   \n\t  "), [])
        self.assertEqual(analyze_sentence_morphemes(None), [])

        self.assertEqual(analyze_document_sentences(""), [])
        self.assertEqual(analyze_document_sentences("   \n\t  "), [])
        self.assertEqual(analyze_document_sentences(None), [])

    def test_document_sentence_attribution_single_chunk(self):
        from ui.sentence_analyzer import analyze_document_sentences
        text = "Бұл бірінші сөйлем. Бұл екінші сөйлем."
        chunk = DocumentChunk(
            0,
            text,
            0,
            len(text),
            6,
            2,
            ai_probability=0.999,
            gate_value=0.52,
            is_ai=True
        )
        doc_res = DocumentAnalysisResult(
            "Machine-Generated",
            0.999,
            1.0,
            0.9980,
            6,
            2,
            1,
            chunk,
            [chunk]
        )
        sents = analyze_document_sentences(text, doc_res)
        self.assertEqual(len(sents), 2)
        self.assertEqual(sents[0]["index"], 0)
        self.assertEqual(sents[0]["text"], "Бұл бірінші сөйлем.")
        self.assertAlmostEqual(sents[0]["ai_probability"], 0.999, places=3)
        self.assertAlmostEqual(sents[0]["gate_value"], 0.52, places=2)
        self.assertTrue(sents[0]["is_ai"])

        self.assertEqual(sents[1]["index"], 1)
        self.assertEqual(sents[1]["text"], "Бұл екінші сөйлем.")
        self.assertAlmostEqual(sents[1]["ai_probability"], 0.999, places=3)
        self.assertTrue(sents[1]["is_ai"])

    def test_document_sentence_attribution_multi_chunk_worst_case(self):
        from ui.sentence_analyzer import analyze_document_sentences
        # Document with 3 sentences: s0 (human), s1 (in overlap), s2 (ai)
        # s1 is covered by chunk0 (prob=0.15) and chunk1 (prob=0.9995)
        # Defensive worst-case attribution must assign max(0.15, 0.9995) = 0.9995 to s1
        s0 = "Бірінші адам сөйлемі."
        s1 = "Екінші ортақ сөйлем."
        s2 = "Үшінші жасанды сөйлем."
        full_text = f"{s0} {s1} {s2}"

        s0_start = 0
        s0_end = len(s0)
        s1_start = full_text.index(s1)
        s1_end = s1_start + len(s1)
        s2_start = full_text.index(s2)
        s2_end = s2_start + len(s2)

        chunk0_text = full_text[s0_start:s1_end]
        chunk0 = DocumentChunk(
            0,
            chunk0_text,
            s0_start,
            s1_end,
            6,
            2,
            ai_probability=0.15,
            gate_value=0.60,
            is_ai=False
        )

        chunk1_text = full_text[s1_start:s2_end]
        chunk1 = DocumentChunk(
            1,
            chunk1_text,
            s1_start,
            s2_end,
            6,
            2,
            ai_probability=0.9995,
            gate_value=0.45,
            is_ai=True
        )

        doc_res = DocumentAnalysisResult(
            "Partially AI",
            0.85,
            0.5,
            0.9980,
            9,
            3,
            2,
            chunk1,
            [chunk0, chunk1]
        )

        sents = analyze_document_sentences(full_text, doc_res)
        self.assertEqual(len(sents), 3)

        # Sentence 0: only chunk0
        self.assertAlmostEqual(sents[0]["ai_probability"], 0.15, places=2)
        self.assertFalse(sents[0]["is_ai"])

        # Sentence 1: overlapping, worst-case attribution max(0.15, 0.9995) = 0.9995
        self.assertAlmostEqual(sents[1]["ai_probability"], 0.9995, places=4)
        self.assertTrue(sents[1]["is_ai"])
        self.assertAlmostEqual(sents[1]["gate_value"], 0.45, places=2)

        # Sentence 2: only chunk1
        self.assertAlmostEqual(sents[2]["ai_probability"], 0.9995, places=4)
        self.assertTrue(sents[2]["is_ai"])

    def test_document_sentence_attribution_fallback_no_chunks(self):
        from ui.sentence_analyzer import analyze_document_sentences
        text = "Жалғыз сөйлем құжат."
        doc_res = DocumentAnalysisResult(
            "Authentic Human",
            0.12,
            0.0,
            0.9980,
            3,
            1,
            0,
            None,
            []
        )
        sents = analyze_document_sentences(text, doc_res)
        self.assertEqual(len(sents), 1)
        self.assertAlmostEqual(sents[0]["ai_probability"], 0.12, places=2)
        self.assertEqual(sents[0]["gate_value"], 0.5)
        self.assertFalse(sents[0]["is_ai"])

    def test_document_sentence_attribution_none_result(self):
        from ui.sentence_analyzer import analyze_document_sentences
        text = "Жалғыз сөйлем құжат."
        sents = analyze_document_sentences(text, None)
        self.assertEqual(len(sents), 1)
        self.assertEqual(sents[0]["ai_probability"], 0.0)
        self.assertEqual(sents[0]["gate_value"], 0.5)
        self.assertFalse(sents[0]["is_ai"])

    def test_document_sentence_with_dict_payload(self):
        from ui.sentence_analyzer import analyze_document_sentences
        text = "Бірінші сөйлем. Екінші сөйлем."
        dict_payload = {
            "verdict": "Machine-Generated",
            "document_ai_probability": 0.999,
            "calibrated_threshold": 0.9980,
            "chunks": [
                {
                    "index": 0,
                    "text": text,
                    "start_char": 0,
                    "end_char": len(text),
                    "ai_probability": 0.9992,
                    "gate_value": 0.53,
                    "is_ai": True
                }
            ]
        }
        sents = analyze_document_sentences(text, dict_payload)
        self.assertEqual(len(sents), 2)
        self.assertAlmostEqual(sents[0]["ai_probability"], 0.9992, places=4)
        self.assertTrue(sents[0]["is_ai"])

    def test_document_sentence_with_fst_analyzer_morphemes(self):
        from ui.sentence_analyzer import analyze_document_sentences
        text = "Қалалардың үйлері."
        fst = AdvancedKazakhFSTAnalyzer()
        sents = analyze_document_sentences(text, fst_analyzer=fst)
        self.assertEqual(len(sents), 1)
        self.assertIn("morphemes", sents[0])
        self.assertGreater(len(sents[0]["morphemes"]), 0)


if __name__ == "__main__":
    unittest.main()
