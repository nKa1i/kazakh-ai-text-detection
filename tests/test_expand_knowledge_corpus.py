# -*- coding: utf-8 -*-
"""
Tests for Kazakh Knowledge Corpus Expansion tooling.
Validates sentence segmentation, passage validation schema, and multi-domain corpus generation.
"""

import os
import json
import tempfile
import unittest

from scripts.expand_knowledge_corpus import (
    build_curated_knowledge_corpus,
    segment_sentences_kazakh,
    validate_knowledge_passage,
    VALID_DOMAINS
)


class TestExpandKnowledgeCorpus(unittest.TestCase):
    def test_segment_sentences_kazakh(self):
        text = "Абай Құнанбайұлы 1845 ж. туған. Ол ұлы қазақ ақыны! Оның еңбектері көп пе? Әрине, өте көп."
        sentences = segment_sentences_kazakh(text)
        self.assertGreaterEqual(len(sentences), 3)
        self.assertTrue(any("1845 ж. туған" in s for s in sentences))

    def test_segment_sentences_with_multiple_abbreviations(self):
        text = "Бұл жәдігер б.з.д. V ғ. жатады. Мұражайда қылыш, сауыт т.б. заттар сақталған. Проф. Сәтбаев зерттеу жүргізді."
        sentences = segment_sentences_kazakh(text)
        self.assertGreaterEqual(len(sentences), 2)
        # Verify abbreviations didn't cause premature splits
        self.assertTrue(any("б.з.д. V ғ." in s for s in sentences))
        self.assertTrue(any("т.б. заттар" in s for s in sentences))

    def test_segment_sentences_word_ending_in_q(self):
        # Words ending in 'қ' followed by a period must not be swallowed by the abbreviation 'қ.'
        text = "Бұл ұлы халық. Олар өз бостандығын қорғады."
        sentences = segment_sentences_kazakh(text)
        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0], "Бұл ұлы халық.")
        self.assertEqual(sentences[1], "Олар өз бостандығын қорғады.")

    def test_validate_knowledge_passage_valid(self):
        passage = {
            "passage_id": "wiki_kz_001",
            "title": "Қазақстан тарихы",
            "text": "Қазақстан Республикасы 1991 жылы өз тәуелсіздігін жариялап, егемен мемлекет ретінде әлемге танылды. Бұл тарихи оқиға халықтың сан ғасырлық күресінің нәтижесі болды.",
            "domain": "history"
        }
        self.assertTrue(validate_knowledge_passage(passage))

    def test_validate_knowledge_passage_invalid(self):
        short_passage = {
            "passage_id": "wiki_kz_short",
            "title": "Қысқа",
            "text": "Тым қысқа мәтін.",
            "domain": "unknown"
        }
        self.assertFalse(validate_knowledge_passage(short_passage))

        missing_fields = {
            "passage_id": "wiki_kz_002",
            "text": "Қазақстан Республикасының аумағы өте үлкен және бай табиғи ресурстарға ие."
        }
        self.assertFalse(validate_knowledge_passage(missing_fields))

        invalid_domain = {
            "passage_id": "wiki_kz_003",
            "title": "Спорт",
            "text": "Қазақстанда бокс және күрес спорт түрлері кеңінен дамыған және халықаралық жарыстарда жеңістерге жетуде.",
            "domain": "sports_invalid"
        }
        self.assertFalse(validate_knowledge_passage(invalid_domain))

        non_dict = ["not", "a", "dict"]
        self.assertFalse(validate_knowledge_passage(non_dict))

    def test_build_curated_knowledge_corpus_domains(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "corpus.jsonl")
            corpus = build_curated_knowledge_corpus(out_file)
            self.assertGreaterEqual(len(corpus), 60)
            domains = {p["domain"] for p in corpus}
            expected_domains = {
                "history", "science", "geography", "law",
                "literature", "arts", "technology", "society"
            }
            self.assertTrue(expected_domains.issubset(domains))
            self.assertTrue(os.path.exists(out_file))

            # Verify file format on disk
            with open(out_file, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]
            self.assertEqual(len(lines), len(corpus))
            for p in lines:
                self.assertTrue(validate_knowledge_passage(p))


if __name__ == "__main__":
    unittest.main()
