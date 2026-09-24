# -*- coding: utf-8 -*-
"""
Tests for Kaggle Runner Kernel Packaging and Corpus Embedding Tool.
Validates gzip+base64 compression roundtrip, corpus integrity,
and dynamic injection of embedded knowledge corpus into kernel scripts.
"""

import ast
import json
import os
import tempfile
import unittest

from scripts.package_kaggle_kernel import (
    compress_corpus_to_base64,
    decompress_corpus_from_base64,
    package_kernel_script,
)


class TestKaggleRunnerPackage(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_compress_and_decompress_roundtrip(self):
        sample_records = [
            {
                "passage_id": "test_001",
                "title": "Қазақ тілі",
                "text": "Қазақ тілі — түркі тілдерінің қыпшақ тобына жататын мемлекеттік тіл.",
                "domain": "linguistics"
            },
            {
                "passage_id": "test_002",
                "title": "Әл-Фараби мұрасы",
                "text": "Әбу Насыр әл-Фараби — Отырар қаласында туған ұлы ғалым, екінші ұстаз.",
                "domain": "philosophy"
            }
        ]
        corpus_path = os.path.join(self.temp_dir.name, "test_corpus.jsonl")
        with open(corpus_path, "w", encoding="utf-8") as f:
            for rec in sample_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        b64_str = compress_corpus_to_base64(corpus_path)
        self.assertIsInstance(b64_str, str)
        self.assertGreater(len(b64_str), 0)

        decompressed = decompress_corpus_from_base64(b64_str)
        self.assertEqual(len(decompressed), 2)
        self.assertEqual(decompressed, sample_records)

    def test_compress_default_corpus(self):
        b64_str = compress_corpus_to_base64()
        self.assertIsInstance(b64_str, str)
        self.assertGreater(len(b64_str), 1000)

        decompressed = decompress_corpus_from_base64(b64_str)
        self.assertEqual(len(decompressed), 72)
        expected_keys = {"passage_id", "title", "text", "domain"}
        domains = set()
        for doc in decompressed:
            self.assertTrue(expected_keys.issubset(doc.keys()))
            self.assertGreater(len(doc["text"].strip()), 0)
            domains.add(doc["domain"])

        self.assertIn("history", domains)
        self.assertIn("science", domains)
        self.assertIn("literature", domains)
        self.assertEqual(len(domains), 8)

    def test_compress_missing_file_raises(self):
        non_existent = os.path.join(self.temp_dir.name, "does_not_exist.jsonl")
        with self.assertRaises(FileNotFoundError):
            compress_corpus_to_base64(non_existent)

    def test_decompress_empty_payload(self):
        records = decompress_corpus_from_base64("")
        self.assertEqual(records, [])

    def test_package_kernel_script_replaces_existing_placeholder(self):
        template_code = (
            "# -*- coding: utf-8 -*-\n"
            '"""Kernel header docstring."""\n\n'
            "import os\n"
            "import json\n\n"
            'EMBEDDED_KNOWLEDGE_CORPUS_B64 = ""\n\n'
            "print('Ready')\n"
        )
        template_path = os.path.join(self.temp_dir.name, "template_kernel.py")
        output_path = os.path.join(self.temp_dir.name, "output_kernel.py")

        with open(template_path, "w", encoding="utf-8") as f:
            f.write(template_code)

        dummy_b64 = "SGVsYWxvV29ybGQ="
        package_kernel_script(template_path, output_path, dummy_b64)

        self.assertTrue(os.path.exists(output_path))
        with open(output_path, "r", encoding="utf-8") as f:
            packaged_content = f.read()

        self.assertIn(f'EMBEDDED_KNOWLEDGE_CORPUS_B64 = "{dummy_b64}"', packaged_content)
        parsed = ast.parse(packaged_content)
        self.assertIsNotNone(parsed)

    def test_package_kernel_script_inserts_when_no_placeholder(self):
        template_code = (
            "# -*- coding: utf-8 -*-\n"
            '"""Kernel header docstring."""\n\n'
            "import os\n"
            "import json\n\n"
            "print('No placeholder')\n"
        )
        template_path = os.path.join(self.temp_dir.name, "template_kernel_raw.py")
        output_path = os.path.join(self.temp_dir.name, "sub_dir", "output_kernel_raw.py")

        with open(template_path, "w", encoding="utf-8") as f:
            f.write(template_code)

        dummy_b64 = "S2F6YWtoQ29ycHVz"
        package_kernel_script(template_path, output_path, dummy_b64)

        self.assertTrue(os.path.exists(output_path))
        with open(output_path, "r", encoding="utf-8") as f:
            packaged_content = f.read()

        self.assertIn(f'EMBEDDED_KNOWLEDGE_CORPUS_B64 = "{dummy_b64}"', packaged_content)
        parsed = ast.parse(packaged_content)
        self.assertIsNotNone(parsed)

    def test_package_kernel_script_missing_template_raises(self):
        non_existent = os.path.join(self.temp_dir.name, "missing_kernel.py")
        output_path = os.path.join(self.temp_dir.name, "out.py")
        with self.assertRaises(FileNotFoundError):
            package_kernel_script(non_existent, output_path, "dummy_b64")


if __name__ == "__main__":
    unittest.main()
