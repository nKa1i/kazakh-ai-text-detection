# -*- coding: utf-8 -*-
"""
Tests for Autonomous Kaggle GPU Training Packager and Runner.
Validates dataset compression round-trip, kernel script packaging,
self-contained training and evaluation dry-run execution, and artifact generation.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from scripts.package_nli_kaggle_kernel import (
    compress_dataset_to_base64,
    decompress_dataset_from_base64,
    package_nli_kernel,
)


class TestPackageNLIKaggleKernel(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_compress_and_decompress_roundtrip(self):
        sample_records = [
            {
                "id": "c1",
                "claim": "Қазақстан 1991 жылы тәуелсіздік алды.",
                "label": "SUPPORTED",
                "article_title": "Қазақстан тәуелсіздігі",
                "evidence_sentences": ["Қазақстан 1991 жылы өз тәуелсіздігін жариялады."],
            },
            {
                "id": "c2",
                "claim": "Астана 1995 жылы елорда болды.",
                "label": "REFUTES",
                "article_title": "Астана",
                "evidence_sentences": ["Астана 1997 жылы елорда болып жарияланды."],
            },
            {
                "id": "c3",
                "claim": "Қазақстанда марс базасы салынды.",
                "label": "NOT_ENOUGH_INFO",
                "article_title": "Байқоңыр",
                "evidence_sentences": [],
            },
        ]
        test_file = os.path.join(self.temp_dir.name, "sample_fever.jsonl")
        with open(test_file, "w", encoding="utf-8") as f:
            for rec in sample_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        b64_str = compress_dataset_to_base64(test_file)
        self.assertIsInstance(b64_str, str)
        self.assertGreater(len(b64_str), 0)

        decompressed = decompress_dataset_from_base64(b64_str)
        self.assertEqual(len(decompressed), 3)
        self.assertEqual(decompressed, sample_records)

    def test_compress_missing_file_raises(self):
        non_existent = os.path.join(self.temp_dir.name, "non_existent.jsonl")
        with self.assertRaises(FileNotFoundError):
            compress_dataset_to_base64(non_existent)

    def test_decompress_empty_string(self):
        records = decompress_dataset_from_base64("")
        self.assertEqual(records, [])

    def test_package_nli_kernel_dry_run(self):
        out_script = os.path.join(self.temp_dir.name, "train_nli_kernel.py")
        res = package_nli_kernel(output_path=out_script, dry_run=True)
        self.assertTrue(os.path.exists(out_script))
        self.assertEqual(os.path.abspath(res), os.path.abspath(out_script))

        with open(out_script, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("EMBEDDED_TRAIN_B64", content)
        self.assertIn("EMBEDDED_DEV_B64", content)
        self.assertIn("EMBEDDED_TEST_B64", content)
        self.assertIn("EMBEDDED_CORPUS_B64", content)
        self.assertIn("MorphoNLIVerifier", content)

        parsed = ast.parse(content)
        self.assertIsNotNone(parsed)

    def test_package_nli_kernel_full_packaging(self):
        out_script = os.path.join(self.temp_dir.name, "packaged_train_nli_kernel.py")
        res = package_nli_kernel(output_path=out_script, dry_run=False, push=False)
        self.assertTrue(os.path.exists(out_script))

        with open(out_script, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertNotIn("__TRAIN_DATA_B64__", content)
        self.assertNotIn("__DEV_DATA_B64__", content)
        self.assertNotIn("__TEST_DATA_B64__", content)
        self.assertNotIn("__CORPUS_DATA_B64__", content)

        train_match = re.search(r'EMBEDDED_TRAIN_B64\s*=\s*["\']([^"\']+)["\']', content)
        self.assertIsNotNone(train_match)
        train_b64 = train_match.group(1)
        train_records = decompress_dataset_from_base64(train_b64)
        self.assertEqual(len(train_records), 638)

        dev_match = re.search(r'EMBEDDED_DEV_B64\s*=\s*["\']([^"\']+)["\']', content)
        self.assertIsNotNone(dev_match)
        dev_b64 = dev_match.group(1)
        dev_records = decompress_dataset_from_base64(dev_b64)
        self.assertEqual(len(dev_records), 135)

        test_match = re.search(r'EMBEDDED_TEST_B64\s*=\s*["\']([^"\']+)["\']', content)
        self.assertIsNotNone(test_match)
        test_b64 = test_match.group(1)
        test_records = decompress_dataset_from_base64(test_b64)
        self.assertEqual(len(test_records), 141)

        corpus_match = re.search(r'EMBEDDED_CORPUS_B64\s*=\s*["\']([^"\']+)["\']', content)
        self.assertIsNotNone(corpus_match)
        corpus_b64 = corpus_match.group(1)
        corpus_records = decompress_dataset_from_base64(corpus_b64)
        self.assertEqual(len(corpus_records), 108)

    @patch("subprocess.run")
    def test_package_nli_kernel_push(self, mock_subproc):
        mock_subproc.return_value = MagicMock(returncode=0, stdout="Kernel successfully pushed")

        kernel_dir = os.path.join(self.temp_dir.name, "kaggle_kernel_push")
        out_script = os.path.join(kernel_dir, "train_nli_kernel.py")

        res = package_nli_kernel(output_path=out_script, dry_run=False, push=True)
        self.assertTrue(os.path.exists(out_script))

        metadata_path = os.path.join(kernel_dir, "kernel-metadata.json")
        self.assertTrue(os.path.exists(metadata_path))
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.assertEqual(metadata.get("code_file"), "train_nli_kernel.py")
        self.assertTrue(metadata.get("enable_gpu"))
        self.assertTrue(metadata.get("enable_internet"))

        called_push = False
        for call_args in mock_subproc.call_args_list:
            cmd = call_args[0][0] if call_args[0] else []
            if len(cmd) >= 3 and cmd[0] == "kaggle" and cmd[1] == "kernels" and cmd[2] == "push":
                called_push = True
                self.assertIn("-p", cmd)
                self.assertIn(kernel_dir, cmd)
        self.assertTrue(called_push)

    def test_train_nli_kernel_unpack_and_dry_run_execution(self):
        from kaggle_runner.train_nli_kernel import (
            unpack_embedded_datasets,
            prepare_nli_pairs,
            run_kernel_pipeline,
        )

        unpacked = unpack_embedded_datasets()
        self.assertIn("train", unpacked)
        self.assertIn("dev", unpacked)
        self.assertIn("test", unpacked)
        self.assertIn("corpus", unpacked)

        pairs = prepare_nli_pairs(unpacked["test"][:5], unpacked["corpus"])
        self.assertGreater(len(pairs), 0)
        self.assertIn("claim", pairs[0])
        self.assertIn("evidence", pairs[0])
        self.assertIn("label", pairs[0])

        out_tex = os.path.join(self.temp_dir.name, "output", "table2_verification_main_results.tex")
        out_json = os.path.join(self.temp_dir.name, "output", "verification_benchmark_results.json")

        benchmark_results = run_kernel_pipeline(
            dry_run=True,
            output_latex=out_tex,
            output_json=out_json,
        )
        self.assertIn("mBERT-base", benchmark_results)
        self.assertIn("XLM-RoBERTa-base", benchmark_results)
        self.assertIn("Ours (Hybrid + Morpho)", benchmark_results)

        self.assertTrue(os.path.exists(out_tex))
        self.assertTrue(os.path.exists(out_json))

        with open(out_tex, "r", encoding="utf-8") as f:
            tex_content = f.read()
        self.assertIn(r"\begin{table}", tex_content)
        self.assertIn("tab:main_results", tex_content)
        self.assertIn("mBERT-base", tex_content)
        self.assertIn(r"\textbf{Ours (Hybrid + Morpho)}", tex_content)

        with open(out_json, "r", encoding="utf-8") as f:
            json_content = json.load(f)
        self.assertIn("mBERT-base", json_content)
        self.assertIn("accuracy", json_content["mBERT-base"])
        self.assertIn("macro_f1", json_content["mBERT-base"])

    def test_zero_emojis_and_no_placeholders(self):
        emoji_pattern = re.compile(
            r"[\U00010000-\U0010ffff]|[\u2600-\u27bf]|[\u2300-\u23ff]"
        )
        files_to_check = [
            "scripts/package_nli_kaggle_kernel.py",
            "kaggle_runner/train_nli_kernel.py",
            "tests/test_package_nli_kaggle_kernel.py",
        ]
        for rel_path in files_to_check:
            if not os.path.exists(rel_path):
                continue
            with open(rel_path, "r", encoding="utf-8") as f:
                content = f.read()
            emojis_found = emoji_pattern.findall(content)
            self.assertEqual(
                emojis_found,
                [],
                f"Emojis found in {rel_path}: {emojis_found}",
            )
            forbidden = ["TO" + "DO", "TB" + "D"]
            for bad_word in forbidden:
                self.assertNotIn(bad_word, content, f"{bad_word} found in {rel_path}")


if __name__ == "__main__":
    unittest.main()
