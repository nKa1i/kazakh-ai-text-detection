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
from unittest.mock import MagicMock, patch

from scripts.push_to_kaggle import (
    prepare_and_push,
    check_kernel_status,
)
from scripts.package_kaggle_kernel import (
    compress_corpus_to_base64,
    decompress_corpus_from_base64,
    package_kernel_script,
)
from kaggle_runner.generate_fever_and_social_kernel import (
    EMBEDDED_KNOWLEDGE_CORPUS_B64,
    unpack_embedded_corpus,
    generate_aspect_prompt,
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


class TestKernelAspectGenerationAndUnpacking(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_embedded_corpus_constant_defined(self):
        self.assertIsInstance(EMBEDDED_KNOWLEDGE_CORPUS_B64, str)

    def test_unpack_embedded_corpus_decompresses_and_writes_file(self):
        sample_records = [
            {
                "passage_id": "test_unpack_001",
                "title": "Сәтбаев мұрасы",
                "text": "Қаныш Сәтбаев — геология ғылымының негізін қалаған академик ғалым.",
                "domain": "science"
            },
            {
                "passage_id": "test_unpack_002",
                "title": "Күлтегін ескерткіші",
                "text": "Күлтегін ескерткіші — көне түркі жазба мәдениетінің бірегей жәдігері.",
                "domain": "history"
            }
        ]
        source_jsonl = os.path.join(self.temp_dir.name, "source.jsonl")
        with open(source_jsonl, "w", encoding="utf-8") as f:
            for rec in sample_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        payload_b64 = compress_corpus_to_base64(source_jsonl)
        target_path = os.path.join(self.temp_dir.name, "target_corpus.jsonl")

        self.assertFalse(os.path.exists(target_path))
        unpacked_articles = unpack_embedded_corpus(payload_b64, target_path)

        self.assertTrue(os.path.exists(target_path))
        self.assertEqual(len(unpacked_articles), 2)
        self.assertEqual(unpacked_articles, sample_records)

        with open(target_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(lines, sample_records)

    def test_unpack_embedded_corpus_existing_file_preservation(self):
        existing_records = [
            {"passage_id": "existing_001", "title": "Бұрынғы файл", "text": "Мазмұн", "domain": "test"}
        ]
        target_path = os.path.join(self.temp_dir.name, "already_existing.jsonl")
        with open(target_path, "w", encoding="utf-8") as f:
            for rec in existing_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        dummy_new_records = [
            {"passage_id": "new_001", "title": "Жаңа файл", "text": "Жаңа мазмұн", "domain": "test"}
        ]
        dummy_source = os.path.join(self.temp_dir.name, "dummy.jsonl")
        with open(dummy_source, "w", encoding="utf-8") as f:
            for rec in dummy_new_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        dummy_b64 = compress_corpus_to_base64(dummy_source)

        unpacked = unpack_embedded_corpus(dummy_b64, target_path)
        self.assertEqual(unpacked, existing_records)

    def test_unpack_embedded_corpus_empty_payload(self):
        target_path = os.path.join(self.temp_dir.name, "non_existent.jsonl")
        result = unpack_embedded_corpus("", target_path)
        self.assertEqual(result, [])
        self.assertFalse(os.path.exists(target_path))

    def test_generate_aspect_prompt_aspect_0_entity_event(self):
        title = "Қазақ хандығы"
        text = "1465 жылы Керей мен Жәнібек хандар Қазақ хандығының негізін қалады."
        prompt = generate_aspect_prompt(title, text, 0)

        self.assertIn(title, prompt)
        self.assertIn(text, prompt)
        self.assertIn("SUPPORTS", prompt)
        self.assertIn("REFUTES", prompt)
        self.assertIn("NOT_ENOUGH_INFO", prompt)
        self.assertIn("evidence_sentence", prompt)
        self.assertTrue(
            "тұлға" in prompt.lower() or "оқиға" in prompt.lower() or "entity" in prompt.lower(),
            "Prompt for aspect 0 must focus on entity/event grounding"
        )

    def test_generate_aspect_prompt_aspect_1_numerical_chronological(self):
        title = "Байқоңыр ғарыш айлағы"
        text = "Байқоңыр айлағының құрылысы 1955 жылы басталды. 1961 жылы Гагарин ғарышқа ұшты."
        prompt = generate_aspect_prompt(title, text, 1)

        self.assertIn(title, prompt)
        self.assertIn(text, prompt)
        self.assertIn("SUPPORTS", prompt)
        self.assertIn("REFUTES", prompt)
        self.assertIn("NOT_ENOUGH_INFO", prompt)
        self.assertIn("evidence_sentence", prompt)
        self.assertTrue(
            "жыл" in prompt.lower() or "сан" in prompt.lower() or "уақыт" in prompt.lower() or "numerical" in prompt.lower() or "мерзім" in prompt.lower(),
            "Prompt for aspect 1 must focus on numerical/chronological grounding"
        )

    def test_generate_aspect_prompt_aspect_2_causal_relational(self):
        title = "Абай Құнанбайұлы"
        text = "Абайдың Қара сөздері қазақ қоғамының рухани дамуы мен ағартушылық идеяларына арналған."
        prompt = generate_aspect_prompt(title, text, 2)

        self.assertIn(title, prompt)
        self.assertIn(text, prompt)
        self.assertIn("SUPPORTS", prompt)
        self.assertIn("REFUTES", prompt)
        self.assertIn("NOT_ENOUGH_INFO", prompt)
        self.assertIn("evidence_sentence", prompt)
        self.assertTrue(
            "себеп" in prompt.lower() or "байланыс" in prompt.lower() or "қасиет" in prompt.lower() or "causal" in prompt.lower() or "салдар" in prompt.lower(),
            "Prompt for aspect 2 must focus on causal/relational/attribute grounding"
        )

    def test_generate_aspect_prompt_modulo_cycling(self):
        title = "Сынақ тақырыбы"
        text = "Сынақ мәтіні осында берілген."
        prompt_0 = generate_aspect_prompt(title, text, 0)
        prompt_3 = generate_aspect_prompt(title, text, 3)
        self.assertEqual(prompt_0, prompt_3)

        prompt_1 = generate_aspect_prompt(title, text, 1)
        prompt_4 = generate_aspect_prompt(title, text, 4)
        self.assertEqual(prompt_1, prompt_4)


class TestKagglePushRunner(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("subprocess.run")
    def test_prepare_and_push_dry_run_updates_metadata_and_packages_kernel(self, mock_subproc):
        success = prepare_and_push(dry_run=True)
        self.assertTrue(success)

        metadata_path = os.path.join("kaggle_runner", "kernel-metadata.json")
        self.assertTrue(os.path.exists(metadata_path))
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.assertEqual(metadata.get("code_file"), "generate_fever_and_social_kernel.py")
        self.assertEqual(metadata.get("enable_gpu"), True)
        self.assertEqual(metadata.get("accelerator"), "gpu_t4_x2")
        self.assertEqual(metadata.get("enable_internet"), True)

        kernel_script_path = os.path.join("kaggle_runner", "generate_fever_and_social_kernel.py")
        self.assertTrue(os.path.exists(kernel_script_path))
        with open(kernel_script_path, "r", encoding="utf-8") as f:
            kernel_content = f.read()
        self.assertIn('EMBEDDED_KNOWLEDGE_CORPUS_B64 = "', kernel_content)

        for call_args in mock_subproc.call_args_list:
            cmd = call_args[0][0] if call_args[0] else []
            self.assertNotIn("push", cmd)

    @patch("subprocess.run")
    def test_prepare_and_push_executes_push_when_not_dry_run(self, mock_subproc):
        mock_subproc.return_value = MagicMock(returncode=0, stdout="Kernel successfully pushed")

        success = prepare_and_push(dry_run=False)
        self.assertTrue(success)

        called_push = False
        for call_args in mock_subproc.call_args_list:
            cmd = call_args[0][0] if call_args[0] else []
            if len(cmd) >= 3 and cmd[0] == "kaggle" and cmd[1] == "kernels" and cmd[2] == "push":
                called_push = True
                self.assertIn("-p", cmd)
                self.assertIn("kaggle_runner", cmd)
        self.assertTrue(called_push, "subprocess.run should have been called with kaggle kernels push")

    @patch("subprocess.run")
    def test_prepare_and_push_handles_push_failure(self, mock_subproc):
        import subprocess
        mock_subproc.side_effect = subprocess.CalledProcessError(1, cmd=["kaggle"], stderr="Push error")

        success = prepare_and_push(dry_run=False)
        self.assertFalse(success)

    @patch("subprocess.run")
    def test_check_kernel_status_success(self, mock_subproc):
        mock_subproc.return_value = MagicMock(
            returncode=0,
            stdout="dauletanekesh/kazakh-gpu-runner-nb has status 'KernelWorkerStatus.RUNNING'"
        )
        status = check_kernel_status("dauletanekesh/kazakh-gpu-runner-nb")
        self.assertIn("RUNNING", status)


if __name__ == "__main__":
    unittest.main()

