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

    def test_pytorch_nli_wrapper_inference(self):
        from kaggle_runner.train_nli_kernel import PyTorchNLIWrapper, HAS_TORCH

        class MockTokenizer:
            def __call__(self, text, text_pair=None, **kwargs):
                return {
                    "input_ids": [101, 102] if not HAS_TORCH else __import__("torch").tensor([[101, 102]]),
                    "attention_mask": [1, 1] if not HAS_TORCH else __import__("torch").tensor([[1, 1]]),
                }

        class MockOutputs:
            def __init__(self, logits):
                self.logits = logits

        class MockModel:
            def __init__(self):
                self.training = True

            def eval(self):
                self.training = False
                return self

            def __call__(self, *args, **kwargs):
                if HAS_TORCH:
                    import torch
                    return MockOutputs(torch.tensor([[2.5, -1.0, 0.1]]))
                return MockOutputs([[2.5, -1.0, 0.1]])

        wrapper = PyTorchNLIWrapper(
            model=MockModel(),
            tokenizer=MockTokenizer(),
            device="cpu",
            is_morpho=False,
        )
        res = wrapper.predict_pair("Астана – Қазақстан астанасы.", "Астана 1997 жылдан бері бас қала.")
        self.assertIsInstance(res, dict)
        self.assertIn("label", res)
        self.assertEqual(res["label"], "SUPPORTED")

        preds = wrapper.predict([{"claim": "Астана – Қазақстан астанасы.", "evidence": "Бас қала."}])
        self.assertEqual(len(preds), 1)
        self.assertEqual(preds[0], "SUPPORTED")

    def test_evaluate_nli_test_suite_with_pytorch_wrapper(self):
        from kaggle_runner.train_nli_kernel import (
            PyTorchNLIWrapper,
            evaluate_nli_test_suite,
            HAS_TORCH,
        )

        class MockTokenizer:
            def __call__(self, text, text_pair=None, **kwargs):
                return {
                    "input_ids": [101, 102] if not HAS_TORCH else __import__("torch").tensor([[101, 102]]),
                    "attention_mask": [1, 1] if not HAS_TORCH else __import__("torch").tensor([[1, 1]]),
                }

        class MockOutputs:
            def __init__(self, logits):
                self.logits = logits

        class MockModel:
            def eval(self):
                return self

            def __call__(self, *args, **kwargs):
                if HAS_TORCH:
                    import torch
                    return MockOutputs(torch.tensor([[0.1, 3.0, -0.5]]))
                return MockOutputs([[0.1, 3.0, -0.5]])

        wrapper = PyTorchNLIWrapper(
            model=MockModel(),
            tokenizer=MockTokenizer(),
            device="cpu",
            is_morpho=False,
        )
        test_pairs = [
            {"claim": "Тест талап 1", "evidence": "Дәйексөз 1", "label": "REFUTES"},
            {"claim": "Тест талап 2", "evidence": "Дәйексөз 2", "label": "NOT_ENOUGH_INFO"},
        ]
        results = evaluate_nli_test_suite({"WrappedNeuralModel": wrapper}, test_pairs)
        self.assertIn("WrappedNeuralModel", results)
        self.assertIn("accuracy", results["WrappedNeuralModel"])
        self.assertIn("macro_f1", results["WrappedNeuralModel"])
        self.assertIn("fever_score", results["WrappedNeuralModel"])
        self.assertIn("hard_nei_f1", results["WrappedNeuralModel"])

    def test_kernel_compute_class_weights_and_scheduler(self):
        from kaggle_runner.train_nli_kernel import (
            compute_class_weights,
            get_cosine_schedule_with_warmup,
            get_parameter_groups,
            train_gpu_epoch,
        )

        sample_pairs = [
            {"label": "SUPPORTED"},
            {"label": "SUPPORTED"},
            {"label": "REFUTES"},
            {"label": "NOT_ENOUGH_INFO"},
            {"label": "NOT_ENOUGH_INFO"},
            {"label": "NOT_ENOUGH_INFO"},
        ]
        weights = compute_class_weights(sample_pairs)
        self.assertEqual(len(weights), 3)
        self.assertAlmostEqual(sum(weights), 3.0, places=5)
        # REFUTES is rarest (1 count), so highest weight
        self.assertGreater(weights[1], weights[0])
        self.assertGreater(weights[0], weights[2])

        class MockOpt:
            def __init__(self):
                self.param_groups = [{"lr": 2e-5}]

        opt = MockOpt()
        sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=5, num_training_steps=50)
        self.assertIsNotNone(sched)
        self.assertTrue(hasattr(sched, "step"))
        sched.step()

        class MockParam:
            def __init__(self, req=True):
                self.requires_grad = req

        class MockM:
            def named_parameters(self):
                return [
                    ("encoder.layer.weight", MockParam(True)),
                    ("classifier.weight", MockParam(True)),
                ]

        groups = get_parameter_groups(MockM(), lr_encoder=2e-5, lr_head=2e-4, is_morpho=True)
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["lr"], 2e-5)
        self.assertEqual(groups[1]["lr"], 2e-4)

        loss = train_gpu_epoch(
            model=None,
            dataloader=None,
            optimizer=None,
            scaler=None,
            device="cpu",
            use_morpho=False,
            scheduler=sched,
            class_weights=weights,
        )
        self.assertEqual(loss, 0.0)

    def test_kernel_evaluate_dev_epoch(self):
        from kaggle_runner.train_nli_kernel import evaluate_dev_epoch

        res = evaluate_dev_epoch(None, None, criterion=None, device="cpu", is_morpho=False)
        self.assertIsInstance(res, dict)
        self.assertIn("loss", res)
        self.assertIn("accuracy", res)
        self.assertIn("macro_f1", res)
        self.assertEqual(res["loss"], 0.0)
        self.assertEqual(res["accuracy"], 0.0)
        self.assertEqual(res["macro_f1"], 0.0)

    def test_kernel_training_history_telemetry_export(self):
        from kaggle_runner.train_nli_kernel import run_kernel_pipeline

        out_tex = os.path.join(self.temp_dir.name, "output", "table2.tex")
        out_json = os.path.join(self.temp_dir.name, "output", "results.json")
        out_hist = os.path.join(self.temp_dir.name, "output", "training_history.json")

        run_kernel_pipeline(
            dry_run=True,
            output_latex=out_tex,
            output_json=out_json,
            output_history=out_hist,
            epochs=3,
        )

        self.assertTrue(os.path.exists(out_hist), f"Telemetry file {out_hist} was not created")
        with open(out_hist, "r", encoding="utf-8") as f:
            history = json.load(f)

        self.assertIsInstance(history, list)
        self.assertEqual(len(history), 3)

        for entry in history:
            self.assertIn("epoch", entry)
            self.assertIn("train_loss", entry)
            self.assertIn("dev_loss", entry)
            self.assertIn("dev_accuracy", entry)
            self.assertIn("dev_macro_f1", entry)
            self.assertIn("best_epoch", entry)

            self.assertIsInstance(entry["epoch"], int)
            self.assertIsInstance(entry["best_epoch"], int)
            self.assertIsInstance(entry["train_loss"], (int, float))
            self.assertIsInstance(entry["dev_loss"], (int, float))
            self.assertIsInstance(entry["dev_accuracy"], (int, float))
            self.assertIsInstance(entry["dev_macro_f1"], (int, float))

    def test_kernel_best_checkpoint_restoration(self):
        import copy

        class MockWeightsModel:
            def __init__(self):
                self.weights = {"param": 0}

            def state_dict(self):
                return copy.deepcopy(self.weights)

            def load_state_dict(self, state_dict):
                self.weights = copy.deepcopy(state_dict)

        model = MockWeightsModel()
        epochs_dev_f1 = [72.0, 84.5, 79.0]
        best_dev_macro_f1 = -1.0
        best_epoch = 0
        best_state_dict = None

        for ep, dev_f1 in enumerate(epochs_dev_f1):
            model.weights["param"] = ep + 1
            if dev_f1 > best_dev_macro_f1:
                best_dev_macro_f1 = dev_f1
                best_epoch = ep + 1
                best_state_dict = copy.deepcopy(model.state_dict())

        self.assertEqual(best_epoch, 2)
        self.assertEqual(best_state_dict["param"], 2)

        # In epoch 3, weights became 3
        self.assertEqual(model.weights["param"], 3)

        # Restore best checkpoint
        model.load_state_dict(best_state_dict)
        self.assertEqual(model.weights["param"], 2)

    def test_kernel_morphological_feature_masking(self):
        from kaggle_runner.train_nli_kernel import MorphologicalAffixExtractor

        extractor = MorphologicalAffixExtractor()
        claim = "Қазақстан 1991 жылы тәуелсіздік алмаған еді."
        evidence = "Қазақстан 1993 жылы тәуелсіздік алды, осылай болыпты."

        raw = extractor.extract_features(claim, evidence)
        self.assertEqual(len(raw), 16)
        self.assertEqual(raw[2], 1.0)
        self.assertEqual(raw[3], 1.0)

        # Mask indices
        masked_neg = extractor.extract_features(claim, evidence, mask_indices=[2])
        self.assertEqual(masked_neg[2], 0.0)
        self.assertEqual(masked_neg[3], 1.0)

        # Ablation mode
        m_evid = extractor.extract_features(claim, evidence, ablation_mode="- Evidentials & Epistemic Modals")
        for idx in [4, 5, 6, 7]:
            self.assertEqual(m_evid[idx], 0.0)

    def test_kernel_morphological_ablation_study_and_formatting(self):
        from kaggle_runner.train_nli_kernel import (
            run_morphological_ablation_study,
            format_ablation_table_latex,
        )

        test_pairs = [
            {"claim": "Астана — елорда.", "evidence": "Астана қаласы — бас қала.", "label": "SUPPORTED"},
            {"claim": "Қазақстан 1998 жылы тәуелсіздік алды.", "evidence": "Қазақстан 1991 жылы тәуелсіздік алды.", "label": "REFUTES"},
            {"claim": "Айда қала бар.", "evidence": "Қазақстан дамып келеді.", "label": "NOT_ENOUGH_INFO"},
        ]

        ablation_results = run_morphological_ablation_study(test_pairs=test_pairs)
        self.assertEqual(len(ablation_results), 6)
        self.assertIn("Ours (Full Morpho Cross-Encoder)", ablation_results)
        self.assertIn("- Negation Alignment", ablation_results)
        self.assertIn("- Temporal / Calendar Conflicts", ablation_results)
        self.assertIn("- Evidentials & Epistemic Modals", ablation_results)
        self.assertIn("- FST Root Analysis", ablation_results)
        self.assertIn("Surface Token Overlap Only", ablation_results)

        latex = format_ablation_table_latex(ablation_results)
        self.assertIn(r"\label{tab:ablation_results}", latex)
        self.assertIn("Ours (Full Morpho Cross-Encoder)", latex)

    def test_kernel_pipeline_with_ablation_outputs(self):
        from kaggle_runner.train_nli_kernel import run_kernel_pipeline

        out_tex = os.path.join(self.temp_dir.name, "output", "table2.tex")
        out_json = os.path.join(self.temp_dir.name, "output", "results.json")
        out_abl_tex = os.path.join(self.temp_dir.name, "output", "table_ablation_results.tex")
        out_abl_json = os.path.join(self.temp_dir.name, "output", "ablation_results.json")

        run_kernel_pipeline(
            dry_run=True,
            output_latex=out_tex,
            output_json=out_json,
            output_ablation_latex=out_abl_tex,
            output_ablation_json=out_abl_json,
        )

        self.assertTrue(os.path.exists(out_abl_tex))
        self.assertTrue(os.path.exists(out_abl_json))

        with open(out_abl_tex, "r", encoding="utf-8") as f:
            tex_content = f.read()
        self.assertIn(r"\label{tab:ablation_results}", tex_content)
        self.assertIn("Ours (Full Morpho Cross-Encoder)", tex_content)


if __name__ == "__main__":
    unittest.main()
