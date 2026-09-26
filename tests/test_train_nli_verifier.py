# -*- coding: utf-8 -*-
"""
tests/test_train_nli_verifier.py: Unit tests for NLI Claim-Evidence Dataset Pairer & Evaluator.
"""

import os
import sys
import json
import tempfile
import unittest
from pathlib import Path
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_nli_verifier import (
    prepare_nli_pairs,
    train_nli_classifier,
    evaluate_nli_test_suite,
    format_table2_latex,
    normalize_label,
    compute_class_weights,
    get_cosine_schedule_with_warmup,
    get_parameter_groups,
    train_gpu_epoch,
    evaluate_dev_epoch,
    run_morphological_ablation_study,
    format_ablation_table_latex,
)



class TestTrainNLIVerifier(unittest.TestCase):
    def setUp(self):
        self.sample_corpus = {
            "Қазақстан тәуелсіздігі": (
                "Қазақстан 1991 жылы өз тәуелсіздігін жариялады. "
                "Тәуелсіздік Конституциялық заңмен бекітілді. "
                "Астана қаласы мемлекеттің жаңа елордасы болды."
            ),
            "Абай Құнанбайұлы": (
                "Абай Құнанбайұлы — ұлы қазақ ақыны және ойшылы. "
                "Ол Семей өңірінде дүниеге келген."
            ),
        }

        self.sample_claims = [
            {
                "id": "c1",
                "claim": "Қазақстан 1991 жылы тәуелсіздік алды.",
                "label": "SUPPORTED",
                "domain": "history",
                "article_title": "Қазақстан тәуелсіздігі",
                "evidence_sentences": ["Қазақстан 1991 жылы өз тәуелсіздігін жариялады."],
            },
            {
                "id": "c2",
                "claim": "Қазақстан 1998 жылы тәуелсіздік алды.",
                "label": "REFUTES",
                "domain": "history",
                "article_title": "Қазақстан тәуелсіздігі",
                "evidence_sentences": ["Қазақстан 1991 жылы өз тәуелсіздігін жариялады."],
            },
            {
                "id": "c3",
                "claim": "Қазақстан 1991 жылы ғарыш станциясын сатып алды.",
                "label": "NOT_ENOUGH_INFO",
                "domain": "history",
                "article_title": "Қазақстан тәуелсіздігі",
                "evidence_sentences": [],
            },
        ]

    def test_normalize_label(self):
        self.assertEqual(normalize_label("SUPPORTS"), "SUPPORTED")
        self.assertEqual(normalize_label("SUPPORTED"), "SUPPORTED")
        self.assertEqual(normalize_label("REFUTES"), "REFUTES")
        self.assertEqual(normalize_label("REFUTED"), "REFUTES")
        self.assertEqual(normalize_label("NOT_ENOUGH_INFO"), "NOT_ENOUGH_INFO")
        self.assertEqual(normalize_label("NEI"), "NOT_ENOUGH_INFO")

    def test_prepare_nli_pairs_format_and_labels(self):
        pairs = prepare_nli_pairs(self.sample_claims, self.sample_corpus)
        self.assertEqual(len(pairs), 3)

        for pair in pairs:
            self.assertIn("id", pair)
            self.assertIn("claim", pair)
            self.assertIn("evidence", pair)
            self.assertIn("label", pair)
            self.assertIn("domain", pair)
            self.assertIn("article_title", pair)
            self.assertIn("evidence_sentences", pair)
            self.assertIsInstance(pair["evidence"], str)
            self.assertGreater(len(pair["evidence"]), 0)

        self.assertEqual(pairs[0]["label"], "SUPPORTED")
        self.assertEqual(pairs[1]["label"], "REFUTES")
        self.assertEqual(pairs[2]["label"], "NOT_ENOUGH_INFO")

        # For NOT_ENOUGH_INFO, evidence should be retrieved distractor sentences from article
        self.assertIn("Қазақстан", pairs[2]["evidence"])

    def test_prepare_nli_pairs_supports_alias(self):
        claims_with_supports = [
            {
                "id": "c4",
                "claim": "Абай — қазақ ақыны.",
                "label": "SUPPORTS",
                "article_title": "Абай Құнанбайұлы",
                "evidence_sentences": ["Абай Құнанбайұлы — ұлы қазақ ақыны және ойшылы."],
            }
        ]
        pairs = prepare_nli_pairs(claims_with_supports, self.sample_corpus)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["label"], "SUPPORTED")
        self.assertEqual(
            pairs[0]["evidence"],
            "Абай Құнанбайұлы — ұлы қазақ ақыны және ойшылы.",
        )

    def test_prepare_nli_pairs_fallback_without_evidence_sentences(self):
        claims_no_ev = [
            {
                "id": "c5",
                "claim": "Қазақстан елордасы Астана.",
                "label": "SUPPORTED",
                "article_title": "Қазақстан тәуелсіздігі",
                "evidence_sentences": [],
            }
        ]
        pairs = prepare_nli_pairs(claims_no_ev, self.sample_corpus)
        self.assertEqual(len(pairs), 1)
        self.assertTrue(len(pairs[0]["evidence"]) > 0)

    def test_train_nli_classifier_dry_run(self):
        train_pairs = prepare_nli_pairs(self.sample_claims, self.sample_corpus)
        dev_pairs = prepare_nli_pairs(self.sample_claims, self.sample_corpus)

        for model_type in ["ours_morpho", "mbert_base", "xlmr_base"]:
            model = train_nli_classifier(
                train_pairs=train_pairs,
                dev_pairs=dev_pairs,
                model_type=model_type,
                dry_run=True,
            )
            self.assertIsNotNone(model)
            self.assertTrue(hasattr(model, "predict_pair") or hasattr(model, "predict"))

            # Test single pair prediction
            if hasattr(model, "predict_pair"):
                res = model.predict_pair("Астана — бас қала.", "Астана — Қазақстанның елордасы.")
                self.assertIsInstance(res, dict)
                self.assertIn("label", res)
                self.assertIn(res["label"], ["SUPPORTED", "REFUTES", "NOT_ENOUGH_INFO"])

    def test_evaluate_nli_test_suite_metrics(self):
        test_pairs = [
            {
                "id": "t1",
                "claim": "Астана — елорда.",
                "evidence": "Астана қаласы — Қазақстанның елордасы.",
                "label": "SUPPORTED",
                "evidence_sentences": ["Астана қаласы — Қазақстанның елордасы."],
            },
            {
                "id": "t2",
                "claim": "Қазақстан 1998 жылы тәуелсіздік алды.",
                "evidence": "Қазақстан 1991 жылы тәуелсіздік жариялады.",
                "label": "REFUTES",
                "evidence_sentences": ["Қазақстан 1991 жылы тәуелсіздік жариялады."],
            },
            {
                "id": "t3",
                "claim": "Айда қала салынған.",
                "evidence": "Астана қаласы күннен күнге дамып келеді.",
                "label": "NOT_ENOUGH_INFO",
                "evidence_sentences": [],
            },
        ]

        class MockPerfectModel:
            def predict_pair(self, claim: str, evidence: str) -> Dict[str, Any]:
                if "елорда" in claim:
                    return {"label": "SUPPORTED", "confidence": 0.9}
                elif "1998" in claim:
                    return {"label": "REFUTES", "confidence": 0.9}
                else:
                    return {"label": "NOT_ENOUGH_INFO", "confidence": 0.8}

        models_dict = {"MockModel": MockPerfectModel()}
        retrieved_evidence = [
            ["Астана қаласы — Қазақстанның елордасы."],
            ["Қазақстан 1991 жылы тәуелсіздік жариялады."],
            [],
        ]

        results = evaluate_nli_test_suite(
            models_dict=models_dict,
            test_pairs=test_pairs,
            retrieved_evidence=retrieved_evidence,
        )

        self.assertIn("MockModel", results)
        m = results["MockModel"]
        self.assertIn("accuracy", m)
        self.assertIn("macro_f1", m)
        self.assertIn("fever_score", m)
        self.assertIn("hard_nei_f1", m)

        self.assertAlmostEqual(m["accuracy"], 100.0, places=1)
        self.assertAlmostEqual(m["macro_f1"], 100.0, places=1)
        self.assertAlmostEqual(m["fever_score"], 100.0, places=1)
        self.assertAlmostEqual(m["hard_nei_f1"], 100.0, places=1)

    def test_evaluate_nli_test_suite_partial_and_fever_condition(self):
        test_pairs = [
            {
                "id": "t1",
                "claim": "Астана — елорда.",
                "evidence": "Астана — елорда.",
                "label": "SUPPORTED",
                "evidence_sentences": ["Астана — елорда."],
            },
            {
                "id": "t2",
                "claim": "Қазақстан 1998 жылы тәуелсіздік алды.",
                "evidence": "Қазақстан 1991 жылы тәуелсіздік жариялады.",
                "label": "REFUTES",
                "evidence_sentences": ["Қазақстан 1991 жылы тәуелсіздік жариялады."],
            },
            {
                "id": "t3",
                "claim": "Айда қала салынған.",
                "evidence": "Астана дамып келеді.",
                "label": "NOT_ENOUGH_INFO",
                "evidence_sentences": [],
            },
        ]

        # Model gets all 3 labels right, but retrieved evidence misses t2
        class MockLabelModel:
            def predict_pair(self, claim: str, evidence: str) -> Dict[str, Any]:
                if "елорда" in claim:
                    return {"label": "SUPPORTED"}
                elif "1998" in claim:
                    return {"label": "REFUTES"}
                else:
                    return {"label": "NOT_ENOUGH_INFO"}

        # t2 retrieved evidence does not match gold evidence sentence
        retrieved_evidence = [
            ["Астана — елорда."],
            ["Басқа мүлдем сәйкес келмейтін сөйлем."],
            [],
        ]

        results = evaluate_nli_test_suite(
            {"MockLabelModel": MockLabelModel()},
            test_pairs,
            retrieved_evidence=retrieved_evidence,
        )
        m = results["MockLabelModel"]
        self.assertAlmostEqual(m["accuracy"], 100.0, places=1)
        # 2 out of 3 satisfy FEVER score (t1 matched label+evidence, t3 matched label; t2 failed evidence)
        self.assertAlmostEqual(m["fever_score"], 66.7, places=1)

    def test_format_table2_latex(self):
        results = {
            "mBERT-base": {
                "accuracy": 64.2,
                "macro_f1": 63.8,
                "fever_score": 51.4,
                "hard_nei_f1": 48.2,
            },
            "XLM-RoBERTa-base": {
                "accuracy": 71.8,
                "macro_f1": 71.2,
                "fever_score": 58.6,
                "hard_nei_f1": 55.7,
            },
            "Ours (Hybrid + Morpho)": {
                "accuracy": 82.6,
                "macro_f1": 82.1,
                "fever_score": 71.4,
                "hard_nei_f1": 72.8,
            },
        }
        latex = format_table2_latex(results)
        self.assertIn("\\begin{table}", latex)
        self.assertIn("tab:main_results", latex)
        self.assertIn("mBERT-base", latex)
        self.assertIn("\\textbf{Ours (Hybrid + Morpho)}", latex)
        self.assertIn("\\textbf{82.6}", latex)
        self.assertIn("\\textbf{82.1}", latex)
        self.assertIn("\\textbf{71.4}", latex)
        self.assertIn("\\textbf{72.8}", latex)
        self.assertIn("\\toprule", latex)
        self.assertIn("\\midrule", latex)
        self.assertIn("\\bottomrule", latex)
        self.assertNotIn("TO" + "DO", latex)
        self.assertNotIn("TB" + "D", latex)

    def test_edge_cases_empty_and_missing_fields(self):
        # Empty claims
        pairs = prepare_nli_pairs([], {})
        self.assertEqual(pairs, [])

        # Missing fields
        malformed = [{"id": "m1"}]
        pairs_malformed = prepare_nli_pairs(malformed, {})
        self.assertEqual(len(pairs_malformed), 1)
        self.assertEqual(pairs_malformed[0]["label"], "NOT_ENOUGH_INFO")
        self.assertEqual(pairs_malformed[0]["evidence"], "")

        # Empty test pairs evaluation
        empty_res = evaluate_nli_test_suite({}, [])
        self.assertEqual(empty_res, {})

    def test_cli_execution_dry_run(self):
        import subprocess

        with tempfile.TemporaryDirectory() as tmpdir:
            out_latex = os.path.join(tmpdir, "table2.tex")
            out_json = os.path.join(tmpdir, "results.json")

            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "train_nli_verifier.py"),
                "--train_data",
                str(PROJECT_ROOT / "data" / "kazakh_fever_train.jsonl"),
                "--dev_data",
                str(PROJECT_ROOT / "data" / "kazakh_fever_dev.jsonl"),
                "--test_data",
                str(PROJECT_ROOT / "data" / "kazakh_fever_test.jsonl"),
                "--corpus",
                str(PROJECT_ROOT / "data" / "kazakh_knowledge_corpus.jsonl"),
                "--output_latex",
                out_latex,
                "--output_json",
                out_json,
                "--dry_run",
            ]
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                cwd=str(PROJECT_ROOT),
            )
            self.assertEqual(
                res.returncode,
                0,
                f"CLI execution failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}",
            )
            self.assertTrue(os.path.exists(out_latex))
            self.assertTrue(os.path.exists(out_json))

            with open(out_latex, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("tab:main_results", content)
            self.assertIn("Ours (Hybrid + Morpho)", content)

            with open(out_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("models", data)
            self.assertIn("Ours (Hybrid + Morpho)", data["models"])

    def test_train_nli_classifier_kazroberta(self):
        train_pairs = prepare_nli_pairs(self.sample_claims, self.sample_corpus)
        dev_pairs = prepare_nli_pairs(self.sample_claims, self.sample_corpus)
        model = train_nli_classifier(
            train_pairs=train_pairs,
            dev_pairs=dev_pairs,
            model_type="kazroberta_base",
            dry_run=True,
        )
        self.assertIsNotNone(model)
        self.assertEqual(model.canonical_name, "KazRoBERTa")

    def test_classifier_predict_formats(self):
        train_pairs = prepare_nli_pairs(self.sample_claims, self.sample_corpus)
        model = train_nli_classifier(
            train_pairs=train_pairs,
            dev_pairs=None,
            model_type="ours_morpho",
            dry_run=True,
        )
        # Predict with dicts
        preds_dict = model.predict([{"claim": "Астана — елорда.", "evidence": "Астана — қала."}])
        self.assertEqual(len(preds_dict), 1)

        # Predict with tuples
        preds_tuple = model.predict([("Астана — елорда.", "Астана — қала.")])
        self.assertEqual(len(preds_tuple), 1)

        # Predict with strings
        preds_str = model.predict(["Астана — елорда."])
        self.assertEqual(len(preds_str), 1)

    def test_format_table2_latex_fractional_inputs(self):
        # When values are passed as fractions in [0.0, 1.0]
        results = {
            "mBERT-base": {
                "accuracy": 0.642,
                "macro_f1": 0.638,
                "fever_score": 0.514,
                "hard_nei_f1": 0.482,
            },
            "Ours (Hybrid + Morpho)": {
                "accuracy": 0.826,
                "macro_f1": 0.821,
                "fever_score": 0.714,
                "hard_nei_f1": 0.728,
            },
        }
        latex = format_table2_latex(results)
        self.assertIn("64.2", latex)
        self.assertIn("63.8", latex)
        self.assertIn("\\textbf{82.6}", latex)
        self.assertIn("\\textbf{72.8}", latex)

    def test_evaluate_nli_test_suite_precomputed(self):
        precomputed = {
            "CustomModel": {
                "accuracy": 85.0,
                "macro_f1": 84.5,
                "fever_score": 75.0,
                "hard_nei_f1": 70.0,
            }
        }
        res = evaluate_nli_test_suite(precomputed, self.sample_claims)
        self.assertIn("CustomModel", res)
        self.assertEqual(res["CustomModel"]["accuracy"], 85.0)

    def test_compute_class_weights_balanced(self):
        pairs = (
            [{"label": "SUPPORTED"}] * 10
            + [{"label": "REFUTES"}] * 10
            + [{"label": "NOT_ENOUGH_INFO"}] * 10
        )
        weights = compute_class_weights(pairs)
        self.assertEqual(len(weights), 3)
        self.assertAlmostEqual(weights[0], 1.0, places=5)
        self.assertAlmostEqual(weights[1], 1.0, places=5)
        self.assertAlmostEqual(weights[2], 1.0, places=5)
        self.assertAlmostEqual(sum(weights), 3.0, places=5)

    def test_compute_class_weights_imbalanced(self):
        pairs = (
            [{"label": "SUPPORTED"}] * 10
            + [{"label": "REFUTES"}] * 20
            + [{"label": "NOT_ENOUGH_INFO"}] * 70
        )
        weights = compute_class_weights(pairs)
        self.assertEqual(len(weights), 3)
        self.assertAlmostEqual(sum(weights), 3.0, places=5)
        # Rarest class gets highest weight
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[1], weights[2])

        # Exact formula validation: w_c = N / (3 * N_c), normalized so sum is 3.0
        raw_0 = 100.0 / (3.0 * 10.0)
        raw_1 = 100.0 / (3.0 * 20.0)
        raw_2 = 100.0 / (3.0 * 70.0)
        sum_raw = raw_0 + raw_1 + raw_2
        expected_0 = (raw_0 / sum_raw) * 3.0
        expected_1 = (raw_1 / sum_raw) * 3.0
        expected_2 = (raw_2 / sum_raw) * 3.0

        self.assertAlmostEqual(weights[0], expected_0, places=4)
        self.assertAlmostEqual(weights[1], expected_1, places=4)
        self.assertAlmostEqual(weights[2], expected_2, places=4)

    def test_compute_class_weights_defensive(self):
        self.assertEqual(compute_class_weights([]), [1.0, 1.0, 1.0])
        self.assertEqual(compute_class_weights(None), [1.0, 1.0, 1.0])

        # Missing class (0 occurrences of REFUTES)
        pairs_missing = (
            [{"label": "SUPPORTED"}] * 15
            + [{"label": "NOT_ENOUGH_INFO"}] * 15
        )
        weights = compute_class_weights(pairs_missing)
        self.assertEqual(len(weights), 3)
        self.assertAlmostEqual(sum(weights), 3.0, places=5)
        for w in weights:
            self.assertGreater(w, 0.0)

    def test_get_cosine_schedule_with_warmup_mock_mode(self):
        class MockOptimizer:
            def __init__(self):
                self.param_groups = [{"lr": 1e-4}]

        optimizer = MockOptimizer()
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=100)
        self.assertIsNotNone(scheduler)
        self.assertTrue(hasattr(scheduler, "step"))
        scheduler.step()
        self.assertEqual(getattr(scheduler, "step_count", 1), 1)

    def test_get_cosine_schedule_with_warmup_lambda_decay(self):
        import unittest.mock

        class MockOptimizer:
            def __init__(self):
                self.param_groups = [{"lr": 1e-4}]

        num_warmup = 10
        num_training = 100

        mock_lambda_lr = unittest.mock.MagicMock(side_effect=lambda opt, lr_fn, **kwargs: lr_fn)
        mock_torch = unittest.mock.MagicMock()
        mock_torch.optim.lr_scheduler.LambdaLR = mock_lambda_lr

        with unittest.mock.patch.dict("sys.modules", {
            "torch": mock_torch,
            "torch.optim": mock_torch.optim,
            "torch.optim.lr_scheduler": mock_torch.optim.lr_scheduler,
        }):
            with unittest.mock.patch("scripts.train_nli_verifier.HAS_TORCH", True):
                lr_fn = get_cosine_schedule_with_warmup(MockOptimizer(), num_warmup, num_training)
                # At step 0: lr multiplier should be 0.0
                self.assertAlmostEqual(lr_fn(0), 0.0, places=5)
                # At warmup midpoint: lr multiplier should be 0.5
                self.assertAlmostEqual(lr_fn(5), 0.5, places=5)
                # At warmup completion (step 10): lr multiplier should be 1.0
                self.assertAlmostEqual(lr_fn(10), 1.0, places=5)
                # At midway through cosine decay (step 55): progress = 45/90 = 0.5, cos(pi/2)=0 => 0.5
                self.assertAlmostEqual(lr_fn(55), 0.5, places=5)
                # At final training step (step 100): progress = 90/90 = 1.0, cos(pi)=-1 => 0.0
                self.assertAlmostEqual(lr_fn(100), 0.0, places=5)
                # Beyond training step: clamped to >= 0.0
                self.assertAlmostEqual(lr_fn(120), 0.0, places=5)

    def test_differential_parameter_groups(self):
        class MockParam:
            def __init__(self, requires_grad=True):
                self.requires_grad = requires_grad

        class MockMorphoModel:
            def named_parameters(self):
                return [
                    ("encoder.embeddings.word_embeddings.weight", MockParam(True)),
                    ("encoder.layer.0.attention.weight", MockParam(True)),
                    ("classifier.weight", MockParam(True)),
                    ("classifier.bias", MockParam(True)),
                    ("encoder.frozen.weight", MockParam(False)),
                ]

            def parameters(self):
                return [p for _, p in self.named_parameters()]

        model = MockMorphoModel()
        param_groups = get_parameter_groups(model, lr_encoder=2e-5, lr_head=2e-4, is_morpho=True)

        self.assertEqual(len(param_groups), 2)
        # Group 0: encoder params
        self.assertEqual(param_groups[0]["lr"], 2e-5)
        self.assertEqual(len(param_groups[0]["params"]), 2)
        # Group 1: head/classifier params
        self.assertEqual(param_groups[1]["lr"], 2e-4)
        self.assertEqual(len(param_groups[1]["params"]), 2)

    def test_train_gpu_epoch_interface(self):
        loss = train_gpu_epoch(
            model=None,
            dataloader=None,
            optimizer=None,
            scaler=None,
            device="cpu",
            use_morpho=True,
            scheduler=None,
            class_weights=[1.0, 1.0, 1.0],
        )
        self.assertIsInstance(loss, float)
        self.assertEqual(loss, 0.0)

    def test_evaluate_dev_epoch_defensive(self):
        metrics = evaluate_dev_epoch(
            model=None,
            dataloader=None,
            criterion=None,
            device="cpu",
            is_morpho=False,
        )
        self.assertIsInstance(metrics, dict)
        self.assertIn("loss", metrics)
        self.assertIn("accuracy", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertEqual(metrics["loss"], 0.0)
        self.assertEqual(metrics["accuracy"], 0.0)
        self.assertEqual(metrics["macro_f1"], 0.0)

    def test_evaluate_dev_epoch_computation(self):
        class MockTensorLogits:
            def __init__(self, data):
                self.data = data

            def argmax(self, dim=-1):
                return [int(row.index(max(row))) for row in self.data]

        class MockDevModel:
            def __init__(self):
                self.eval_called = False

            def eval(self):
                self.eval_called = True
                return self

            def __call__(self, input_ids=None, attention_mask=None, morpho_features=None):
                val = input_ids[0][0] if isinstance(input_ids[0], (list, tuple)) else input_ids[0, 0].item()
                if val == 1:
                    return MockTensorLogits([
                        [2.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [0.0, 0.0, 2.0],
                    ])
                else:
                    return MockTensorLogits([
                        [2.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0],  # misclassified label 1 as class 0
                        [0.0, 0.0, 2.0],
                    ])

        batch1 = {
            "input_ids": [[1, 0], [1, 1], [1, 2]],
            "attention_mask": [[1, 1], [1, 1], [1, 1]],
            "label": [0, 1, 2],
        }
        batch2 = {
            "input_ids": [[2, 0], [2, 1], [2, 2]],
            "attention_mask": [[1, 1], [1, 1], [1, 1]],
            "label": [0, 1, 2],
        }

        model = MockDevModel()
        dataloader = [batch1, batch2]

        class MockLoss:
            def item(self):
                return 0.35

        criterion = lambda logits, labels: MockLoss()

        metrics = evaluate_dev_epoch(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            device="cpu",
            is_morpho=False,
        )

        self.assertTrue(model.eval_called)
        self.assertIn("loss", metrics)
        self.assertIn("accuracy", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertGreater(metrics["loss"], 0.0)
        # 5 out of 6 correct = 83.33%
        self.assertAlmostEqual(metrics["accuracy"], 83.3, delta=0.5)
        # Macro-F1 should be ~82.2%
        self.assertAlmostEqual(metrics["macro_f1"], 82.2, delta=0.5)

    def test_evaluate_dev_epoch_morpho_passthrough(self):
        class MockMorphoModel:
            def __init__(self):
                self.received_morpho = False

            def __call__(self, input_ids=None, attention_mask=None, morpho_features=None):
                if morpho_features is not None:
                    self.received_morpho = True
                return [[1.0, 0.0, 0.0]]

        batch = {
            "input_ids": [[1, 2]],
            "attention_mask": [[1, 1]],
            "label": [0],
            "morpho_features": [0.0] * 16,
        }
        model = MockMorphoModel()
        metrics = evaluate_dev_epoch(
            model=model,
            dataloader=[batch],
            criterion=lambda l, y: 0.1,
            device="cpu",
            is_morpho=True,
        )
    def test_morphological_affix_extractor_feature_masking(self):
        from models.morpho_nli_verifier import MorphologicalAffixExtractor

        extractor = MorphologicalAffixExtractor()
        claim = "Қазақстан 1991 жылы тәуелсіздік алмаған еді."
        evidence = "Қазақстан 1993 жылы тәуелсіздік алды, осылай болыпты."

        raw = extractor.extract_features(claim, evidence)
        self.assertEqual(len(raw), 16)
        self.assertEqual(raw[2], 1.0)  # directional negation mismatch
        self.assertEqual(raw[3], 1.0)  # calendar year conflict (1991 vs 1993)
        self.assertEqual(raw[6], 1.0)  # evidential marker in evidence (болыпты)

        # 1. Mask via mask_indices
        m_neg_idx = extractor.extract_features(claim, evidence, mask_indices=[2])
        self.assertEqual(m_neg_idx[2], 0.0)
        self.assertEqual(m_neg_idx[3], 1.0)

        m_multi_idx = extractor.extract_features(claim, evidence, mask_indices=[2, 3])
        self.assertEqual(m_multi_idx[2], 0.0)
        self.assertEqual(m_multi_idx[3], 0.0)

        # 2. Mask via ablation_mode: Negation Alignment (mask [2])
        m_neg = extractor.extract_features(claim, evidence, ablation_mode="- Negation Alignment")
        self.assertEqual(m_neg[2], 0.0)
        self.assertEqual(m_neg[3], 1.0)

        # 3. Mask via ablation_mode: Temporal / Calendar Conflicts (mask [3])
        m_temp = extractor.extract_features(claim, evidence, ablation_mode="- Temporal / Calendar Conflicts")
        self.assertEqual(m_temp[3], 0.0)
        self.assertEqual(m_temp[2], 1.0)

        # 4. Mask via ablation_mode: Evidentials & Epistemic Modals (mask [4, 5, 6, 7])
        m_evid = extractor.extract_features(claim, evidence, ablation_mode="- Evidentials & Epistemic Modals")
        for idx in [4, 5, 6, 7]:
            self.assertEqual(m_evid[idx], 0.0)

        # 5. Mask via ablation_mode: FST Root Analysis (mask [8, 12, 13])
        m_fst = extractor.extract_features(claim, evidence, ablation_mode="- FST Root Analysis")
        for idx in [8, 12, 13]:
            self.assertEqual(m_fst[idx], 0.0)

        # 6. Mask via ablation_mode: Surface Token Overlap Only (keeps [9, 10])
        m_surf = extractor.extract_features(claim, evidence, ablation_mode="Surface Token Overlap Only")
        self.assertGreater(m_surf[10], 0.0)
        for idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13]:
            self.assertEqual(m_surf[idx], 0.0)

        # Extractor initialized with default mask_indices
        ext_masked = MorphologicalAffixExtractor(mask_indices=[2, 3])
        m_init = ext_masked.extract_features(claim, evidence)
        self.assertEqual(m_init[2], 0.0)
        self.assertEqual(m_init[3], 0.0)

    def test_format_ablation_table_latex(self):
        results = {
            "Ours (Full Morpho Cross-Encoder)": {
                "accuracy": 82.6,
                "macro_f1": 82.1,
                "fever_score": 71.4,
                "hard_nei_f1": 72.8,
            },
            "- Negation Alignment": {
                "accuracy": 79.5,
                "macro_f1": 78.7,
                "fever_score": 68.2,
                "hard_nei_f1": 68.3,
            },
            "- Temporal / Calendar Conflicts": {
                "accuracy": 80.2,
                "macro_f1": 79.5,
                "fever_score": 68.6,
                "hard_nei_f1": 69.1,
            },
            "- Evidentials & Epistemic Modals": {
                "accuracy": 80.8,
                "macro_f1": 80.2,
                "fever_score": 69.5,
                "hard_nei_f1": 69.8,
            },
            "- FST Root Analysis": {
                "accuracy": 75.3,
                "macro_f1": 74.6,
                "fever_score": 63.2,
                "hard_nei_f1": 64.2,
            },
            "Surface Token Overlap Only": {
                "accuracy": 72.1,
                "macro_f1": 71.5,
                "fever_score": 59.8,
                "hard_nei_f1": 63.7,
            },
        }

        latex = format_ablation_table_latex(results)
        self.assertIn(r"\begin{table}", latex)
        self.assertIn(r"\label{tab:ablation_results}", latex)
        self.assertIn(r"\toprule", latex)
        self.assertIn(r"\midrule", latex)
        self.assertIn(r"\bottomrule", latex)
        self.assertIn(r"\textbf{Ours (Full Morpho Cross-Encoder)}", latex)
        self.assertIn(r"\textbf{82.6}", latex)
        self.assertIn(r"\textbf{82.1}", latex)
        self.assertIn(r"\textbf{71.4}", latex)
        self.assertIn(r"\textbf{72.8}", latex)
        self.assertIn(r"- Negation Alignment", latex)
        self.assertIn(r"- Temporal / Calendar Conflicts", latex)
        self.assertIn(r"- Evidentials \& Epistemic Modals", latex)
        self.assertIn(r"- FST Root Analysis", latex)
        self.assertIn(r"Surface Token Overlap Only", latex)
        self.assertNotIn("TO" + "DO", latex)
        self.assertNotIn("TB" + "D", latex)

    def test_run_morphological_ablation_study_configurations(self):
        train_pairs = prepare_nli_pairs(self.sample_claims, self.sample_corpus)
        test_pairs = prepare_nli_pairs(self.sample_claims, self.sample_corpus)

        model = train_nli_classifier(
            train_pairs=train_pairs,
            dev_pairs=None,
            model_type="ours_morpho",
            dry_run=True,
        )

        ablation_results = run_morphological_ablation_study(
            model_or_pairs=model,
            test_pairs=test_pairs,
        )

        expected_configs = [
            "Ours (Full Morpho Cross-Encoder)",
            "- Negation Alignment",
            "- Temporal / Calendar Conflicts",
            "- Evidentials & Epistemic Modals",
            "- FST Root Analysis",
            "Surface Token Overlap Only",
        ]
        for cfg in expected_configs:
            self.assertIn(cfg, ablation_results, f"Missing config: {cfg}")
            metrics = ablation_results[cfg]
            for m_key in ["accuracy", "macro_f1", "fever_score", "hard_nei_f1"]:
                self.assertIn(m_key, metrics)
                self.assertIsInstance(metrics[m_key], (int, float))

    def test_run_morphological_ablation_study_file_exports(self):
        train_pairs = prepare_nli_pairs(self.sample_claims, self.sample_corpus)
        test_pairs = prepare_nli_pairs(self.sample_claims, self.sample_corpus)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_tex = os.path.join(tmpdir, "table_ablation_results.tex")
            out_json = os.path.join(tmpdir, "ablation_results.json")

            res = run_morphological_ablation_study(
                model_or_pairs=train_pairs,
                dev_pairs=train_pairs,
                test_pairs=test_pairs,
                output_latex=out_tex,
                output_json=out_json,
            )

            self.assertTrue(os.path.exists(out_tex))
            self.assertTrue(os.path.exists(out_json))

            with open(out_tex, "r", encoding="utf-8") as f:
                tex_data = f.read()
            self.assertIn(r"\label{tab:ablation_results}", tex_data)
            self.assertIn("Ours (Full Morpho Cross-Encoder)", tex_data)

            with open(out_json, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            self.assertIn("Ours (Full Morpho Cross-Encoder)", json_data)
            self.assertIn("- Negation Alignment", json_data)

    def test_cli_execution_with_ablation_exports(self):
        import subprocess

        with tempfile.TemporaryDirectory() as tmpdir:
            out_latex = os.path.join(tmpdir, "table2.tex")
            out_json = os.path.join(tmpdir, "results.json")
            out_abl_tex = os.path.join(tmpdir, "table_ablation_results.tex")
            out_abl_json = os.path.join(tmpdir, "ablation_results.json")

            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "train_nli_verifier.py"),
                "--train_data",
                str(PROJECT_ROOT / "data" / "kazakh_fever_train.jsonl"),
                "--dev_data",
                str(PROJECT_ROOT / "data" / "kazakh_fever_dev.jsonl"),
                "--test_data",
                str(PROJECT_ROOT / "data" / "kazakh_fever_test.jsonl"),
                "--corpus",
                str(PROJECT_ROOT / "data" / "kazakh_knowledge_corpus.jsonl"),
                "--output_latex",
                out_latex,
                "--output_json",
                out_json,
                "--output_ablation_latex",
                out_abl_tex,
                "--output_ablation_json",
                out_abl_json,
                "--dry_run",
            ]
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                cwd=str(PROJECT_ROOT),
            )
            self.assertEqual(
                res.returncode,
                0,
                f"CLI execution failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}",
            )
            self.assertTrue(os.path.exists(out_abl_tex))
            self.assertTrue(os.path.exists(out_abl_json))

            with open(out_abl_tex, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn(r"\label{tab:ablation_results}", content)
            self.assertIn("Ours (Full Morpho Cross-Encoder)", content)


if __name__ == "__main__":
    unittest.main()

