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


if __name__ == "__main__":
    unittest.main()

