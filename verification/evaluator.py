# -*- coding: utf-8 -*-
"""
verification/evaluator.py: Evaluation Suite for Kazakh-FEVER and Document Trust Verifier.
Computes Retrieval Recall@K, 3-class NLI Macro-F1, and Strict Joint FEVER Accuracy.
"""

from typing import List, Dict, Any, Tuple
from collections import defaultdict


class FEVEREvaluator:
    """
    Computes standard academic verification metrics matching the FEVER protocol.
    """

    LABELS = ["SUPPORTED", "REFUTED", "NOT ENOUGH INFO"]

    def compute_metrics(
        self,
        y_true: List[str],
        y_pred: List[str],
        retrieved_ids: List[List[str]],
        ground_truth_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Calculates precision, recall, macro-F1, retrieval recall@5, and joint FEVER score.
        """
        N = len(y_true)
        if N == 0:
            return {
                "total_samples": 0,
                "nli_accuracy": 0.0,
                "nli_macro_f1": 0.0,
                "retrieval_recall_at_5": 0.0,
                "fever_score": 0.0,
                "per_class": {}
            }

        # 1. Overall NLI Accuracy
        correct_nli = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
        nli_acc = correct_nli / N

        # 2. Per-class Precision, Recall, and F1
        per_class = {}
        f1_sum = 0.0

        for label in self.LABELS:
            tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp == label)
            fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != label and yp == label)
            fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp != label)

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            per_class[label] = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "support": sum(1 for yt in y_true if yt == label)
            }
            f1_sum += f1

        macro_f1 = f1_sum / len(self.LABELS)

        # 3. Retrieval Recall@5
        correct_retrieval = 0
        for gid, r_list in zip(ground_truth_ids, retrieved_ids):
            if gid in r_list[:5]:
                correct_retrieval += 1
        retrieval_recall_5 = correct_retrieval / N

        # 4. Strict FEVER Score
        # If ground-truth evidence is specified, it must be retrieved in top-5
        # If no ground-truth is specified (e.g. true ungrounded NEI), correct label is sufficient
        correct_fever = 0
        for yt, yp, gid, r_list in zip(y_true, y_pred, ground_truth_ids, retrieved_ids):
            if yt == yp:
                if gid:
                    if gid in r_list[:5]:
                        correct_fever += 1
                else:
                    correct_fever += 1

        fever_score = correct_fever / N

        return {
            "total_samples": N,
            "nli_accuracy": round(nli_acc, 4),
            "nli_macro_f1": round(macro_f1, 4),
            "retrieval_recall_at_5": round(retrieval_recall_5, 4),
            "fever_score": round(fever_score, 4),
            "per_class": per_class
        }

    def evaluate_verifier(
        self,
        verifier: Any,
        benchmark_records: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluates a TrustworthyDocumentVerifier on benchmark records.
        """
        y_true = []
        y_pred = []
        retrieved_ids = []
        ground_truth_ids = []

        for rec in benchmark_records:
            claim_text = rec["claim"]
            true_label = rec["label"]
            gt_id = rec.get("evidence_id", "")

            # Run verifier
            doc_res = verifier.verify(claim_text)
            if doc_res.claims:
                verdicts = [c.verdict for c in doc_res.claims]
                if "REFUTED" in verdicts:
                    pred_verdict = "REFUTED"
                elif all(v == "SUPPORTED" for v in verdicts):
                    pred_verdict = "SUPPORTED"
                else:
                    pred_verdict = "NOT ENOUGH INFO"

                # Union of retrieved passage IDs preserving rank order
                r_ids = []
                for c in doc_res.claims:
                    for p in c.evidence:
                        if p.passage_id not in r_ids:
                            r_ids.append(p.passage_id)
            else:
                pred_verdict = "NOT ENOUGH INFO"
                r_ids = []

            y_true.append(true_label)
            y_pred.append(pred_verdict)
            retrieved_ids.append(r_ids)
            ground_truth_ids.append(gt_id)

        return self.compute_metrics(y_true, y_pred, retrieved_ids, ground_truth_ids)
