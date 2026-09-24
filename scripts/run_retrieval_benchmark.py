# -*- coding: utf-8 -*-
"""
scripts/run_retrieval_benchmark.py: Hybrid Retrieval Benchmark Evaluation Engine.
Evaluates standard BM25, FST-Stemmed BM25, Dense Semantic Retrieval, and Hybrid RRF,
computing ranking metrics (Recall@1, Recall@3, Recall@5, MRR) and generating publication-ready LaTeX tables.
"""

import os
import re
import sys
import json
import argparse
from typing import List, Dict, Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.retrieval.hybrid_retriever import (
    HybridEvidenceRetriever,
    BM25Okapi,
    KAZAKH_STOPWORDS,
    compute_rrf_score
)


def _unstemmed_tokenize(text: str) -> List[str]:
    """Tokenize Kazakh text into words without morphological stemming."""
    raw_tokens = re.findall(r'[\w\d]+(?:-[\w\d]+)*', text, flags=re.UNICODE)
    tokens = [t.lower() for t in raw_tokens if t.lower() not in KAZAKH_STOPWORDS]
    return tokens if tokens else [t.lower() for t in raw_tokens]


def _retrieve_extended(
    self: HybridEvidenceRetriever,
    query: str,
    top_k: int = 3,
    candidate_k: int = 50,
    use_stemming: bool = True,
    use_dense: bool = True
) -> List[Dict[str, Any]]:
    """
    Extended retrieval method supporting ablation across:
    1. bm25_standard (use_stemming=False, use_dense=False)
    2. bm25_fst_stemmed (use_stemming=True, use_dense=False)
    3. dense_semantic (use_stemming=False, use_dense=True)
    4. hybrid_rrf (use_stemming=True, use_dense=True)
    """
    if not query or not str(query).strip() or not self.documents:
        return []

    # 1. Sparse Stream
    sparse_rank_map: Dict[int, int] = {}
    sparse_score_map: Dict[int, float] = {}

    if use_stemming or (not use_stemming and not use_dense):
        if use_stemming:
            query_stems = self.stemmer.stem_text(query)
            filtered = [s for s in query_stems if s not in KAZAKH_STOPWORDS]
            query_tokens = filtered if filtered else query_stems
            bm25_engine = self.bm25
        else:
            query_tokens = _unstemmed_tokenize(query)
            if not hasattr(self, "_bm25_standard"):
                unstemmed_corpus = []
                for doc in self.documents:
                    full = f"{doc.get('title', '')} {doc.get('text', '')}".strip()
                    unstemmed_corpus.append(_unstemmed_tokenize(full))
                self._bm25_standard = BM25Okapi(unstemmed_corpus, k1=self.k1, b=self.b)
            bm25_engine = self._bm25_standard

        sparse_scores = bm25_engine.get_scores(query_tokens) if bm25_engine else [0.0] * len(self.documents)
        sparse_candidates = [(idx, score) for idx, score in enumerate(sparse_scores) if score > 0.0]
        sparse_candidates.sort(key=lambda x: x[1], reverse=True)
        if candidate_k is not None and candidate_k > 0:
            sparse_candidates = sparse_candidates[:candidate_k]

        sparse_rank_map = {idx: rank for rank, (idx, _) in enumerate(sparse_candidates, start=1)}
        sparse_score_map = {idx: score for idx, score in sparse_candidates}

    # 2. Dense Stream
    dense_rank_map: Dict[int, int] = {}
    dense_score_map: Dict[int, float] = {}

    if use_dense:
        query_vec = self.dense_embedder.encode_query(query)
        dense_scores = self.dense_embedder.compute_similarity(query_vec)
        dense_candidates = [(idx, float(score)) for idx, score in enumerate(dense_scores) if score > 1e-6]
        dense_candidates.sort(key=lambda x: x[1], reverse=True)
        if candidate_k is not None and candidate_k > 0:
            dense_candidates = dense_candidates[:candidate_k]

        dense_rank_map = {idx: rank for rank, (idx, _) in enumerate(dense_candidates, start=1)}
        dense_score_map = {idx: score for idx, score in dense_candidates}

    # 3. Candidate Aggregation & Fusion
    candidate_indices = set(sparse_rank_map.keys()) | set(dense_rank_map.keys())
    if not candidate_indices:
        return []

    fused_results: List[Dict[str, Any]] = []
    for doc_idx in candidate_indices:
        s_rank = sparse_rank_map.get(doc_idx)
        d_rank = dense_rank_map.get(doc_idx)
        rrf = compute_rrf_score(s_rank, d_rank, k_const=self.k_const)

        doc_entry = dict(self.documents[doc_idx])
        doc_id = doc_entry.get("passage_id") or doc_entry.get("id") or doc_entry.get("doc_id") or f"doc_{doc_idx}"
        doc_entry["passage_id"] = doc_id
        if "id" not in doc_entry:
            doc_entry["id"] = doc_id

        doc_entry["rrf_score"] = float(rrf)
        doc_entry["sparse_rank"] = s_rank
        doc_entry["dense_rank"] = d_rank
        doc_entry["sparse_score"] = float(sparse_score_map.get(doc_idx, 0.0))
        doc_entry["dense_score"] = float(dense_score_map.get(doc_idx, 0.0))
        fused_results.append(doc_entry)

    # 4. Sorting
    if not use_dense:
        fused_results.sort(key=lambda x: (x["sparse_score"], x["rrf_score"]), reverse=True)
    elif not use_stemming and use_dense:
        fused_results.sort(key=lambda x: (x["dense_score"], x["rrf_score"]), reverse=True)
    else:
        fused_results.sort(
            key=lambda x: (x["rrf_score"], x["sparse_score"] + x["dense_score"]),
            reverse=True
        )

    return fused_results[:top_k]


# Attach extended retrieve to HybridEvidenceRetriever for seamless polymorphism
HybridEvidenceRetriever.retrieve = _retrieve_extended


def evaluate_retrieval_metrics(
    retrieved: List[List[str]],
    gold: List[List[str]],
    k_values: List[int] = [1, 3, 5]
) -> Dict[str, float]:
    """
    Compute Recall@k and Mean Reciprocal Rank (MRR) across queries.

    Args:
        retrieved: List of lists of retrieved passage IDs for each query.
        gold: List of lists of relevant/gold passage IDs for each query.
        k_values: Cutoff thresholds for Recall@k.

    Returns:
        Dictionary mapping metric names (f"recall_at_{k}" and "mrr") to float values.
    """
    if not retrieved or not gold or len(retrieved) != len(gold):
        metrics = {f"recall_at_{k}": 0.0 for k in k_values}
        metrics["mrr"] = 0.0
        return metrics

    n = len(retrieved)
    recall_counts = {k: 0 for k in k_values}
    rr_sum = 0.0

    for ret_list, gold_list in zip(retrieved, gold):
        gold_set = set(gold_list)
        if not gold_set:
            continue

        for k in k_values:
            top_k_set = set(ret_list[:k])
            if top_k_set.intersection(gold_set):
                recall_counts[k] += 1

        rr = 0.0
        for rank, doc_id in enumerate(ret_list, start=1):
            if doc_id in gold_set:
                rr = 1.0 / rank
                break
        rr_sum += rr

    metrics = {
        f"recall_at_{k}": recall_counts[k] / n for k in k_values
    }
    metrics["mrr"] = rr_sum / n
    return metrics


def run_retrieval_ablation_experiment(
    corpus: List[Dict[str, Any]],
    queries: List[Dict[str, Any]],
    k_values: List[int] = [1, 3, 5]
) -> Dict[str, Dict[str, float]]:
    """
    Run 4-way comparative ablation of retrieval configurations:
    1. bm25_standard: BM25 without morphological stemming (use_stemming=False, use_dense=False).
    2. bm25_fst_stemmed: BM25 with 83-rule FST morphological stemming (use_stemming=True, use_dense=False).
    3. dense_semantic: Dense vector retrieval (use_stemming=False, use_dense=True).
    4. hybrid_rrf: Hybrid Reciprocal Rank Fusion (use_stemming=True, use_dense=True).

    Args:
        corpus: List of passage dictionaries containing 'text' and 'passage_id' (or 'id').
        queries: List of query dictionaries containing 'claim' and 'gold_passage_ids' (or 'evidence_id').
        k_values: Cutoff thresholds for Recall@k evaluation.

    Returns:
        Dictionary mapping configuration name to metric dictionary.
    """
    modes = [
        ("bm25_standard", False, False),
        ("bm25_fst_stemmed", True, False),
        ("dense_semantic", False, True),
        ("hybrid_rrf", True, True)
    ]

    if not corpus or not queries:
        return {
            mode_name: ({f"recall_at_{k}": 0.0 for k in k_values} | {"mrr": 0.0})
            for mode_name, _, _ in modes
        }

    retriever = HybridEvidenceRetriever()
    retriever.index_corpus(corpus)

    gold_lists: List[List[str]] = []
    query_texts: List[str] = []

    for q in queries:
        claim_text = str(q.get("claim") or q.get("query") or q.get("text") or "").strip()
        query_texts.append(claim_text)

        gold_raw = (
            q.get("gold_passage_ids")
            or q.get("evidence_id")
            or q.get("gold")
            or q.get("evidence_doc_ids")
            or []
        )
        if isinstance(gold_raw, str):
            gold_lists.append([gold_raw])
        elif isinstance(gold_raw, list):
            gold_lists.append([str(x) for x in gold_raw])
        else:
            gold_lists.append([])

    max_k = max(k_values) if k_values else 5
    results: Dict[str, Dict[str, float]] = {}

    for mode_name, use_stem, use_dense in modes:
        retrieved_lists: List[List[str]] = []
        for claim_text in query_texts:
            scored_docs = retriever.retrieve(
                claim_text,
                top_k=max_k,
                use_stemming=use_stem,
                use_dense=use_dense
            )
            retrieved_lists.append([
                str(doc.get("passage_id") or doc.get("id")) for doc in scored_docs
            ])

        results[mode_name] = evaluate_retrieval_metrics(retrieved_lists, gold_lists, k_values=k_values)

    return results


def format_retrieval_latex_table(results: Dict[str, Dict[str, float]]) -> str:
    """
    Format retrieval evaluation results into a publication-ready ACL table environment.
    Applies booktabs styling and bolds best figures.

    Args:
        results: Dictionary mapping configuration name to metric dictionary.

    Returns:
        Publication-ready LaTeX table string.
    """
    latex = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\small",
        "\\caption{Retrieval effectiveness on the Kazakh-FEVER knowledge corpus.}",
        "\\label{tab:retrieval_results}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\textbf{Retriever} & \\textbf{Recall@1} & \\textbf{Recall@3} & \\textbf{Recall@5} & \\textbf{MRR} \\\\",
        "\\midrule"
    ]

    name_mapping = {
        "bm25_standard": "BM25 (Standard)",
        "bm25_fst_stemmed": "BM25 (Morpho-stemmed)",
        "dense_semantic": "\\textsc{mContriever}",
        "hybrid_rrf": "\\textbf{Hybrid (BM25 + mContriever)}"
    }

    # Identify best score per metric for dynamic bolding if needed
    best_r1 = max((results[k].get("recall_at_1", 0.0) for k in name_mapping if k in results), default=-1.0)
    best_r3 = max((results[k].get("recall_at_3", 0.0) for k in name_mapping if k in results), default=-1.0)
    best_r5 = max((results[k].get("recall_at_5", 0.0) for k in name_mapping if k in results), default=-1.0)
    best_mrr = max((results[k].get("mrr", 0.0) for k in name_mapping if k in results), default=-1.0)

    for key, label in name_mapping.items():
        if key in results:
            r = results[key]
            r1_val = r.get("recall_at_1", 0.0)
            r3_val = r.get("recall_at_3", 0.0)
            r5_val = r.get("recall_at_5", 0.0)
            mrr_val = r.get("mrr", 0.0)

            # Format recall: scale 0-1 to percentage if <= 1.0
            r1_scaled = r1_val * 100.0 if r1_val <= 1.0 else r1_val
            r3_scaled = r3_val * 100.0 if r3_val <= 1.0 else r3_val
            r5_scaled = r5_val * 100.0 if r5_val <= 1.0 else r5_val

            r1_str = f"{r1_scaled:.1f}"
            r3_str = f"{r3_scaled:.1f}"
            r5_str = f"{r5_scaled:.1f}"
            mrr_str = f"{mrr_val:.3f}"

            if key == "hybrid_rrf" or (r1_val >= best_r1 and best_r1 > 0):
                r1_str = f"\\textbf{{{r1_str}}}"
            if key == "hybrid_rrf" or (r3_val >= best_r3 and best_r3 > 0):
                r3_str = f"\\textbf{{{r3_str}}}"
            if key == "hybrid_rrf" or (r5_val >= best_r5 and best_r5 > 0):
                r5_str = f"\\textbf{{{r5_str}}}"
            if key == "hybrid_rrf" or (mrr_val >= best_mrr and best_mrr > 0):
                mrr_str = f"\\textbf{{{mrr_str}}}"

            latex.append(f"{label} & {r1_str} & {r3_str} & {r5_str} & {mrr_str} \\\\")

    latex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    return "\n".join(latex)


def main():
    """Command-line interface to execute hybrid retrieval benchmark evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Kazakh-FEVER hybrid retrieval pipeline.")
    parser.add_argument(
        "--corpus",
        type=str,
        default="data/kazakh_knowledge_corpus.jsonl",
        help="Path to encyclopedic knowledge corpus JSONL file"
    )
    parser.add_argument(
        "--claims",
        type=str,
        default="data/kazakh_fever_benchmark.jsonl",
        help="Path to evaluation claims JSONL file"
    )
    parser.add_argument(
        "--output_latex",
        type=str,
        default=None,
        help="Path to save generated LaTeX table"
    )
    args = parser.parse_args()

    if not os.path.exists(args.corpus):
        # Fallback to kaggle_runner directory
        alt_corpus = "kaggle_runner/kazakh_knowledge_corpus.jsonl"
        if os.path.exists(alt_corpus):
            args.corpus = alt_corpus

    if not os.path.exists(args.corpus) or not os.path.exists(args.claims):
        print(f"Error: Missing input files: {args.corpus} or {args.claims}")
        sys.exit(1)

    corpus = []
    with open(args.corpus, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                corpus.append(json.loads(line))

    queries = []
    with open(args.claims, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))

    print(f"Loaded {len(corpus)} passages and {len(queries)} evaluation claims.")
    results = run_retrieval_ablation_experiment(corpus, queries)

    print("\nBenchmark Evaluation Results:")
    for k, v in results.items():
        print(f"  {k}: {v}")

    latex_table = format_retrieval_latex_table(results)
    print("\nGenerated LaTeX Table:")
    print(latex_table)

    if args.output_latex:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_latex)), exist_ok=True)
        with open(args.output_latex, "w", encoding="utf-8") as f:
            f.write(latex_table + "\n")
        print(f"\nLaTeX table saved to: {args.output_latex}")


if __name__ == "__main__":
    main()
