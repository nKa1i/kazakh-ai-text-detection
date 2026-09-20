# -*- coding: utf-8 -*-
"""
src/retrieval/hybrid_retriever.py: Hybrid Evidence Retrieval Architecture
combining 83-rule FST morphological stemmed BM25 with dense semantic vector embeddings via RRF.
"""

import os
import re
import math
import json
import sys
from typing import List, Dict, Optional, Tuple, Any, Set, Union
from collections import defaultdict
import numpy as np

# Ensure root directory is on sys.path for fst_analyzer imports
try:
    from fst_analyzer import AdvancedKazakhFSTAnalyzer
except ImportError:
    try:
        from api.fst_analyzer import AdvancedKazakhFSTAnalyzer
    except ImportError:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)
        from fst_analyzer import AdvancedKazakhFSTAnalyzer

# Try importing rank_bm25; provide pure-Python BM25Okapi fallback if unavailable
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    class BM25Okapi:
        """Pure-Python implementation of Okapi BM25 ranking algorithm."""

        def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
            self.corpus_size = len(corpus)
            self.k1 = k1
            self.b = b
            self.epsilon = epsilon
            self.doc_lengths = [len(doc) for doc in corpus]
            self.avgdl = sum(self.doc_lengths) / self.corpus_size if self.corpus_size > 0 else 0.0
            self.doc_freqs: List[Dict[str, int]] = []
            self.nd: Dict[str, int] = defaultdict(int)

            for doc in corpus:
                freqs: Dict[str, int] = defaultdict(int)
                for term in doc:
                    freqs[term] += 1
                self.doc_freqs.append(freqs)
                for term in freqs:
                    self.nd[term] += 1

            self.idf: Dict[str, float] = {}
            for term, freq in self.nd.items():
                self.idf[term] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

        def get_scores(self, query: List[str]) -> List[float]:
            """Calculates BM25 scores for all corpus documents against query tokens."""
            if self.corpus_size == 0 or not query:
                return [0.0] * self.corpus_size

            scores = [0.0] * self.corpus_size
            for term in query:
                if term not in self.idf:
                    continue
                idf = self.idf[term]
                for i, freqs in enumerate(self.doc_freqs):
                    if term in freqs:
                        tf = freqs[term]
                        dl = self.doc_lengths[i]
                        denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / (self.avgdl or 1.0)))
                        scores[i] += idf * (tf * (self.k1 + 1.0)) / denom
            return scores

        def get_top_n(self, query: List[str], documents: List[Any], n: int = 5) -> List[Any]:
            """Returns top-n documents sorted by BM25 score."""
            scores = self.get_scores(query)
            scored = list(zip(documents, scores))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in scored[:n]]


KAZAKH_STOPWORDS: Set[str] = {
    "және", "мен", "бен", "пен", "де", "да", "та", "те", "бұл", "ол", "оның", "олар",
    "үшін", "туралы", "бойынша", "арқылы", "сияқты", "тура", "бойы", "дейін",
    "болып", "бола", "болған", "болды", "екен", "етіп", "еткен", "етті",
    "бір", "барлық", "бар", "жоқ", "ресми", "түрде", "деген", "деп", "осы", "сол"
}


def compute_rrf_score(
    sparse_rank: Optional[int],
    dense_rank: Optional[int],
    k_const: int = 60
) -> float:
    """
    Computes Reciprocal Rank Fusion (RRF) score for a document across sparse and dense ranks.

    Formula:
        RRF(d) = sum_{m in {sparse, dense}} ( I(d in results_m) / (k_const + rank_m(d)) )
    where ranks are 1-indexed.
    """
    score = 0.0
    if sparse_rank is not None and sparse_rank > 0:
        score += 1.0 / (k_const + sparse_rank)
    if dense_rank is not None and dense_rank > 0:
        score += 1.0 / (k_const + dense_rank)
    return score


class KazakhFSTStemmer:
    """
    Morphological stemmer for Kazakh language using the 83-rule FST analyzer.
    Extracts root morphemes by stripping inflectional and derivational suffixes.
    """

    def __init__(self, fst_analyzer: Optional[AdvancedKazakhFSTAnalyzer] = None):
        self.fst = fst_analyzer if fst_analyzer is not None else AdvancedKazakhFSTAnalyzer()

    def stem_word(self, word: str) -> List[str]:
        """
        Extracts morphological root morphemes and base lemmas for a Kazakh word.
        Returns [surface_form, root] to bridge agglutinative variations.
        """
        w = (word or "").strip().lower()
        w = re.sub(r'^[^\w\d]+|[^\w\d]+$', '', w, flags=re.UNICODE)
        if not w:
            return []
        if len(w) <= 3 or w.isdigit():
            return [w]

        stems = [w]

        # 1. Check FST segmentation
        seg = self.fst.analyze_and_segment(w)
        if " -" in seg:
            root = seg.split(" -")[0].strip()
            if root and len(root) >= 2 and root not in stems:
                stems.append(root)

        # 2. Direct case suffix base check
        match = self.fst.case_re.search(w)
        if match and len(w[:match.start()]) >= 3:
            case_base = w[:match.start()]
            if case_base not in stems:
                stems.append(case_base)

        return stems

    def stem_text(self, text: str) -> List[str]:
        """Tokenizes text and extracts morphological stems and roots."""
        if not text:
            return []
        raw_tokens = re.findall(r'[\w\d]+(?:-[\w\d]+)*', text, flags=re.UNICODE)
        tokens = []
        for t in raw_tokens:
            tokens.extend(self.stem_word(t))
        return tokens


class DenseEmbedder:
    """
    Dense semantic embedding provider.
    Supports SentenceTransformers (e.g. BGE-M3, multilingual-e5-large) with a fast,
    deterministic subword/character TF-IDF cosine similarity fallback for CPU environments.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        use_fallback: Optional[bool] = None
    ):
        self.model_name = model_name
        self.model = None
        self.vectorizer = None
        self.corpus_embeddings: Optional[np.ndarray] = None

        if use_fallback is False and model_name:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name)
                self.use_fallback = False
            except Exception:
                self.model = None
                self.use_fallback = True
        elif use_fallback is True:
            self.use_fallback = True
        else:
            if model_name:
                try:
                    from sentence_transformers import SentenceTransformer
                    self.model = SentenceTransformer(model_name)
                    self.use_fallback = False
                except Exception:
                    self.model = None
                    self.use_fallback = True
            else:
                self.use_fallback = True

    def fit_corpus(self, corpus_texts: List[str]) -> np.ndarray:
        """Computes and caches embeddings for corpus texts."""
        if not corpus_texts:
            self.corpus_embeddings = np.empty((0, 0))
            return self.corpus_embeddings

        if not self.use_fallback and self.model is not None:
            self.corpus_embeddings = self.model.encode(
                corpus_texts,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            return self.corpus_embeddings
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=1
            )
            self.corpus_embeddings = self.vectorizer.fit_transform(corpus_texts).toarray()
            return self.corpus_embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """Encodes query string into a dense vector."""
        if not self.use_fallback and self.model is not None:
            return self.model.encode(
                [query],
                convert_to_numpy=True,
                show_progress_bar=False
            )[0]
        else:
            if self.vectorizer is None or self.corpus_embeddings is None or len(self.corpus_embeddings) == 0:
                return np.array([])
            return self.vectorizer.transform([query]).toarray()[0]

    def compute_similarity(self, query_vec: np.ndarray) -> np.ndarray:
        """Computes cosine similarity between query vector and corpus embeddings."""
        if self.corpus_embeddings is None or len(self.corpus_embeddings) == 0 or len(query_vec) == 0:
            return np.array([])
        q_norm = np.linalg.norm(query_vec)
        if q_norm == 0.0:
            return np.zeros(len(self.corpus_embeddings))
        doc_norms = np.linalg.norm(self.corpus_embeddings, axis=1)
        denom = doc_norms * q_norm
        zero_mask = denom == 0.0
        denom[zero_mask] = 1.0
        sims = np.dot(self.corpus_embeddings, query_vec) / denom
        sims[zero_mask] = 0.0
        return sims


class HybridEvidenceRetriever:
    """
    Hybrid Evidence Retriever combining:
    1. Sparse Stream: 83-rule FST morphological stemmed BM25.
    2. Dense Stream: Semantic vector embeddings (SentenceTransformers / subword TF-IDF cosine).
    3. Fusion: Reciprocal Rank Fusion (RRF) for robust multi-domain evidence retrieval.
    """

    compute_rrf_score = staticmethod(compute_rrf_score)

    def __init__(
        self,
        fst_analyzer: Optional[AdvancedKazakhFSTAnalyzer] = None,
        dense_model_name: Optional[str] = None,
        use_dense_fallback: Optional[bool] = None,
        k1: float = 1.5,
        b: float = 0.75,
        k_const: int = 60
    ):
        self.k1 = k1
        self.b = b
        self.k_const = k_const
        self.stemmer = KazakhFSTStemmer(fst_analyzer)
        self.dense_embedder = DenseEmbedder(
            model_name=dense_model_name,
            use_fallback=use_dense_fallback
        )
        self.documents: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None

    def index_corpus(self, documents: List[Dict[str, Any]]) -> None:
        """
        Indexes a list of documents.
        Each document is expected to have at least 'id' (or 'doc_id', 'passage_id') and 'text'.
        """
        self.documents = []
        if not documents:
            self.bm25 = None
            self.dense_embedder.fit_corpus([])
            return

        corpus_tokens: List[List[str]] = []
        corpus_texts: List[str] = []

        for idx, doc in enumerate(documents):
            doc_copy = dict(doc)
            doc_id = doc_copy.get("id") or doc_copy.get("doc_id") or doc_copy.get("passage_id") or f"doc_{idx}"
            if "id" not in doc_copy:
                doc_copy["id"] = doc_id

            title = str(doc_copy.get("title", "")).strip()
            text = str(doc_copy.get("text", "")).strip()
            full_content = f"{title} {text}".strip() if title else text

            self.documents.append(doc_copy)
            corpus_texts.append(full_content)

            stems = self.stemmer.stem_text(full_content)
            corpus_tokens.append(stems)

        self.bm25 = BM25Okapi(corpus_tokens, k1=self.k1, b=self.b)
        self.dense_embedder.fit_corpus(corpus_texts)

    def index_jsonl(self, file_path: str) -> None:
        """Loads documents from a JSONL file and indexes them into the corpus."""
        if not os.path.exists(file_path):
            self.index_corpus([])
            return

        docs: List[Dict[str, Any]] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    docs.append(data)
                except json.JSONDecodeError:
                    continue
        self.index_corpus(docs)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        candidate_k: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-K documents for the given query using Reciprocal Rank Fusion
        over FST-BM25 sparse candidates and dense semantic candidates.

        Returns list of document dicts containing:
            - all original document fields
            - 'rrf_score': float
            - 'sparse_rank': Optional[int] (1-indexed)
            - 'dense_rank': Optional[int] (1-indexed)
            - 'sparse_score': float
            - 'dense_score': float
        """
        if not query or not query.strip() or not self.documents or self.bm25 is None:
            return []

        # 1. Sparse stream retrieval
        query_stems = self.stemmer.stem_text(query)
        filtered_stems = [s for s in query_stems if s not in KAZAKH_STOPWORDS]
        if not filtered_stems:
            filtered_stems = query_stems

        sparse_scores = self.bm25.get_scores(filtered_stems) if filtered_stems else [0.0] * len(self.documents)

        sparse_candidates = [
            (idx, score) for idx, score in enumerate(sparse_scores) if score > 0.0
        ]
        sparse_candidates.sort(key=lambda x: x[1], reverse=True)
        if candidate_k is not None and candidate_k > 0:
            sparse_candidates = sparse_candidates[:candidate_k]

        sparse_rank_map: Dict[int, int] = {
            doc_idx: rank for rank, (doc_idx, _) in enumerate(sparse_candidates, start=1)
        }
        sparse_score_map: Dict[int, float] = {
            doc_idx: score for doc_idx, score in sparse_candidates
        }

        # 2. Dense stream retrieval
        query_vec = self.dense_embedder.encode_query(query)
        dense_scores = self.dense_embedder.compute_similarity(query_vec)

        dense_candidates = [
            (idx, float(score)) for idx, score in enumerate(dense_scores) if score > 1e-6
        ]
        dense_candidates.sort(key=lambda x: x[1], reverse=True)
        if candidate_k is not None and candidate_k > 0:
            dense_candidates = dense_candidates[:candidate_k]

        dense_rank_map: Dict[int, int] = {
            doc_idx: rank for rank, (doc_idx, _) in enumerate(dense_candidates, start=1)
        }
        dense_score_map: Dict[int, float] = {
            doc_idx: score for doc_idx, score in dense_candidates
        }

        # 3. Reciprocal Rank Fusion
        candidate_indices = set(sparse_rank_map.keys()) | set(dense_rank_map.keys())
        if not candidate_indices:
            return []

        fused_results: List[Dict[str, Any]] = []
        for doc_idx in candidate_indices:
            s_rank = sparse_rank_map.get(doc_idx)
            d_rank = dense_rank_map.get(doc_idx)
            rrf = compute_rrf_score(s_rank, d_rank, k_const=self.k_const)

            doc_entry = dict(self.documents[doc_idx])
            doc_entry["rrf_score"] = float(rrf)
            doc_entry["sparse_rank"] = s_rank
            doc_entry["dense_rank"] = d_rank
            doc_entry["sparse_score"] = float(sparse_score_map.get(doc_idx, 0.0))
            doc_entry["dense_score"] = float(dense_score_map.get(doc_idx, 0.0))
            fused_results.append(doc_entry)

        # Sort descending by rrf_score, breaking ties by sum of raw scores
        fused_results.sort(
            key=lambda x: (x["rrf_score"], x["sparse_score"] + x["dense_score"]),
            reverse=True
        )

        return fused_results[:top_k]
