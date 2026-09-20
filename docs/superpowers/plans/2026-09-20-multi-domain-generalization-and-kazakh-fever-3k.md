# Multi-Domain Generalization, Kazakh-FEVER 3K & Publication Tooling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the data generation tooling, hybrid retrieval engine, social media benchmark pipeline, and advisor briefing guide to execute the two-paper publication roadmap approved by Professor Guo.

**Architecture:** A four-task modular architecture: (1) Kazakh-FEVER 3K crawler and Hard-NEI generation pipeline; (2) Social Media cross-domain dataset builder with PII anonymization; (3) Hybrid Evidence Retrieval engine combining 83-rule FST stemmed BM25 with dense embeddings via Reciprocal Rank Fusion (RRF); (4) Comprehensive Publication Strategy and Advisor Briefing Guide.

**Tech Stack:** Python 3.12, rank-bm25, sentence-transformers, unittest, Markdown / LaTeX.

## Global Constraints

- Under no circumstances shall `aist2026/paper.tex` be touched or modified (0 diff lines).
- Zero decorative emojis across all code, datasets, documentation, and comments.
- All Python file operations must explicitly specify `encoding='utf-8'`.
- All code files must contain complete, production-ready code with zero placeholders ("TODO", "TBD").
- Maintain strict backward compatibility with existing test suites (327 passing tests).

---

### Task 1: Kazakh-FEVER 3K Data Generation & Validation Pipeline

**Files:**
- Create: `scripts/build_kazakh_fever_3k.py`
- Test: `tests/test_build_kazakh_fever_3k.py`

**Interfaces:**
- Consumes: `data/kazakh_knowledge_corpus.jsonl`, `verification/fever_generator.py`.
- Produces: `scripts/build_kazakh_fever_3k.py` exporting:
  - `generate_candidate_claims_prompt(article_title: str, text: str, domain: str) -> str`: Generates strict span-grounded few-shot prompts for Qwen-2.5/GPT-4o enforcing Hard NEI generation.
  - `validate_claim_record(record: dict) -> bool`: Validates schema, word count ($8 \le \text{tokens} \le 35$), label legality (`SUPPORTS`, `REFUTES`, `NOT_ENOUGH_INFO`), and non-empty evidence.
  - `partition_dataset(records: list[dict]) -> tuple[list[dict], list[dict], list[dict]]`: Splits claims into 2,000 Train, 500 Dev, 500 Test (stratified by domain: Encyclopedic vs. Factcheck.kz).

- [ ] **Step 1: Write failing tests for Kazakh-FEVER 3K generation pipeline**

```python
# tests/test_build_kazakh_fever_3k.py
import unittest
from scripts.build_kazakh_fever_3k import (
    generate_candidate_claims_prompt,
    validate_claim_record,
    partition_dataset
)

class TestBuildKazakhFever3K(unittest.TestCase):
    def test_prompt_enforces_hard_nei_and_span_grounding(self):
        prompt = generate_candidate_claims_prompt("Абай Құнанбайұлы", "Абай 1845 жылы туған.", "wikipedia")
        self.assertIn("Абай Құнанбайұлы", prompt)
        self.assertIn("SUPPORTS", prompt)
        self.assertIn("REFUTES", prompt)
        self.assertIn("NOT_ENOUGH_INFO", prompt)
        self.assertIn("Hard NEI", prompt)
        self.assertIn("evidence_sentence", prompt)

    def test_validate_claim_record_valid(self):
        rec = {
            "id": "kz_fever_0001",
            "claim": "Абай Құнанбайұлы 1845 жылы Шығыс Қазақстанда дүниеге келген.",
            "evidence_sentences": ["Абай 1845 жылы туған."],
            "label": "SUPPORTS",
            "domain": "wikipedia"
        }
        self.assertTrue(validate_claim_record(rec))

    def test_validate_claim_record_invalid_label(self):
        rec = {
            "id": "kz_fever_0002",
            "claim": "Қысқа сөйлем.",
            "evidence_sentences": [],
            "label": "INVALID_LABEL",
            "domain": "wikipedia"
        }
        self.assertFalse(validate_claim_record(rec))

    def test_partition_dataset_distribution(self):
        dummy_data = [
            {"id": f"rec_{i}", "domain": "wikipedia" if i % 2 == 0 else "factcheck_kz"}
            for i in range(300)
        ]
        train, dev, test = partition_dataset(dummy_data)
        self.assertEqual(len(train) + len(dev) + len(test), 300)
        self.assertGreater(len(train), len(dev))
        self.assertGreater(len(dev), 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe -m unittest tests/test_build_kazakh_fever_3k.py`  
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.build_kazakh_fever_3k'`

- [ ] **Step 3: Implement `scripts/build_kazakh_fever_3k.py`**

Implement complete generation script with CLI arguments (`--input`, `--output`, `--generate-prompts`, `--dry-run`), schema validation, Hard NEI prompt construction, and stratified splitting.

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe -m unittest tests/test_build_kazakh_fever_3k.py`  
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add scripts/build_kazakh_fever_3k.py tests/test_build_kazakh_fever_3k.py
git commit -m "feat(dataset): implement Kazakh-FEVER 3K generation and validation pipeline"
```

---

### Task 2: Social Media Cross-Domain Benchmark Tooling

**Files:**
- Create: `scripts/build_social_media_benchmark.py`
- Test: `tests/test_build_social_media_benchmark.py`

**Interfaces:**
- Produces: `scripts/build_social_media_benchmark.py` exporting:
  - `anonymize_social_text(text: str) -> str`: Strips user handles (`@username`), URLs (`https://...`), and telephone numbers, replacing with standardized tokens (`@user_anon`, `[URL]`).
  - `generate_social_media_prompt(genre: str, persona: str) -> str`: Prompts LLMs for colloquial Kazakh text with realistic slang, conversational suffixes (`-сың ғой`, `-ма екен`), and code-switching.
  - `validate_social_record(record: dict) -> bool`: Verifies length ($10 \le \text{tokens} \le 120$), source labeling, and PII absence.

- [ ] **Step 1: Write failing tests for social media pipeline**

```python
# tests/test_build_social_media_benchmark.py
import unittest
from scripts.build_social_media_benchmark import (
    anonymize_social_text,
    generate_social_media_prompt,
    validate_social_record
)

class TestBuildSocialMediaBenchmark(unittest.TestCase):
    def test_anonymize_social_text(self):
        raw = "Сәлем @daulet_kz! Мына сайтты қараңыз: https://example.com/test тел: +77011234567"
        cleaned = anonymize_social_text(raw)
        self.assertNotIn("@daulet_kz", cleaned)
        self.assertNotIn("https://example.com/test", cleaned)
        self.assertIn("@user_anon", cleaned)
        self.assertIn("[URL]", cleaned)

    def test_generate_social_media_prompt(self):
        prompt = generate_social_media_prompt("telegram", "student")
        self.assertIn("Kazakh", prompt)
        self.assertIn("slang", prompt.lower())
        self.assertIn("code-switching", prompt.lower())

    def test_validate_social_record(self):
        valid_rec = {
            "id": "soc_001",
            "text": "Бұл жаңа телефон шынымен жақсы ма екен, кім қолданып көрді?",
            "label": "human",
            "platform": "telegram"
        }
        self.assertTrue(validate_social_record(valid_rec))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe -m unittest tests/test_build_social_media_benchmark.py`  
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.build_social_media_benchmark'`

- [ ] **Step 3: Implement `scripts/build_social_media_benchmark.py`**

Implement complete anonymization regex rules, prompt generators, and JSONL benchmark exporter.

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe -m unittest tests/test_build_social_media_benchmark.py`  
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add scripts/build_social_media_benchmark.py tests/test_build_social_media_benchmark.py
git commit -m "feat(dataset): implement social media cross-domain benchmark tooling"
```

---

### Task 3: Hybrid Evidence Retrieval Architecture (BM25 + Dense Embeddings + RRF)

**Files:**
- Create: `src/retrieval/hybrid_retriever.py`
- Test: `tests/test_hybrid_retriever.py`

**Interfaces:**
- Consumes: `src/morphology/fst_analyzer.py` (83-rule FST stemmer).
- Produces: `HybridEvidenceRetriever` class with methods:
  - `index_corpus(documents: list[dict])`: Indexes documents with morphological BM25 and dense embeddings.
  - `retrieve(query: str, top_k: int = 3) -> list[dict]`: Computes Reciprocal Rank Fusion (RRF) over sparse and dense candidates, returning top-$K$ passages with fused scores.
  - `compute_rrf_score(sparse_rank: int, dense_rank: int, k_const: int = 60) -> float`:
    $$\text{score} = \frac{1}{k_{const} + \text{sparse\_rank}} + \frac{1}{k_{const} + \text{dense\_rank}}$$

- [ ] **Step 1: Write failing tests for hybrid retriever**

```python
# tests/test_hybrid_retriever.py
import unittest
from src.retrieval.hybrid_retriever import HybridEvidenceRetriever

class TestHybridEvidenceRetriever(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            {"id": "doc_1", "text": "Абай Құнанбайұлы 1845 жылы туған ұлы ақын."},
            {"id": "doc_2", "text": "Мұхтар Әуезов Абай жолы роман-эпопеясын жазды."},
            {"id": "doc_3", "text": "Астана қаласы Қазақстанның елордасы болып табылады."}
        ]
        self.retriever = HybridEvidenceRetriever()
        self.retriever.index_corpus(self.corpus)

    def test_retrieval_returns_relevant_top_k(self):
        query = "Абай Құнанбайұлы қай жылы туған?"
        results = self.retriever.retrieve(query, top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "doc_1")
        self.assertIn("rrf_score", results[0])

    def test_rrf_calculation(self):
        score = self.retriever.compute_rrf_score(sparse_rank=1, dense_rank=1, k_const=60)
        expected = (1.0 / 61.0) + (1.0 / 61.0)
        self.assertAlmostEqual(score, expected, places=5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe -m unittest tests/test_hybrid_retriever.py`  
Expected: FAIL with `ModuleNotFoundError: No module named 'src.retrieval'`

- [ ] **Step 3: Implement `src/retrieval/hybrid_retriever.py`**

Implement `HybridEvidenceRetriever` with FST morphological stemming, dense semantic vector scoring (with a fallback cosine similarity embedder for CPU test environments), and RRF ranking.

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe -m unittest tests/test_hybrid_retriever.py`  
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add src/retrieval/ tests/test_hybrid_retriever.py
git commit -m "feat(retrieval): implement hybrid evidence retriever with FST BM25 and RRF"
```

---

### Task 4: Master Publication Strategy & Advisor Briefing Guide

**Files:**
- Create: `docs/publication_strategy_and_advisor_briefing.md`
- Create: `C:\Users\Roza\.gemini\antigravity\brain\29832fd0-dcc9-4b3c-9300-06719048089c\publication_strategy_and_advisor_briefing.md`
- Test: `tests/test_publication_strategy.py`

**Interfaces:**
- Produces: Comprehensive markdown briefing document for Daulet, Wangna, and Prof. Guo containing:
  - Venue analysis: EMNLP, COLING, LREC-COLING, EACL, and ACM TALLIP.
  - ARR submission calendar with milestone deadlines.
  - Anti-dual-submission compliance rules and journal 30%+ novelty breakdown.
  - Pre-drafted LaTeX outline structure for Paper 1 (Kazakh-FEVER 3K).

- [ ] **Step 1: Write test for briefing guide completeness**

```python
# tests/test_publication_strategy.py
import os
import unittest

BRIEF_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs",
    "publication_strategy_and_advisor_briefing.md"
)

class TestPublicationStrategy(unittest.TestCase):
    def test_briefing_file_exists_and_covers_venues(self):
        self.assertTrue(os.path.isfile(BRIEF_PATH))
        with open(BRIEF_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("EMNLP", content)
        self.assertIn("COLING", content)
        self.assertIn("ACM TALLIP", content)
        self.assertIn("ACL Rolling Review", content)
        self.assertIn("Kazakh-FEVER 3K", content)
        self.assertNotIn("TODO", content)
        self.assertNotIn("TBD", content)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe -m unittest tests/test_publication_strategy.py`  
Expected: FAIL with `AssertionError: False is not true`

- [ ] **Step 3: Write `docs/publication_strategy_and_advisor_briefing.md` and mirror to artifact**

Write the complete guide and mirror it to the brain artifact directory.

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe -m unittest tests/test_publication_strategy.py`  
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add -f docs/publication_strategy_and_advisor_briefing.md tests/test_publication_strategy.py
git commit -m "docs(publication): add master publication strategy and advisor briefing guide"
```
