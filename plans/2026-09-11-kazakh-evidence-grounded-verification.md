# Kazakh Evidence-Grounded Factual Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the complete Evidence-Grounded Factual Verification Engine (Topic 3 / Paper 2, 30% of MSc Thesis) for low-resource Kazakh, including reference Wikipedia knowledge storage, FST-stemmed hybrid retrieval, atomic claim extraction, NLI contradiction detection, dual-risk trust scoring, and the synthetic Kazakh-FEVER benchmark.

**Architecture:** A modular pipeline where input Kazakh documents are chunked and evaluated for AI generation risk ($\text{Risk}_{\text{AI}}$), decomposed into atomic verifiable claims, matched against a verified Kazakh Wikipedia knowledge corpus using an FST-stemmed hybrid retriever, checked for factual consistency using a 3-way NLI verifier, and fused into an overall trustworthiness risk score with 4-quadrant categorization.

**Tech Stack:** Python 3.12, PyTorch/Transformers (with defensive pure-Python fallbacks), Okapi BM25 with `AdvancedKazakhFSTAnalyzer` morphological stemming, JSONL persistence, Unittest.

## Global Constraints

- Under no circumstances shall `aist2026/paper.tex` be touched or modified.
- Full regression suite must maintain 100% passing rate (all existing 241 tests preserved).
- Mandatory zero decorative emojis across all logs, code, outputs, and documentation.
- Defensive operation in standard CPU and mock environments without GPU hardware dependencies.
- Clean UTF-8 encoding across all Kazakh Cyrillic strings.

---

### Task 1: Evidence Data Structures, Knowledge Store & FST-Stemmed Inverted Index

**Files:**
- Create: `verification/__init__.py`
- Create: `verification/evidence.py`
- Create: `verification/knowledge_store.py`
- Create: `data/kazakh_knowledge_corpus.jsonl`
- Test: `tests/test_knowledge_store.py`

**Interfaces:**
- Consumes: `AdvancedKazakhFSTAnalyzer` from `fst_analyzer.py`
- Produces: `EvidencePassage`, `AtomicClaim`, `ClaimVerificationResult`, `DocumentTrustResult` in `verification/evidence.py`; `KnowledgeStore` with `.add_passage()`, `.get_passage()`, `.get_inverted_index()`, `.build_from_corpus()` in `verification/knowledge_store.py`.

- [ ] **Step 1: Write the failing tests in `tests/test_knowledge_store.py`**
- [ ] **Step 2: Run test to confirm failure**
  `C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe -m unittest tests/test_knowledge_store.py`
- [ ] **Step 3: Implement data structures in `verification/evidence.py`**
- [ ] **Step 4: Implement reference corpus seed in `data/kazakh_knowledge_corpus.jsonl` and `verification/knowledge_store.py`**
- [ ] **Step 5: Re-run tests to confirm pass**
- [ ] **Step 6: Git commit**
  `git add verification/ data/ tests/test_knowledge_store.py`
  `git commit -m "feat(verification): implement evidence data structures and FST-stemmed knowledge store"`

---

### Task 2: Hybrid Evidence Retriever (BM25 Sparse + Semantic Reranking)

**Files:**
- Create: `verification/retriever.py`
- Test: `tests/test_evidence_retriever.py`

**Interfaces:**
- Consumes: `KnowledgeStore`, `EvidencePassage` from Task 1; `AdvancedKazakhFSTAnalyzer` from `fst_analyzer.py`
- Produces: `HybridEvidenceRetriever(knowledge_store, top_k=5)` with `.retrieve(query: str, top_k: int = 5) -> List[EvidencePassage]`

- [ ] **Step 1: Write the failing tests in `tests/test_evidence_retriever.py`**
  Verify query stemming (*Алматыда* retrieves passage containing *Алматының*), BM25 score ranking, and graceful fallback when PyTorch embeddings are omitted.
- [ ] **Step 2: Run test to confirm failure**
- [ ] **Step 3: Implement `verification/retriever.py`**
- [ ] **Step 4: Re-run tests to confirm pass**
- [ ] **Step 5: Git commit**
  `git add verification/retriever.py tests/test_evidence_retriever.py`
  `git commit -m "feat(verification): implement FST-stemmed hybrid BM25 evidence retriever"`

---

### Task 3: Kazakh Atomic Claim Extractor

**Files:**
- Create: `verification/claim_extractor.py`
- Test: `tests/test_claim_extractor.py`

**Interfaces:**
- Consumes: `SentencePreservingChunker` from `kaz_mage/chunker.py`; `AdvancedKazakhFSTAnalyzer` from `fst_analyzer.py`
- Produces: `KazakhClaimExtractor` with `.extract_claims(text: str) -> List[AtomicClaim]`

- [ ] **Step 1: Write the failing tests in `tests/test_claim_extractor.py`**
  Verify hedge stripping (*Меніңше*), compound clause splitting (*және*, *әрі*, *бірақ*), and opinion filtering (*Өте керемет*).
- [ ] **Step 2: Run test to confirm failure**
- [ ] **Step 3: Implement `verification/claim_extractor.py`**
- [ ] **Step 4: Re-run tests to confirm pass**
- [ ] **Step 5: Git commit**
  `git add verification/claim_extractor.py tests/test_claim_extractor.py`
  `git commit -m "feat(verification): implement Kazakh atomic claim extraction engine"`

---

### Task 4: NLI Claim Verifier Engine (Contradiction, Negation & Entailment)

**Files:**
- Create: `verification/nli_verifier.py`
- Test: `tests/test_nli_verifier.py`

**Interfaces:**
- Consumes: `AtomicClaim`, `EvidencePassage`, `ClaimVerificationResult` from `verification/evidence.py`
- Produces: `NLIClaimVerifier` with `.verify_claim(claim: AtomicClaim, evidence_passages: List[EvidencePassage]) -> ClaimVerificationResult`

- [ ] **Step 1: Write the failing tests in `tests/test_nli_verifier.py`**
  Test 3 verdicts: `SUPPORTED` (exact year/entity alignment), `REFUTED` (numerical contradiction, negative suffix conflict), `NOT ENOUGH INFO` (unmentioned facts).
- [ ] **Step 2: Run test to confirm failure**
- [ ] **Step 3: Implement `verification/nli_verifier.py`**
- [ ] **Step 4: Re-run tests to confirm pass**
- [ ] **Step 5: Git commit**
  `git add verification/nli_verifier.py tests/test_nli_verifier.py`
  `git commit -m "feat(verification): implement 3-way NLI claim verification engine"`

---

### Task 5: Dual-Risk Trust Scorer & Four-Quadrant Categorization

**Files:**
- Create: `verification/trust_scorer.py`
- Test: `tests/test_trust_scorer.py`

**Interfaces:**
- Consumes: `ClaimVerificationResult`, `DocumentTrustResult` from `verification/evidence.py`; `DocumentAnalysisResult` from `kaz_mage/document.py`
- Produces: `DualRiskTrustScorer(alpha=0.5)` with `.score_document(doc_text: str, ai_result: Optional[DocumentAnalysisResult] = None) -> DocumentTrustResult`

- [ ] **Step 1: Write the failing tests in `tests/test_trust_scorer.py`**
  Verify factual penalty aggregation, mathematical risk formula $\alpha \text{Risk}_{\text{AI}} + (1 - \alpha) \text{Risk}_{\text{Fact}}$, and four-quadrant assignments: `Verified Human Fact`, `Human Misinformation`, `Accurate AI Synthesis`, `Hallucinatory AI Disinformation`.
- [ ] **Step 2: Run test to confirm failure**
- [ ] **Step 3: Implement `verification/trust_scorer.py`**
- [ ] **Step 4: Re-run tests to confirm pass**
- [ ] **Step 5: Git commit**
  `git add verification/trust_scorer.py tests/test_trust_scorer.py`
  `git commit -m "feat(verification): implement dual-risk trust scorer and four-quadrant matrix"`

---

### Task 6: Kazakh-FEVER Benchmark Generator, Evaluator & Full System Regression

**Files:**
- Create: `verification/fever_generator.py`
- Create: `verification/evaluator.py`
- Create: `data/kazakh_fever_benchmark.jsonl`
- Test: `tests/test_kazakh_fever.py`

**Interfaces:**
- Consumes: All verification subsystems from Tasks 1-5
- Produces: `KazakhFEVERGenerator.generate_benchmark()`, `FEVEREvaluator.evaluate_system()`

- [ ] **Step 1: Write failing tests in `tests/test_kazakh_fever.py`**
- [ ] **Step 2: Run test to confirm failure**
- [ ] **Step 3: Implement `verification/fever_generator.py` and `verification/evaluator.py`**
- [ ] **Step 4: Generate seed benchmark in `data/kazakh_fever_benchmark.jsonl`**
- [ ] **Step 5: Run full regression across all 241+ tests**
  `C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe -m unittest discover -s tests -p "test_*.py"`
- [ ] **Step 6: Git commit and push**
  `git add verification/ data/ tests/`
  `git commit -m "feat(verification): implement Kazakh-FEVER benchmark generator, evaluation harness, and test suite"`
  `git push origin main`
