# Kaz-DocumentEngine: Multi-Paragraph Sliding Window Chunking & Aggregation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a robust, production-grade document-level inference engine that segments arbitrary-length Kazakh texts ($500 \text{--} 10{,}000+$ words) into sentence-preserving sliding windows with character span tracking, evaluates them via batched dual-stream inference, and aggregates chunk scores using Top-$K$ worst-chunk pooling to reliably detect both full and partially machine-generated (hybrid) documents.

**Architecture:**
- `kaz_mage/document.py`: Structured dataclasses for `DocumentChunk` and `DocumentAnalysisResult`.
- `kaz_mage/chunker.py`: `SentencePreservingChunker` with Kazakh linguistic abbreviation protection (`т.б.`, `ж.б.`, `ғ.`, `ғғ.`, `ж.`, `жж.`, `қ.`, `мыс.`), quote preservation (`«...»`), and character offset tracking.
- `kaz_mage/aggregator.py`: `DocumentAggregator` computing Top-$K$ worst-chunk pooling, volume-weighted AI ratio, and three-tier classification (`"Authentic Human"`, `"Partially AI / Hybrid"`, `"Machine-Generated"`).
- `models/document_detector.py`: `DocumentDetector` facade orchestrating batched chunk inference through `MorphoContrastiveDetector` with micro-batching to prevent GPU OOM.

**Tech Stack:** Python 3.12, PyTorch, Transformers, `AdvancedKazakhFSTAnalyzer`, unittest.

## Global Constraints
- Under no circumstances shall `aist2026/paper.tex` be touched or modified.
- Full regression test suite must pass with 0 errors (all existing 102 tests preserved).
- Zero severed words or broken morpheme chains: text windowing must respect sentence and word boundaries.
- Defensive execution: all modules must operate cleanly in CPU/mock environments as well as GPU.
- Character offset fidelity: `document[chunk.start_char : chunk.end_char]` must match chunk content without index drift.

---

### Task 1: Document Data Structures & Representation

**Files:**
- Create: `kaz_mage/document.py`
- Modify: `kaz_mage/__init__.py`
- Test: `tests/test_document_chunker.py`

**Interfaces:**
- Produces:
  - `DocumentChunk(index, text, start_char, end_char, word_count, sentence_count, ai_probability, gate_value, is_ai)`
  - `DocumentAnalysisResult(verdict, document_ai_probability, ai_content_ratio, calibrated_threshold, total_words, total_sentences, total_chunks, worst_chunk, chunks)`
  - Methods: `to_dict()` on both dataclasses for clean JSON serialization.

- [x] **Step 1: Write the failing test**

```python
# tests/test_document_chunker.py
import unittest
from kaz_mage.document import DocumentChunk, DocumentAnalysisResult

class TestDocumentDataStructures(unittest.TestCase):
    def test_document_chunk_serialization(self):
        chunk = DocumentChunk(
            index=0,
            text="Бұл сынақ мәтіні.",
            start_char=0,
            end_char=17,
            word_count=3,
            sentence_count=1,
            ai_probability=0.05,
            gate_value=0.52,
            is_ai=False
        )
        d = chunk.to_dict()
        self.assertEqual(d["index"], 0)
        self.assertEqual(d["word_count"], 3)
        self.assertFalse(d["is_ai"])

    def test_document_analysis_result_serialization(self):
        chunk = DocumentChunk(0, "Мәтін", 0, 5, 1, 1, 0.999, 0.51, True)
        res = DocumentAnalysisResult(
            verdict="Machine-Generated",
            document_ai_probability=0.999,
            ai_content_ratio=1.0,
            calibrated_threshold=0.998,
            total_words=1,
            total_sentences=1,
            total_chunks=1,
            worst_chunk=chunk,
            chunks=[chunk]
        )
        d = res.to_dict()
        self.assertEqual(d["verdict"], "Machine-Generated")
        self.assertEqual(len(d["chunks"]), 1)
        self.assertEqual(d["worst_chunk"]["ai_probability"], 0.999)

if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_document_chunker.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'kaz_mage.document'`

- [x] **Step 3: Implement `kaz_mage/document.py` and export in `kaz_mage/__init__.py`**

- [x] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_document_chunker.py`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add kaz_mage/document.py kaz_mage/__init__.py tests/test_document_chunker.py
git commit -m "feat: implement DocumentChunk and DocumentAnalysisResult data structures"
```

---

### Task 2: Kazakh Sentence-Preserving Sliding Window Chunker

**Files:**
- Create: `kaz_mage/chunker.py`
- Modify: `kaz_mage/__init__.py`
- Test: `tests/test_document_chunker.py`

**Interfaces:**
- Consumes:
  - `DocumentChunk` from `kaz_mage.document`
- Produces:
  - `SentencePreservingChunker(max_words=200, overlap_sentences=1)`
  - `chunker.split_sentences(text: str) -> List[Tuple[str, int, int]]` (sentence text, start_char, end_char)
  - `chunker.chunk_document(text: str) -> List[DocumentChunk]`

- [x] **Step 1: Write the failing test**

```python
# Add to tests/test_document_chunker.py:
    def test_kazakh_abbreviations_protection(self):
        from kaz_mage.chunker import SentencePreservingChunker
        chunker = SentencePreservingChunker()
        text = "Бұл 2024 ж. болған оқиға. Онда т.б. мәселелер мен 15 ғғ. мұралары қаралды."
        sentences = chunker.split_sentences(text)
        # Should be exactly 2 sentences, NOT split at "ж.", "т.б.", or "ғғ."
        self.assertEqual(len(sentences), 2)
        self.assertIn("2024 ж.", sentences[0][0])
        self.assertIn("т.б.", sentences[1][0])

    def test_kazakh_quotes_protection(self):
        from kaz_mage.chunker import SentencePreservingChunker
        chunker = SentencePreservingChunker()
        text = "«Бұл өте маңызды жоба!» деді министр. Ол жұмыстың сәтті аяқталғанын айтты."
        sentences = chunker.split_sentences(text)
        self.assertEqual(len(sentences), 2)
        self.assertTrue(sentences[0][0].startswith("«Бұл"))

    def test_chunking_word_budget_and_overlap(self):
        from kaz_mage.chunker import SentencePreservingChunker
        chunker = SentencePreservingChunker(max_words=30, overlap_sentences=1)
        # Create 5 sentences of ~10 words each = ~50 words total
        sents = [f"Бұл құжаттың {i}-ші сөйлемі болып табылады және мағыналы ақпарат береді." for i in range(5)]
        doc = " ".join(sents)
        chunks = chunker.chunk_document(doc)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(c.word_count, 45)
            # Offset verification
            self.assertEqual(doc[c.start_char : c.end_char], c.text)

    def test_empty_and_whitespace_inputs(self):
        from kaz_mage.chunker import SentencePreservingChunker
        chunker = SentencePreservingChunker()
        self.assertEqual(chunker.chunk_document(""), [])
        self.assertEqual(chunker.chunk_document("   \n\n\t  "), [])
```

- [x] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_document_chunker.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'kaz_mage.chunker'`

- [x] **Step 3: Implement `kaz_mage/chunker.py`**

Implement sentence segmentation with abbreviation regex masking (`т.б.`, `ж.б.`, `ғ.`, `ғғ.`, `ж.`, `жж.`, `қ.`, `мыс.`, `проф.`, `акад.`), quote handling, sentence-preserving sliding packing with offsets, and export in `kaz_mage/__init__.py`.

- [x] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_document_chunker.py`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add kaz_mage/chunker.py kaz_mage/__init__.py tests/test_document_chunker.py
git commit -m "feat: implement SentencePreservingChunker with Kazakh linguistic protections"
```

---

### Task 3: Document Aggregator & Three-Tier Verdict Classification

**Files:**
- Create: `kaz_mage/aggregator.py`
- Modify: `kaz_mage/__init__.py`
- Test: `tests/test_document_aggregator.py`

**Interfaces:**
- Consumes:
  - `DocumentChunk`, `DocumentAnalysisResult` from `kaz_mage.document`
- Produces:
  - `DocumentAggregator(calibrated_threshold=0.9980, top_k_cfg=2)`
  - `aggregator.aggregate(chunks: List[DocumentChunk], total_words: int, total_sentences: int) -> DocumentAnalysisResult`

- [x] **Step 1: Write the failing test**

```python
# tests/test_document_aggregator.py
import unittest
from kaz_mage.document import DocumentChunk
from kaz_mage.aggregator import DocumentAggregator

class TestDocumentAggregator(unittest.TestCase):
    def setUp(self):
        self.agg = DocumentAggregator(calibrated_threshold=0.9980, top_k_cfg=2)

    def test_pure_human_document(self):
        chunks = [
            DocumentChunk(0, "Текст 1", 0, 10, 50, 3, ai_probability=0.02, gate_value=0.51, is_ai=False),
            DocumentChunk(1, "Текст 2", 11, 21, 60, 4, ai_probability=0.08, gate_value=0.52, is_ai=False),
            DocumentChunk(2, "Текст 3", 22, 32, 55, 3, ai_probability=0.04, gate_value=0.51, is_ai=False),
        ]
        res = self.agg.aggregate(chunks, total_words=165, total_sentences=10)
        self.assertEqual(res.verdict, "Authentic Human")
        self.assertLess(res.document_ai_probability, 0.10)
        self.assertEqual(res.ai_content_ratio, 0.0)

    def test_hybrid_document_isolated_ai_insertion(self):
        # 4 human chunks + 1 AI chunk (20% AI volume)
        chunks = [
            DocumentChunk(0, "Хуман 1", 0, 10, 50, 3, ai_probability=0.03, gate_value=0.51, is_ai=False),
            DocumentChunk(1, "Хуман 2", 11, 21, 50, 3, ai_probability=0.05, gate_value=0.52, is_ai=False),
            DocumentChunk(2, "AI блок", 22, 32, 50, 3, ai_probability=0.9995, gate_value=0.52, is_ai=True),
            DocumentChunk(3, "Хуман 3", 33, 43, 50, 3, ai_probability=0.04, gate_value=0.51, is_ai=False),
            DocumentChunk(4, "Хуман 4", 44, 54, 50, 3, ai_probability=0.02, gate_value=0.51, is_ai=False),
        ]
        res = self.agg.aggregate(chunks, total_words=250, total_sentences=15)
        self.assertEqual(res.verdict, "Partially AI / Hybrid")
        self.assertEqual(res.worst_chunk.index, 2)
        self.assertAlmostEqual(res.ai_content_ratio, 0.20, places=2)

    def test_machine_generated_document(self):
        chunks = [
            DocumentChunk(0, "AI 1", 0, 10, 50, 3, ai_probability=0.9998, gate_value=0.52, is_ai=True),
            DocumentChunk(1, "AI 2", 11, 21, 50, 3, ai_probability=0.9992, gate_value=0.52, is_ai=True),
            DocumentChunk(2, "AI 3", 22, 32, 50, 3, ai_probability=0.9999, gate_value=0.52, is_ai=True),
        ]
        res = self.agg.aggregate(chunks, total_words=150, total_sentences=9)
        self.assertEqual(res.verdict, "Machine-Generated")
        self.assertEqual(res.ai_content_ratio, 1.0)

    def test_empty_document_safety(self):
        res = self.agg.aggregate([], total_words=0, total_sentences=0)
        self.assertEqual(res.verdict, "Authentic Human")
        self.assertEqual(res.total_chunks, 0)
        self.assertIsNone(res.worst_chunk)

if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_document_aggregator.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'kaz_mage.aggregator'`

- [x] **Step 3: Implement `kaz_mage/aggregator.py`**

Implement Top-$K$ worst-chunk calculation, weighted AI ratio, three-tier classification, and export in `kaz_mage/__init__.py`.

- [x] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_document_aggregator.py`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add kaz_mage/aggregator.py kaz_mage/__init__.py tests/test_document_aggregator.py
git commit -m "feat: implement DocumentAggregator with Top-K worst-chunk pooling and three-tier classification"
```

---

### Task 4: DocumentDetector Facade with Batched Inference

**Files:**
- Create: `models/document_detector.py`
- Modify: `models/__init__.py`
- Test: `tests/test_document_detector.py`

**Interfaces:**
- Consumes:
  - `SentencePreservingChunker` from `kaz_mage.chunker`
  - `DocumentAggregator` from `kaz_mage.aggregator`
  - `MorphoContrastiveDetector` from `models.morpho_contrastive_detector`
- Produces:
  - `DocumentDetector(model, raw_tokenizer, morpheme_tokenizer, calibrated_threshold=0.9980, device=None)`
  - `detector.predict_document(text: str, top_k: int = 2, batch_size: int = 16) -> DocumentAnalysisResult`

- [x] **Step 1: Write the failing test**

```python
# tests/test_document_detector.py
import unittest
from models.document_detector import DocumentDetector

class DummyMorphoModel:
    def __init__(self):
        self.device = "cpu"
    def eval(self):
        pass
    def to(self, dev):
        self.device = dev
        return self
    def __call__(self, input_ids, attention_mask, morpheme_ids=None):
        import torch
        b = input_ids.shape[0]
        # Return mock logits: first half human, second half AI
        logits = torch.zeros((b, 2))
        logits[:, 0] = 2.0
        logits[:, 1] = -2.0
        if b > 1:
            # make last chunk AI
            logits[-1, 0] = -5.0
            logits[-1, 1] = 5.0
        gate = torch.full((b, 768), 0.52)
        return {"logits": logits, "gate": gate}

class TestDocumentDetector(unittest.TestCase):
    def test_predict_document_integration(self):
        detector = DocumentDetector(
            model=DummyMorphoModel(),
            raw_tokenizer=None,
            morpheme_tokenizer=None,
            calibrated_threshold=0.9980
        )
        long_text = "Қазақстанның цифрлық дамуы жоғары қарқынмен жүріп жатыр. " * 30
        res = detector.predict_document(long_text, batch_size=2)
        self.assertIn(res.verdict, ["Authentic Human", "Partially AI / Hybrid", "Machine-Generated"])
        self.assertGreater(res.total_chunks, 1)
        self.assertIsNotNone(res.worst_chunk)
        self.assertEqual(len(res.chunks), res.total_chunks)

if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_document_detector.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.document_detector'`

- [x] **Step 3: Implement `models/document_detector.py`**

Implement `DocumentDetector` with defensive micro-batching (`batch_size = 16`), device handling, and seamless mock/CPU fallback.

- [x] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_document_detector.py`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add models/document_detector.py models/__init__.py tests/test_document_detector.py
git commit -m "feat: implement DocumentDetector facade with batched multi-chunk inference"
```

---

### Task 5: End-to-End Hybrid Document Verification, Whole-Branch Review & Merge

- [x] **Step 1: Write hybrid document synthesis test**
Add a test in `tests/test_document_detector.py` testing a mixed 1,000-word document (authentic human paragraphs + 1 injected AI paragraph) and verifying that:
1. `res.verdict == "Partially AI / Hybrid"`.
2. `res.worst_chunk` matches the exact offset of the injected AI paragraph.
3. `res.ai_content_ratio` reflects the injected proportion.

- [x] **Step 2: Run full regression test suite**
Run all tests:
`python -m unittest tests/test_morpheme_tokenizer.py tests/test_supcon_loss.py tests/test_morpho_detector.py tests/test_metrics_evaluator.py tests/test_training_pipeline.py tests/test_kaz_raid_tier1.py tests/test_kaz_raid_tier2.py tests/test_kaz_raid_tiers3_4.py tests/test_adversarial_loss.py tests/test_kaz_raid_benchmark.py tests/test_kaz_mage_data.py tests/test_kaz_mage_long_doc.py tests/test_generate_kaz_mage.py tests/test_evaluate_kaz_mage.py tests/test_kaz_mage_kernel_build.py tests/test_multi_domain_data.py tests/test_domain_stratified_sampler.py tests/test_train_multi_domain.py tests/test_multi_domain_kernel_build.py tests/test_document_chunker.py tests/test_document_aggregator.py tests/test_document_detector.py`
Expected: All tests pass.

- [x] **Step 3: Dispatch whole-branch code reviewer subagent**
Verify zero regressions, spec compliance, and zero modifications to `aist2026/paper.tex`.

- [ ] **Step 4: Merge `feat/kaz-document-chunking-engine` into `main`**

```bash
git checkout main
git merge --no-ff feat/kaz-document-chunking-engine -m "Merge branch 'feat/kaz-document-chunking-engine' into main"
```

