# Kaz-MAGE: Cross-Domain & Wild Data Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the ACL 2024 MAGE benchmark protocol (Seen vs. Unseen LLMs $\times$ Seen vs. Unseen Domains) for the Kazakh language, evaluating 6 detector models across 4 quadrants on real News, Wikipedia, and Review data with dynamic gate routing analysis.

**Architecture:** 
- `kaz_mage/`: Module defining standard MAGE sample structures, quadrant filtering (`Q1`, `Q2`, `Q3`, `Q4`), and dataset loading.
- `models/` & `api/`: Extend positional embeddings ($256 \to 512$) and morpheme batch encoding ($64 \to 256$) to handle long-form paragraph texts ($150 \text{--} 350$ words).
- `scripts/`: Data generation pipeline with prefix-continuation pairing (`generate_kaz_mage_dataset.py`), evaluation suite with bootstrap CIs and gate shift tracking (`evaluate_kaz_mage.py`), and Kaggle GPU remote runner bundle (`prepare_kaz_mage_kernel.py`).

**Tech Stack:** Python 3.12, PyTorch, HuggingFace Transformers (`kz-transformers/kaz-roberta-conversational`), `AdvancedKazakhFSTAnalyzer`, Kaggle Dual Tesla T4 GPUs.

## Global Constraints
- Do NOT touch or modify `aist2026/paper.tex` under any circumstances (camera-ready paper submitted).
- All new scripts and tests must run deterministically with process-invariant seeding (`zlib.crc32`) and handle UTF-8 cleanly on Windows.
- Provide defensive fallbacks (`HAS_TORCH`, `HAS_TRANSFORMERS`) so all unit tests execute locally in any Python 3 environment without GPU requirements.
- Maintain test regression: all previous 58 tests must continue to pass.

---

### Task 1: Kaz-MAGE Dataset Infrastructure & Quadrant Slicing

**Files:**
- Create: `kaz_mage/__init__.py`
- Create: `kaz_mage/data.py`
- Test: `tests/test_kaz_mage_data.py`

**Interfaces:**
- Produces:
  - `KazMageSample(id, domain, generator, is_unseen_domain, is_unseen_generator, quadrant, prefix, text, label, char_length, word_count)`
  - `load_mage_dataset(path: str) -> list[KazMageSample]`
  - `filter_quadrant(dataset: list[KazMageSample], quadrant: str) -> list[KazMageSample]`
  - `get_quadrant_slices(dataset: list[KazMageSample]) -> dict[str, list[KazMageSample]]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kaz_mage_data.py
import unittest
import os
import json
import tempfile
from kaz_mage.data import KazMageSample, load_mage_dataset, filter_quadrant, get_quadrant_slices

class TestKazMageData(unittest.TestCase):
    def setUp(self):
        self.sample_records = [
            {
                "id": "s1",
                "domain": "consumer_reviews",
                "generator": "Sherkala-7B",
                "is_unseen_domain": False,
                "is_unseen_generator": False,
                "quadrant": "Q1",
                "prefix": "Өте жақсы",
                "text": "Өте жақсы тауар, бәрі ұнады.",
                "label": 1,
                "char_length": 28,
                "word_count": 5
            },
            {
                "id": "s2",
                "domain": "consumer_reviews",
                "generator": "Qwen-2.5-7B-Instruct",
                "is_unseen_domain": False,
                "is_unseen_generator": True,
                "quadrant": "Q2",
                "prefix": "Каспиден алдым",
                "text": "Каспиден алдым, өте тез жеткізді.",
                "label": 1,
                "char_length": 33,
                "word_count": 4
            },
            {
                "id": "s3",
                "domain": "news",
                "generator": "Sherkala-7B",
                "is_unseen_domain": True,
                "is_unseen_generator": False,
                "quadrant": "Q3",
                "prefix": "Қазақстанда жаңа заң",
                "text": "Қазақстанда жаңа заң қабылданды. Үкімет отырысында осы мәселе қаралды.",
                "label": 1,
                "char_length": 70,
                "word_count": 8
            },
            {
                "id": "s4",
                "domain": "wikipedia",
                "generator": "Qwen-2.5-7B-Instruct",
                "is_unseen_domain": True,
                "is_unseen_generator": True,
                "quadrant": "Q4",
                "prefix": "Алматы қаласы",
                "text": "Алматы қаласы – Қазақстанның ірі мәдени орталығы.",
                "label": 1,
                "char_length": 49,
                "word_count": 6
            },
            {
                "id": "s0",
                "domain": "news",
                "generator": "human",
                "is_unseen_domain": True,
                "is_unseen_generator": False,
                "quadrant": "human_news",
                "prefix": "",
                "text": "Үкімет жаңа қаулы қабылдады.",
                "label": 0,
                "char_length": 28,
                "word_count": 4
            }
        ]
        self.temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(self.sample_records, self.temp_file, ensure_ascii=False)
        self.temp_file.close()

    def tearDown(self):
        if os.path.exists(self.temp_file.name):
            os.remove(self.temp_file.name)

    def test_load_mage_dataset(self):
        dataset = load_mage_dataset(self.temp_file.name)
        self.assertEqual(len(dataset), 5)
        self.assertIsInstance(dataset[0], KazMageSample)
        self.assertEqual(dataset[0].quadrant, "Q1")

    def test_filter_quadrant(self):
        dataset = load_mage_dataset(self.temp_file.name)
        q1 = filter_quadrant(dataset, "Q1")
        self.assertEqual(len(q1), 1)
        self.assertEqual(q1[0].id, "s1")

        q4 = filter_quadrant(dataset, "Q4")
        self.assertEqual(len(q4), 1)
        self.assertEqual(q4[0].id, "s4")

    def test_get_quadrant_slices(self):
        dataset = load_mage_dataset(self.temp_file.name)
        slices = get_quadrant_slices(dataset)
        self.assertIn("Q1", slices)
        self.assertIn("Q2", slices)
        self.assertIn("Q3", slices)
        self.assertIn("Q4", slices)
        self.assertEqual(len(slices["Q1"]), 1)
        self.assertEqual(len(slices["Q4"]), 1)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_kaz_mage_data.py`
Expected: FAIL with "No module named 'kaz_mage'"

- [ ] **Step 3: Implement `kaz_mage/data.py` and `kaz_mage/__init__.py`**

```python
# kaz_mage/data.py
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class KazMageSample:
    id: str
    domain: str
    generator: str
    is_unseen_domain: bool
    is_unseen_generator: bool
    quadrant: str
    prefix: str
    text: str
    label: int
    char_length: int
    word_count: int

def load_mage_dataset(path: str) -> List[KazMageSample]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"MAGE dataset not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = []
    for item in data:
        samples.append(KazMageSample(
            id=str(item.get("id", "")),
            domain=str(item.get("domain", "")),
            generator=str(item.get("generator", "")),
            is_unseen_domain=bool(item.get("is_unseen_domain", False)),
            is_unseen_generator=bool(item.get("is_unseen_generator", False)),
            quadrant=str(item.get("quadrant", "")),
            prefix=str(item.get("prefix", "")),
            text=str(item.get("text", "")),
            label=int(item.get("label", 0)),
            char_length=int(item.get("char_length", len(str(item.get("text", ""))))),
            word_count=int(item.get("word_count", len(str(item.get("text", "")).split())))
        ))
    return samples

def filter_quadrant(dataset: List[KazMageSample], quadrant: str) -> List[KazMageSample]:
    return [s for s in dataset if s.quadrant == quadrant]

def get_quadrant_slices(dataset: List[KazMageSample]) -> Dict[str, List[KazMageSample]]:
    slices = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}
    for s in dataset:
        if s.quadrant in slices:
            slices[s.quadrant].append(s)
    return slices
```

```python
# kaz_mage/__init__.py
from .data import KazMageSample, load_mage_dataset, filter_quadrant, get_quadrant_slices

__all__ = [
    "KazMageSample",
    "load_mage_dataset",
    "filter_quadrant",
    "get_quadrant_slices"
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_kaz_mage_data.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add kaz_mage/ tests/test_kaz_mage_data.py
git commit -m "feat: implement Kaz-MAGE dataset schema and quadrant slicing"
```

---

### Task 2: Long-Document Sequence Scaling & Robust Paragraph Encoding

**Files:**
- Modify: `api/morpheme_tokenizer.py`
- Modify: `models/morpho_contrastive_detector.py`
- Test: `tests/test_kaz_mage_long_doc.py`

**Interfaces:**
- `MorphemeEncoder.__init__(max_pos=512)`: clamps position IDs to 511.
- `MorphemeTokenizer.batch_encode(texts, max_length=256)`: supports configurable sequence length up to 512.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kaz_mage_long_doc.py
import unittest

class TestKazMageLongDoc(unittest.TestCase):
    def test_morpheme_tokenizer_long_text_batch_encode(self):
        from api.morpheme_tokenizer import MorphemeTokenizer
        tok = MorphemeTokenizer()
        # 150-word long paragraph
        long_text = "Қазақстан Республикасының мемлекеттік тілі – қазақ тілі болып табылады. " * 25
        encoded = tok.encode(long_text)
        self.assertGreater(len(encoded), 100)

        tensor_256 = tok.batch_encode([long_text], max_length=256)
        self.assertEqual(tensor_256.shape, (1, 256))

        tensor_512 = tok.batch_encode([long_text], max_length=512)
        self.assertEqual(tensor_512.shape, (1, 512))

    def test_morpheme_encoder_pos_embedding_capacity(self):
        from models.morpho_contrastive_detector import MorphemeEncoder, HAS_TORCH
        encoder = MorphemeEncoder(vocab_size=250, embed_dim=768)
        if HAS_TORCH:
            import torch
            self.assertGreaterEqual(encoder.pos_embedding.num_embeddings, 512)
            # Test forward pass with 300-token sequence (exceeds old 256 bound)
            dummy_ids = torch.randint(0, 200, (2, 300))
            out = encoder(dummy_ids)
            self.assertEqual(out.shape, (2, 768))

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_kaz_mage_long_doc.py`
Expected: FAIL (tensor shape mismatch or pos_embedding index error > 256)

- [ ] **Step 3: Update `MorphemeEncoder` and `MorphemeTokenizer`**

In `models/morpho_contrastive_detector.py`:
- In `MorphemeEncoder.__init__`: change `self.pos_embedding = nn.Embedding(512, self.embed_dim)`.
- In `MorphemeEncoder.forward`:
  ```python
  pos = torch.arange(seq_len, device=morpheme_ids.device).unsqueeze(0).clamp(max=511)
  ```

In `api/morpheme_tokenizer.py`:
- In `batch_encode(self, texts: list[str], max_length: int = 256)`: default `max_length` changed to 256.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_kaz_mage_long_doc.py`
Expected: PASS

- [ ] **Step 5: Run test regression**

Run: `python -m unittest tests/test_morpheme_tokenizer.py tests/test_supcon_loss.py tests/test_morpho_detector.py tests/test_kaz_mage_data.py tests/test_kaz_mage_long_doc.py`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add api/morpheme_tokenizer.py models/morpho_contrastive_detector.py tests/test_kaz_mage_long_doc.py
git commit -m "feat: scale MorphemeEncoder position capacity to 512 tokens for long paragraphs"
```

---

### Task 3: Benchmark Dataset Mining & Assembly Pipeline

**Files:**
- Create: `scripts/generate_kaz_mage_dataset.py`
- Test: `tests/test_generate_kaz_mage.py`

**Interfaces:**
- Produces:
  - `generate_kaz_mage_dataset(output_path: str, samples_per_cell: int = 500)` -> generates `data/kaz_mage_eval_6k.json`.
  - Prefix extraction helper: `extract_prefix(text: str, num_words: int = 20) -> str`
  - Balanced pairing between Human, Sherkala, and Qwen across News, Wikipedia, and Consumer Reviews.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generate_kaz_mage.py
import unittest
import os
import json
import tempfile
from scripts.generate_kaz_mage_dataset import extract_prefix, build_synthetic_pair, assemble_kaz_mage_dataset

class TestGenerateKazMage(unittest.TestCase):
    def test_extract_prefix(self):
        text = "Бүгін Астана қаласында халықаралық экономикалық форум өз жұмысын бастады. Жиынға әлемнің отыз елінен өкілдер қатысуда."
        prefix = extract_prefix(text, num_words=5)
        self.assertEqual(prefix, "Бүгін Астана қаласында халықаралық экономикалық")

    def test_assemble_kaz_mage_dataset_mini(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "mini_mage.json")
            assemble_kaz_mage_dataset(output_path=out_file, num_samples_per_domain=10)
            self.assertTrue(os.path.exists(out_file))
            with open(out_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertGreaterEqual(len(data), 30)
            domains = set(d["domain"] for d in data)
            self.assertIn("news", domains)
            self.assertIn("wikipedia", domains)
            self.assertIn("consumer_reviews", domains)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_generate_kaz_mage.py`
Expected: FAIL with "No module named 'scripts.generate_kaz_mage_dataset'"

- [ ] **Step 3: Implement `scripts/generate_kaz_mage_dataset.py`**

Implement robust curated human news articles (500) and Wikipedia articles (500), combined with the existing Kaspi review seeds, paired with Sherkala and Qwen generations using deterministic prefix continuation and process-invariant CRC32 seeding.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_generate_kaz_mage.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_kaz_mage_dataset.py tests/test_generate_kaz_mage.py
git commit -m "feat: implement Kaz-MAGE multi-domain prefix continuation dataset generator"
```

---

### Task 4: Evaluation Suite & Gate Dynamics Tracking

**Files:**
- Create: `scripts/evaluate_kaz_mage.py`
- Test: `tests/test_evaluate_kaz_mage.py`

**Interfaces:**
- Produces:
  - `evaluate_kaz_mage_matrix(model, tokenizer, morpheme_tok, dataset, device) -> dict`:
    Evaluates Q1, Q2, Q3, Q4, reports ROC-AUC, optimal threshold $\tau^*$, F1, Acc, $\Delta \text{AUC}_{\text{domain}}$, $\Delta \text{AUC}_{\text{wild}}$, and mean gate routing $\bar{\mathbf{g}}$ per domain.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate_kaz_mage.py
import unittest
from scripts.evaluate_kaz_mage import compute_mage_metrics, calculate_domain_degradation

class TestEvaluateKazMage(unittest.TestCase):
    def test_compute_mage_metrics(self):
        # Perfect predictions
        y_true = [0, 0, 1, 1]
        y_prob = [0.1, 0.2, 0.8, 0.9]
        metrics = compute_mage_metrics(y_true, y_prob)
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_calculate_domain_degradation(self):
        quadrant_results = {
            "Q1": {"roc_auc": 0.9900},
            "Q2": {"roc_auc": 0.9850},
            "Q3": {"roc_auc": 0.9500},
            "Q4": {"roc_auc": 0.9300}
        }
        deg = calculate_domain_degradation(quadrant_results)
        self.assertAlmostEqual(deg["delta_auc_domain"], 0.0400, places=4)
        self.assertAlmostEqual(deg["delta_auc_wild"], 0.0600, places=4)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_evaluate_kaz_mage.py`
Expected: FAIL with "No module named 'scripts.evaluate_kaz_mage'"

- [ ] **Step 3: Implement `scripts/evaluate_kaz_mage.py`**

Implement full metric calculation with bootstrap 95% confidence intervals, Youden's $J$ threshold calibration, quadrant metrics, and gate routing dynamics extraction.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_evaluate_kaz_mage.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/evaluate_kaz_mage.py tests/test_evaluate_kaz_mage.py
git commit -m "feat: implement Kaz-MAGE 2x2 matrix evaluation suite with gate tracking"
```

---

### Task 5: Kaggle Dual Tesla T4 Remote Runner Bundle & Benchmark Execution

**Files:**
- Create: `scripts/prepare_kaz_mage_kernel.py`
- Test: `tests/test_kaz_mage_kernel_build.py`
- Output: `kaggle_runner/diagnostic_kernel.py`
- Output: `data/kaz_mage_benchmark_results.json`
- Output: `data/kaz_mage_paper_report.md`

**Interfaces:**
- Bundles complete self-contained executable for Kaggle GPU environment.
- Executes full 6-model evaluation across all 4 quadrants on Dual Tesla T4 GPUs.
- Downloads paper report and metrics JSON.

- [ ] **Step 1: Write test for kernel generator**

```python
# tests/test_kaz_mage_kernel_build.py
import unittest
import os
import py_compile

class TestKazMageKernelBuild(unittest.TestCase):
    def test_kernel_build_and_syntax(self):
        from scripts.prepare_kaz_mage_kernel import build_kaz_mage_kernel
        kernel_path = os.path.join("kaggle_runner", "test_mage_kernel.py")
        build_kaz_mage_kernel(output_path=kernel_path, dry_run=True)
        self.assertTrue(os.path.exists(kernel_path))
        py_compile.compile(kernel_path, doraise=True)
        os.remove(kernel_path)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_kaz_mage_kernel_build.py`
Expected: FAIL

- [ ] **Step 3: Implement `scripts/prepare_kaz_mage_kernel.py`**

Package all 6 models, dataset, and evaluation loops into `kaggle_runner/diagnostic_kernel.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_kaz_mage_kernel_build.py`
Expected: PASS

- [ ] **Step 5: Execute remote GPU benchmark on Kaggle Dual Tesla T4s**

Run:
1. `python scripts/prepare_kaz_mage_kernel.py`
2. `python scripts/push_to_kaggle.py`
3. Monitor execution until completion.
4. Download output files to `data/`.

- [ ] **Step 6: Commit**

```bash
git add scripts/prepare_kaz_mage_kernel.py tests/test_kaz_mage_kernel_build.py data/kaz_mage_*
git commit -m "feat: complete Kaz-MAGE remote benchmark execution and empirical paper report"
```

---

### Task 6: Whole-Branch Code Review, Regression Verification, and Merge

- [ ] **Step 1: Run complete test suite**
Run all tests:
`python -m unittest discover -s tests -p "test_*.py"`
Expected: All tests pass.

- [ ] **Step 2: Dispatch whole-branch code reviewer subagent**
Verify adherence to spec, no regressions, code quality, and zero changes to `aist2026/paper.tex`.

- [ ] **Step 3: Merge `feat/kaz-mage-cross-domain` into `main`**

```bash
git checkout main
git merge --no-ff feat/kaz-mage-cross-domain -m "Merge branch 'feat/kaz-mage-cross-domain' into main"
```
