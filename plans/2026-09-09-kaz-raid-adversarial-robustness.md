# Kaz-RAID: Kazakh Adversarial Robustness Benchmark & Invariant Contrastive Defense Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `kaz_raid`, the first comprehensive 9-attack adversarial robustness benchmark for Kazakh AI text detection, evaluate baseline and dual-stream detectors across 28 perturbation conditions, and harden `MorphoContrastiveDetector` using multi-view contrastive invariance defense.

**Architecture:** A modular object-oriented perturbation engine `kaz_raid/` implementing 9 attack operators across 4 linguistic tiers (Orthographic, Morphological, Code-Switching, Semantic) with budget parameterization $\epsilon \in \{0.05, 0.10, 0.20\}$; integrated with `AdvancedKazakhFSTAnalyzer` for affix manipulation; coupled with an on-the-fly adversarial invariance loss $\mathcal{L}_{\text{inv}}$ and dynamic gate interpretability logging in `MorphoContrastiveDetector`.

**Tech Stack:** Python 3.12+, PyTorch, Hugging Face Transformers, Finite State Transducers (FST), NumPy, Scikit-Learn, SciPy, Kaggle GPU Remote Runner (Dual Tesla T4).

## Global Constraints
- Target test bed: `data/kazakh_aigc_paired_2k.json` ($N = 2,000$ paired samples: 1k Human KazSAnDRA + 1k Qwen-2.5-7B).
- Codebase structure: Keep agentic folders excluded (`agent_engine/`, `docs/`, `.superpowers/`). Save specs to `specs/` and plans to `plans/`.
- Do not modify `aist2026/paper.tex` (submitted paper locked).
- Defensive execution: Pure Python fallbacks for zero-dependency local test execution when PyTorch/GPU are absent.
- Deterministic reproducibility: All perturbators must guarantee deterministic output given a seed integer.

---

## File Structure

```
kaz_raid/
├── __init__.py                # Package exports and unified KazRaidBenchmark runner
├── base.py                    # BasePerturbator abstract class with budget accounting & deterministic RNG
├── orthographic.py            # Tier 1: HomoglyphSwap, KeyboardTypo, ZeroWidthInjection
├── morphological.py           # Tier 2: SuffixTamperer, ColloquialContractor (integrated with FST)
├── code_switch.py             # Tier 3: LoanwordSwap (bidirectional), DiscourseParticle
└── semantic.py                # Tier 4: RoundTripTranslator, LLMParaphraser

models/
├── losses.py                  # SupConLoss + InvarianceLoss (cosine distance regularizer)
└── morpho_contrastive_detector.py # Adds gate logging for interpretability (Delta g)

scripts/
├── generate_kaz_raid_dataset.py # Generates 28-condition evaluation benchmark JSON
├── evaluate_kaz_raid.py         # Evaluates models across 28 conditions with Bootstrap CIs
└── prepare_kaz_raid_kernel.py   # Compiles self-contained Kaggle runner for GPU benchmark & hardening

tests/
├── test_kaz_raid_tier1.py     # Unit tests for orthographic operators
├── test_kaz_raid_tier2.py     # Unit tests for morphological operators & FST
├── test_kaz_raid_tiers3_4.py  # Unit tests for loanwords, particles, and semantic interfaces
├── test_adversarial_loss.py   # Unit tests for InvarianceLoss and multi-view batching
└── test_kaz_raid_benchmark.py # Unit tests for dataset generation and evaluation harness
```

---

### Task 1: Core Base Perturbator & Tier 1 (Orthographic Attacks)

**Files:**
- Create: `kaz_raid/base.py`
- Create: `kaz_raid/orthographic.py`
- Create: `tests/test_kaz_raid_tier1.py`

**Interfaces:**
- Consumes: Standard Python `random`, `re`
- Produces: `BasePerturbator`, `HomoglyphSwap`, `KeyboardTypo`, `ZeroWidthInjection`

- [ ] **Step 1: Write the failing test in `tests/test_kaz_raid_tier1.py`**
```python
import unittest

class TestKazRaidTier1(unittest.TestCase):
    def test_homoglyph_swap(self):
        from kaz_raid.orthographic import HomoglyphSwap
        p = HomoglyphSwap()
        text = "өте жақсы сапалы тауар"
        res = p.perturb(text, rate=0.20, seed=42)
        self.assertNotEqual(res, text)
        self.assertEqual(len(res), len(text))
        # Determinism check
        self.assertEqual(res, p.perturb(text, rate=0.20, seed=42))

    def test_keyboard_typo(self):
        from kaz_raid.orthographic import KeyboardTypo
        p = KeyboardTypo()
        text = "қазақша керемет өнім"
        res = p.perturb(text, rate=0.20, seed=42)
        self.assertNotEqual(res, text)

    def test_zero_width_injection(self):
        from kaz_raid.orthographic import ZeroWidthInjection
        p = ZeroWidthInjection()
        text = "доставка"
        res = p.perturb(text, rate=0.20, seed=42)
        self.assertTrue("​" in res or "‌" in res)

    def test_short_text_guaranteed_minimum(self):
        from kaz_raid.orthographic import HomoglyphSwap
        p = HomoglyphSwap()
        text = "зат" # 3 chars
        res = p.perturb(text, rate=0.05, seed=42)
        self.assertNotEqual(res, text)

if __name__ == "__main__":
    unittest.main()
```
- [ ] **Step 2: Run test to verify failure** (`python -m unittest tests/test_kaz_raid_tier1.py`)
- [ ] **Step 3: Implement `kaz_raid/base.py` and `kaz_raid/orthographic.py`**
  - Implement `BasePerturbator` with `perturb(text, rate, seed)` and minimum 1 perturbation logic.
  - Implement `HomoglyphSwap` with Cyrillic-Latin lookalike dictionary and case preservation.
  - Implement `KeyboardTypo` with Kazakh QWERTY adjacency and diacritic omission map.
  - Implement `ZeroWidthInjection` inserting `​` / `‌` at word interior positions.
- [ ] **Step 4: Run test to verify it passes** (`python -m unittest tests/test_kaz_raid_tier1.py`)
- [ ] **Step 5: Git commit** (`git commit -m "feat: implement Kaz-RAID base class and Tier 1 orthographic perturbators"`)

---

### Task 2: Tier 2 (Morphological Attacks) with FST Integration

**Files:**
- Create: `kaz_raid/morphological.py`
- Create: `tests/test_kaz_raid_tier2.py`

**Interfaces:**
- Consumes: `AdvancedKazakhFSTAnalyzer` from `fst_analyzer.py`, `BasePerturbator` from `kaz_raid.base`
- Produces: `SuffixTamperer`, `ColloquialContractor`

- [ ] **Step 1: Write the failing test in `tests/test_kaz_raid_tier2.py`**
```python
import unittest

class TestKazRaidTier2(unittest.TestCase):
    def test_suffix_tamperer_stripping(self):
        from kaz_raid.morphological import SuffixTamperer
        p = SuffixTamperer(mode="strip")
        text = "Каспиден заттарды тез алдым"
        res = p.perturb(text, rate=0.30, seed=42)
        self.assertNotEqual(res, text)

    def test_suffix_tamperer_vowel_harmony(self):
        from kaz_raid.morphological import SuffixTamperer
        p = SuffixTamperer(mode="harmony")
        text = "балаларға кітаптарды берді"
        res = p.perturb(text, rate=0.30, seed=42)
        self.assertNotEqual(res, text)

    def test_colloquial_contractor(self):
        from kaz_raid.morphological import ColloquialContractor
        p = ColloquialContractor()
        text = "Мен күтіп жатырмын, тауар келе жатыр"
        res = p.perturb(text, rate=0.50, seed=42)
        self.assertTrue("жатырм" in res or "кеватр" in res or "жатыр" in res)

if __name__ == "__main__":
    unittest.main()
```
- [ ] **Step 2: Run test to verify failure** (`python -m unittest tests/test_kaz_raid_tier2.py`)
- [ ] **Step 3: Implement `kaz_raid/morphological.py`**
  - Integrate with `AdvancedKazakhFSTAnalyzer` for word segmentation and suffix identification.
  - Implement `SuffixTamperer`: supporting stripping inflectional affixes and inverting back/front vowel harmony (`-лар ↔ -лер`, `-дан ↔ -ден`, `-ға ↔ -ге`).
  - Implement `ColloquialContractor`: rule-based regex mapper for 30+ spoken and SMS Kazakh review contractions (*жатырмын $	o$ жатырм*, *болып жатыр $	o$ боватр*, *келемін $	o$ келем*).
- [ ] **Step 4: Run test to verify it passes** (`python -m unittest tests/test_kaz_raid_tier2.py`)
- [ ] **Step 5: Git commit** (`git commit -m "feat: implement Kaz-RAID Tier 2 morphological and colloquial perturbators"`)

---

### Task 3: Tier 3 (Lexical & Code-Switching) & Tier 4 (Semantic Interfaces)

**Files:**
- Create: `kaz_raid/code_switch.py`
- Create: `kaz_raid/semantic.py`
- Create: `kaz_raid/__init__.py`
- Create: `tests/test_kaz_raid_tiers3_4.py`

**Interfaces:**
- Consumes: `BasePerturbator` from `kaz_raid.base`
- Produces: `LoanwordSwap`, `DiscourseParticle`, `RoundTripTranslator`, `LLMParaphraser`, `KazRaidBenchmark`

- [ ] **Step 1: Write the failing test in `tests/test_kaz_raid_tiers3_4.py`**
```python
import unittest

class TestKazRaidTiers3And4(unittest.TestCase):
    def test_loanword_swap_bidirectional(self):
        from kaz_raid.code_switch import LoanwordSwap
        p = LoanwordSwap(mode="bidirectional")
        text = "Тауар сапасы жақсы, жеткізу тез болды"
        res = p.perturb(text, rate=0.50, seed=42)
        self.assertNotEqual(res, text)
        self.assertTrue("доставка" in res.lower() or "товар" in res.lower() or "качество" in res.lower())

    def test_discourse_particle_insertion(self):
        from kaz_raid.code_switch import DiscourseParticle
        p = DiscourseParticle()
        text = "Керемет өнім, маған қатты ұнады"
        res = p.perturb(text, rate=0.50, seed=42)
        self.assertTrue(any(pt in res for pt in ["ғой", "қой", "шы", "да", "ау"]))

    def test_unified_benchmark_registry(self):
        import kaz_raid
        operators = kaz_raid.get_all_operators()
        self.assertEqual(len(operators), 9)

if __name__ == "__main__":
    unittest.main()
```
- [ ] **Step 2: Run test to verify failure** (`python -m unittest tests/test_kaz_raid_tiers3_4.py`)
- [ ] **Step 3: Implement `kaz_raid/code_switch.py`, `kaz_raid/semantic.py`, and `kaz_raid/__init__.py`**
  - Implement `LoanwordSwap` with curated 150+ review dictionary supporting `kk2ru`, `ru2kk`, and `bidirectional`.
  - Implement `DiscourseParticle` inserting authentic modal particles at clause/punctuation boundaries.
  - Implement `RoundTripTranslator` and `LLMParaphraser` interface wrappers with fallback offline simulator.
  - Export all operators and `get_all_operators()` in `kaz_raid/__init__.py`.
- [ ] **Step 4: Run test to verify it passes** (`python -m unittest tests/test_kaz_raid_tiers3_4.py`)
- [ ] **Step 5: Git commit** (`git commit -m "feat: implement Kaz-RAID Tier 3 code-switching and Tier 4 semantic operators"`)

---

### Task 4: Invariance Defense Loss & Dynamic Gate Logging

**Files:**
- Modify: `models/losses.py`
- Modify: `models/morpho_contrastive_detector.py`
- Create: `tests/test_adversarial_loss.py`

**Interfaces:**
- Consumes: PyTorch `nn.Module`, `MorphoContrastiveDetector`
- Produces: `InvarianceLoss`, `MorphoContrastiveDetector.forward(..., return_gate=True)`

- [ ] **Step 1: Write the failing test in `tests/test_adversarial_loss.py`**
```python
import unittest

class TestAdversarialLoss(unittest.TestCase):
    def test_invariance_loss_computation(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available locally")
        from models.losses import InvarianceLoss
        loss_fn = InvarianceLoss()
        z_clean = torch.randn(4, 128, requires_grad=True)
        z_adv = torch.randn(4, 128, requires_grad=True)
        loss = loss_fn(z_clean, z_adv)
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(z_clean.grad)

    def test_identical_embeddings_zero_invariance_loss(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available locally")
        from models.losses import InvarianceLoss
        loss_fn = InvarianceLoss()
        z = torch.randn(4, 128)
        loss = loss_fn(z, z)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

if __name__ == "__main__":
    unittest.main()
```
- [ ] **Step 2: Run test to verify failure** (`python -m unittest tests/test_adversarial_loss.py`)
- [ ] **Step 3: Implement `InvarianceLoss` in `models/losses.py` and gate tracking in `models/morpho_contrastive_detector.py`**
  - Implement `InvarianceLoss`: $\mathcal{L}_{	ext{inv}} = rac{1}{B} \sum_{i=1}^B (1 - \cos(\mathbf{z}_i, \mathbf{z}_i^{	ext{adv}}))$.
  - Update `MorphoContrastiveDetector`: preserve and expose `gate` tensor across clean and adversarial passes.
- [ ] **Step 4: Run test to verify it passes** (`python -m unittest tests/test_adversarial_loss.py`)
- [ ] **Step 5: Git commit** (`git commit -m "feat: implement InvarianceLoss and dynamic gate logging for adversarial defense"`)

---

### Task 5: Kaz-RAID Dataset Generator & Benchmark Harness

**Files:**
- Create: `scripts/generate_kaz_raid_dataset.py`
- Create: `scripts/evaluate_kaz_raid.py`
- Create: `tests/test_kaz_raid_benchmark.py`

**Interfaces:**
- Consumes: `kaz_raid` package, `scripts/metrics_evaluator.py`, `data/kazakh_aigc_paired_2k.json`
- Produces: `data/kaz_raid_eval_2k.json` (28 conditions), comprehensive evaluation results

- [ ] **Step 1: Write test in `tests/test_kaz_raid_benchmark.py`**
```python
import unittest

class TestKazRaidBenchmark(unittest.TestCase):
    def test_condition_generation_dry_run(self):
        from scripts.generate_kaz_raid_dataset import generate_benchmark_splits
        sample_records = [
            {"id": "0", "label": 0, "text": "Каспиден тауар алдым, өте жақсы сапа."},
            {"id": "1", "label": 1, "text": "Бұл өнім жоғары сапалы стандарттарға сәйкес келеді."}
        ]
        benchmark = generate_benchmark_splits(sample_records, rates=[0.10], selected_attacks=["homoglyph_swap"])
        self.assertIn("clean", benchmark)
        self.assertIn("homoglyph_swap_rate_0.10", benchmark)
        self.assertEqual(len(benchmark["homoglyph_swap_rate_0.10"]), 2)

if __name__ == "__main__":
    unittest.main()
```
- [ ] **Step 2: Run test to verify failure** (`python -m unittest tests/test_kaz_raid_benchmark.py`)
- [ ] **Step 3: Implement `scripts/generate_kaz_raid_dataset.py` and `scripts/evaluate_kaz_raid.py`**
  - `generate_kaz_raid_dataset.py`: builds the 28 conditions from `data/kazakh_aigc_paired_2k.json`.
  - `evaluate_kaz_raid.py`: evaluates models across all conditions, computing ROC-AUC, EER, ASR, FPR/FNR, and 95% Bootstrap CIs ($B=1,000$).
- [ ] **Step 4: Run test to verify it passes** (`python -m unittest tests/test_kaz_raid_benchmark.py`)
- [ ] **Step 5: Git commit** (`git commit -m "feat: implement Kaz-RAID dataset generator and evaluation suite"`)

---

### Task 6: Remote Kaggle GPU Runner, Benchmark Execution & Hardening

**Files:**
- Create: `scripts/prepare_kaz_raid_kernel.py`
- Update: `kaggle_runner/diagnostic_kernel.py` (or `kaggle_runner/kaz_raid_kernel.py`)
- Generate: `data/kaz_raid_benchmark_results.json`
- Generate: `data/kaz_raid_paper_report.md`

**Execution Steps:**
- [ ] **Step 1: Implement `scripts/prepare_kaz_raid_kernel.py`** bundling:
  - Full `kaz_raid/` operators
  - `AdvancedKazakhFSTAnalyzer`
  - `MorphoContrastiveDetector`
  - 28-condition evaluation loop
  - Adversarial fine-tuning loop (`Model 5-Adv` with $\mathcal{L}_{	ext{inv}}$)
  - Degradation curve generator with 95% Bootstrap CIs
- [ ] **Step 2: Generate and validate self-contained Kaggle script**
  - Run `python scripts/prepare_kaz_raid_kernel.py`
  - Run syntax check: `python -m py_compile kaggle_runner/kaz_raid_kernel.py`
- [ ] **Step 3: Push to Kaggle GPU and execute**
  - Push kernel via Kaggle API
  - Monitor execution until completion
- [ ] **Step 4: Download output results**
  - Download `kaz_raid_benchmark_results.json` and `kaz_raid_paper_report.md`
  - Copy artifacts to `data/`
- [ ] **Step 5: Run full test regression and verify outputs**
  - Run all tests across `tests/`
- [ ] **Step 6: Git commit** (`git commit -m "feat: complete Kaz-RAID benchmark execution, adversarial hardening, and paper report"`)
