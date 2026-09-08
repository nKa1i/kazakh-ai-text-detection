# Design Specification: Kazakh Adversarial Robustness Benchmark (Kaz-RAID) & Invariant Contrastive Defense

**Date:** 2026-09-09  
**Status:** Approved by User  
**Target:** Master's Thesis & ACL/EMNLP Submission (Prof. Guo's Lab)  
**Parent Model:** `MorphoContrastiveDetector` (Topic 1: 0.9960 ROC-AUC, 3.10% EER, 100% Short FP Elimination)  

---

## 1. Problem Formulation & Theoretical Motivation

### 1.1 The Low-Resource Adversarial Evasion Challenge
While existing detectors achieve high accuracy on clean in-distribution data, real-world deployment exposes them to adversarial evasion. In English, the RAID benchmark (ACL 2024: Dugan et al.) showed that even top detectors degrade drastically when text is perturbed. 

However, RAID was designed exclusively for morphologically isolating, high-resource English. In Kazakh (a morphologically rich, agglutinative, Turkic language), detector vulnerabilities operate along fundamentally different linguistic axes:
1. **Subword BPE Explosion:** Standard BERT tokenizers (SentencePiece / BPE) allocate narrow vocabularies to Kazakh. A single typo or homoglyph splits an 8-character word into 4–6 byte fragments, shattering contextual representations.
2. **Agglutinative Suffix Tampering:** Human Kazakh text relies on chains of grammatical suffixes (*үй-лер-іміз-де-гі-лер-ден*). Tampering with or stripping these affixes bypasses detectors relying purely on surface statistical signatures.
3. **Colloquial Register & Code-Switching:** Online consumer reviews (e.g. Kaspi.kz) exhibit extensive Russian-Kazakh loanword mixing (*доставкасы*, *каспиден*) and colloquial verbal contractions (*жатырмын $\to$ жатырм*), whereas LLMs default to hyper-formal, un-contracted Kazakh.
4. **Cyrillic-Latin Homoglyphs in Central Asia:** Students in Kazakhstan routinely replace Cyrillic letters with Latin lookalikes (*а/a, о/o, е/e, с/c*) to bypass university plagiarism systems (StrikePlagiarism, Turnitin).

### 1.2 Theoretical Hypotheses
* **Hypothesis 1 (BPE Fragility of Single-Stream BERT):** Standard `KazRoBERTa Pure` will suffer severe degradation ($ASR > 50\%$) under character homoglyphs and affix tampering because its single-stream subword representation fractures into out-of-vocabulary bytes.
* **Hypothesis 2 (Morphological Inductive Bias as a Defense):** Our dual-stream `MorphoContrastiveDetector` will exhibit inherent robustness ($ASR < 15\%$) because its secondary FST morpheme stream operates over closed-vocabulary grammatical affixes ($V_M \approx 250$) that survive surface subword corruption.
* **Hypothesis 3 (Invariant Contrastive Hardening):** Fine-tuning with an on-the-fly adversarial invariance loss ($\mathcal{L}_{\text{inv}}$) will align clean and perturbed representations, maintaining $> 90\%$ accuracy across all attack tiers without sacrificing clean generalization.

---

## 2. Kaz-RAID Attack Taxonomy (9 Attacks across 4 Tiers)

```
Kaz-RAID Attack Taxonomy (9 Attacks across 4 Tiers):
├── Tier 1: Character / Orthographic Attacks
│   ├── 1. Cyrillic-Latin Homoglyph Substitution (lookalike swaps: а↔a, о↔o, с↔c, х↔x, р↔p, е↔e)
│   ├── 2. Kazakh Keyboard Typos (layout distance swaps & diacritic omission: қ↔к, ң↔н, ғ↔г, ү↔у, ә↔а)
│   └── 3. Zero-Width Space Injection (invisible BPE boundary splitters: \u200B, \u200C)
├── Tier 2: Morphological Attacks (Unique to Kazakh Agglutination)
│   ├── 4. Suffix Stripping & Vowel Harmony Mutation (stripping -лар, -ден; mutating harmony -лар↔-лер)
│   └── 5. Colloquial Suffix Contraction (spoken shortcuts: жатырмын → жатырм, болып жатыр → боватр)
├── Tier 3: Lexical & Code-Switching Attacks
│   ├── 6. Russian-Kazakh Loanword Substitution (доставка ↔ жеткізу, скидка ↔ жеңілдік)
│   └── 7. Discourse Particle Insertion (authentic modal particles: ғой, шы, да, ау, қой)
└── Tier 4: Sentence & Semantic Attacks
    ├── 8. Round-Trip Back-Translation (Kazakh → Russian/English → Kazakh)
    └── 9. Adversarial LLM Paraphrase (Qwen-2.5-7B instructed to rephrase and evade)
```

---

## 3. Component Architecture (`kaz_raid/` Package)

```
kaz_raid/
├── __init__.py                # Package entry point & unified KazRaidBenchmark runner
├── base.py                    # BasePerturbator abstract class with budget accounting & deterministic RNG
├── orthographic.py            # Tier 1: HomoglyphSwap, KeyboardTypo, ZeroWidthInjection
├── morphological.py           # Tier 2: SuffixTamperer, ColloquialContractor (integrated with FST)
├── code_switch.py             # Tier 3: LoanwordSwap (bidirectional), DiscourseParticle
└── semantic.py                # Tier 4: RoundTripTranslator, LLMParaphraser
```

### 3.1 Base Operator Interface
```python
class BasePerturbator:
    def __init__(self, name: str, tier: str):
        self.name = name
        self.tier = tier

    def perturb(self, text: str, rate: float = 0.10, seed: int = 42) -> str:
        """
        Perturbs text with a target budget rate in {0.05, 0.10, 0.20}.
        Guarantees:
        1. Deterministic output given the same seed.
        2. Minimum 1 perturbation on short texts when rate > 0 and eligible targets exist.
        3. Preservation of capitalization, punctuation, and whitespace.
        """
        raise NotImplementedError
```

### 3.2 Operator Specifications
1. **`HomoglyphSwap`**: Maps `{а: a, е: e, о: o, р: p, с: c, х: x, у: y, і: i}` and uppercase variants.
2. **`KeyboardTypo`**: Maps adjacent keys on Kazakh standard layout and drops/substitutes diacritics `{қ: к, ң: н, ғ: г, ү: у, ә: а, ө: о, ұ: у, h: х}`.
3. **`ZeroWidthInjection`**: Injects `\u200B` or `\u200C` inside word stems at rate $\epsilon$.
4. **`SuffixTamperer`**: Segments words via `AdvancedKazakhFSTAnalyzer`. At rate $\epsilon$, either strips inflectional suffixes or inverts front/back vowel harmony.
5. **`ColloquialContractor`**: Regex-guided reduction of formal verb inflections into colloquial review forms.
6. **`LoanwordSwap`**: 150+ curated bilingual review dictionary. Supports `kk2ru`, `ru2kk`, and `bidirectional` modes.
7. **`DiscourseParticle`**: Inserts authentic modal particles `{ғой, қой, да, де, шы, ші, ау}` at clause/comma boundaries.
8. **`RoundTripTranslator`**: Machine translation back-translation pipeline (NLLB / MarianMT).
9. **`LLMParaphraser`**: Qwen-2.5-7B-Instruct with adversarial evasion prompts.

---

## 4. Phase 2A: Attack Vulnerability Benchmark Protocol

### 4.1 Evaluation Conditions ($N = 28$ Conditions)
Evaluated on the $N = 2,000$ paired test bed (`data/kazakh_aigc_paired_2k.json`):
- **Clean Baseline:** $\epsilon = 0.00$
- **3 Budgets per Attack:**
  - Light: $\epsilon = 0.05$ (5% rate)
  - Medium: $\epsilon = 0.10$ (10% rate)
  - Heavy: $\epsilon = 0.20$ (20% rate)
- Total conditions: $1 + (9 \times 3) = 28$ benchmark runs per model.

### 4.2 Models Evaluated
1. **`Model 1 (KazRoBERTa Pure)`**: Standard single-stream subword BERT.
2. **`Model 2 (KazRoBERTa Hybrid FST)`**: Single-stream BERT with FST morpheme preprocessing.
3. **`Model 5 (Morpho-SupCon Full)`**: Dual-stream semantic + morphological gated network.

### 4.3 Evaluation Metrics
- **ROC-AUC & Equal Error Rate (EER):** Threshold-independent separability.
- **Attack Success Rate (ASR):** Proportion of true AI samples that flip to Human (false negatives).
- **False Positive Rate (FPR):** Proportion of true Human samples falsely accused of being AI.
- **Robustness Retention:** $R_{\text{ret}} = \frac{\text{AUC}_{\text{perturbed}}}{\text{AUC}_{\text{clean}}} \times 100\%$.
- **Statistical Significance:** 95% Bootstrap Confidence Intervals ($B = 1,000$ resamples) for all degradation curves.

---

## 5. Phase 2B: Invariant Contrastive Defense Hardening

### 5.1 Training Objective
To harden the model against all 9 attacks without adding parameters or inference latency, we augment training with an on-the-fly multi-view invariance objective:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda_{\text{supcon}} \mathcal{L}_{\text{SupCon}} + \lambda_{\text{inv}} \mathcal{L}_{\text{inv}}$$

Where:
- For each clean sample $x_i$, an adversarial counterpart $x_i^{\text{adv}}$ is generated by sampling an attack operator at random budget $\epsilon \sim U(0.05, 0.20)$.
- **Invariance Regularizer:**
  $$\mathcal{L}_{\text{inv}} = \frac{1}{B} \sum_{i=1}^B \left(1 - \frac{\mathbf{z}_i \cdot \mathbf{z}_i^{\text{adv}}}{\|\mathbf{z}_i\|_2 \|\mathbf{z}_i^{\text{adv}}\|_2}\right)$$
- Hyperparameters: $\lambda_{\text{supcon}} = 0.5$, $\lambda_{\text{inv}} = 0.25$, $\tau = 0.07$.

### 5.2 Dynamic Gate Interpretability
We track the activation shift of the gating vector $\mathbf{g} = \sigma(\mathbf{W}_g [\mathbf{h}_{\text{sem}}; \mathbf{h}_{\text{morph}}] + \mathbf{b}_g)$:
$$\Delta \mathbf{g} = \mathbf{g}_{\text{adv}} - \mathbf{g}_{\text{clean}}$$
- **Hypothesis:** Under character and homoglyph attacks, $\Delta \mathbf{g} < 0$ (the gate shifts reliance toward the robust morphological stream $\mathbf{h}_{\text{morph}}$), demonstrating mechanistic self-healing.

---

## 6. Verification & Test Plan

1. **Automated Unit Tests (`tests/test_kaz_raid.py`):**
   - Verify every operator produces valid strings with deterministic seeds.
   - Verify minimum perturbation on short texts.
   - Verify case preservation (uppercase, titlecase, lowercase).
   - Verify FST fallback behavior on unsegmented tokens.
2. **Invariance Loss Test (`tests/test_adversarial_loss.py`):**
   - Verify gradient flow through both clean and adversarial views.
   - Verify that identical representations yield $\mathcal{L}_{\text{inv}} = 0$.
3. **Kaggle GPU Execution:**
   - Execute the 28-condition evaluation matrix across M1, M2, and M5 on Dual Tesla T4 GPUs.
   - Train the hardened Model 5-Adv checkpoint and measure post-defense retention.
   - Generate `data/kaz_raid_paper_report.md` with complete degradation tables and LaTeX figures.
