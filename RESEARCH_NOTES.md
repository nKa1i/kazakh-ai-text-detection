# Kazakh AI-Generated Text Detection: Research Notes & Paper Synthesis

This document tracks key insights, methodologies, and actionable adaptations from foundational papers recommended by Prof. Guo's lab for our Master's Thesis (**Topic 1 & Topic 2: Morphology-Aware Robust AI-Text Detection in Kazakh**).

---

## 1. Paper 1: RAID (ACL 2024)
* **Title:** *RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors*
* **Authors:** Liam Dugan, Alyssa Hwang, Chris Callison-Burch et al. (UPenn)
* **Official Code:** [github.com/liamdugan/raid](https://github.com/liamdugan/raid)

### Core Takeaway
* Existing AI text detectors achieve high accuracy in-distribution (>95%), but suffer catastrophic performance drops (<60%) when exposed to **unseen generators**, **paraphrasing**, or **adversarial noise**.
* Real-world evaluation must evaluate detectors along 3 orthogonal axes: **Generators $\times$ Domains $\times$ Attacks**.

### Actionable Adaptation for Kazakh (`Kazakh-AIGC-Robust`)
1. **Multi-Generator Coverage:**
   * Expand beyond Sherkala-7B to include modern open LLMs proficient in low-resource/Kazakh text:
     * `Sherkala-7B` (Kazakh-specific LLaMA adaptation)
     * `Qwen-2.5-7B-Instruct` (Extremely strong multilingual & Central Asian language capabilities)
     * `LLaMA-3.1-8B-Instruct` (Global foundational baseline)
2. **Kazakh-Specific Adversarial Attacks:**
   * **Paraphrasing Attack:** Automated LLM rewriting prompted with informal/colloquial instructions.
   * **Orthographic / Keyboard Perturbations:** Realistic Kazakh Cyrillic typos (*ә $\leftrightarrow$ а*, *і $\leftrightarrow$ ы*, *ң $\leftrightarrow$ н*, *ү $\leftrightarrow$ ұ*).
   * **Code-Switching Attack:** Injecting common Russian loanwords with attached Kazakh grammatical suffixes (replicating authentic user reviews).
   * **Suffix / Agglutinative Swaps:** Testing sensitivity to morphological inflections.
3. **Hypothesis for Our Paper:**
   * *Hypothesis:* Standard subword models (KazRoBERTa Pure) will degrade severely under agglutinative and orthographic perturbations, whereas **Morphology-Aware KazRoBERTa (FST / Morphology Encoder)** will demonstrate significantly higher resilience and lower false-positive rates.

---

## 2. Paper 2: MAGE (ACL 2024)
* **Title:** *MAGE: Machine-generated Text Detection in the Wild*
* **Authors:** Yafang Li, Qinjian Li, Leyang Cui, Lingpeng Kong et al.
* **Official Code/Data:** [github.com/yafuly/MAGE](https://github.com/yafuly/MAGE)

### Core Takeaway
* Avoid the common scientific pitfall of random train/test splitting (which leaks generator and domain signatures into the test set).
* Formulate robust evaluation using a clean 2x2 matrix across **Model (Seen vs. Unseen)** and **Domain (Seen vs. Unseen)**.

### MAGE Evaluation Protocol Matrix
| Setting | Model (Generator) | Domain (Genre) | What It Evaluates |
| :--- | :---: | :---: | :--- |
| **1. In-Distribution (ID)** | Seen | Seen | Baseline memorization capability |
| **2. OOD-Model** | Unseen | Seen | Zero-shot transfer to unseen LLMs |
| **3. OOD-Domain** | Seen | Unseen | Cross-genre generalization (e.g., Reviews $\rightarrow$ News) |
| **4. OOD-Wild (Hardest)** | Unseen | Unseen | Real-world wild deployment |

### Actionable Adaptation for Kazakh
* **Training Set:** KazSAnDRA Reviews (Human) + Sherkala-7B Reviews (AI).
* **Test Splits:**
  1. *ID:* Sherkala-7B Reviews
  2. *OOD-Model:* Qwen-2.5-7B Reviews & LLaMA-3.1-8B Reviews
  3. *OOD-Domain:* Sherkala-7B News (*Informburo*) & Wikipedia
  4. *OOD-Wild:* Qwen-2.5-7B News & Wikipedia
* **Significance:** Directly proves that **Morphology-Aware KazRoBERTa** resists domain/model shift better than vanilla KazRoBERTa under strict, leak-free evaluation.

---

## 3. Paper 3: Binoculars (ICML 2024)
* **Title:** *Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text*
* **Authors:** Abhimanyu Hans, Avi Schwarzschild, Valeriia Cherepanova, Tom Goldstein et al. (UMD)
* **Official Code:** [github.com/ahans30/Binoculars](https://github.com/ahans30/Binoculars)

### Core Takeaway
* Training-free (zero-shot) AI detection that avoids classifier overfitting.
* Overcomes the core flaw of raw Perplexity (which falsely flags formal human text) by computing the ratio between **Perplexity** and **Cross-Perplexity** using two related LLMs ($M_1$ Observer, $M_2$ Performer):
  $$\text{Binoculars Score}(x) = \frac{\log \text{PPL}_{M_1}(x)}{\log \text{PPL}_{M_1, M_2}^{\text{cross}}(x)}$$
* Achieves extremely low false positive rates ($\text{FPR} < 0.01\%$) on English benchmarks without needing any labeled training data.

### Actionable Adaptation for Kazakh
1. **Strong Zero-Shot Baseline:**
   * Instantiate Binoculars for Kazakh using a base & instruct pair: e.g., `Qwen-2.5-7B` (Observer) and `Qwen-2.5-7B-Instruct` (Performer).
   * Benchmark Binoculars against our supervised `KazRoBERTa (Pure vs. FST)` models.
2. **Scientific Contribution (Low-Resource Bias Analysis):**
   * Demonstrate how subword token inflation ($T/W$) in agglutinative Kazakh destabilizes perplexity ratios, producing higher false-positive rates on short texts compared to English.
3. **Optional Score Calibration:**
   * Explore threshold calibration or ensemble fusion: combining Binoculars probability features with KazRoBERTa morphological representations.

---

## 4. Paper 4: MultiSocial (ACL 2025) & GenAIDetect (2025)
* **Title:** *MultiSocial: Multilingual Benchmark of Machine-Generated Text Detection of Social-Media Texts*
* **Authors:** Denis Macko, Jakub Kopál, Róbert Móro, Ivan Srba et al. (KInIT)
* **Venue:** ACL 2025
* **Related Task:** *GenAI Content Detection Task 1: Multilingual Machine-Generated Text Detection* (COLING 2025)

### Core Takeaway
* Highlights the critical gap between academic benchmarks (tested on long, clean, formal text) and real-world threats on **social media / consumer platforms** (short, noisy, informal, code-switched text).
* Identified that **short text length ($\le 50$ words)** is the primary factor causing detector accuracy to plummet and false positive rates to explode.

### Actionable Adaptation for Kazakh
1. **Domain & Register Alignment:**
   * Directly justifies the use of our authentic **KazSAnDRA** consumer reviews (Appstore, Bookstore, Mapping, Market).
2. **Solving the Short-Text False Positive Challenge:**
   * In our baseline, short reviews ($\le 60$ chars) produced the highest false positive count (53 cases). Our Hybrid FST reduced this by 43.21% ($p < 10^{-5}$).
   * We position our work as the first dedicated study tackling the MultiSocial challenge for **Central Asian agglutinative languages**.
3. **Handling Code-Switching:**
   * Replicate the real-world Kazakh social media setting by explicitly segmenting Russian/foreign commercial loanwords with Kazakh inflections (*доставкасы $\rightarrow$ доставка -сы*).

---

## 5. Architectural Synthesis: How the 4 Papers Connect

```
[ MultiSocial (ACL 2025) ] ──▶ Real-world domain: Short reviews, informal register, false positive reduction
              ▲
              │
[ RAID (ACL 2024) ] ────────▶ Robustness: Multi-LLM (Qwen, LLaMA) & Kazakh perturbations (typos, paraphrasing)
              ▲
              │
[ MAGE (ACL 2024) ] ────────▶ Experimental integrity: Strict ID, OOD-Model, OOD-Domain, OOD-Wild splits
              ▲
              │
[ Binoculars (ICML 2024) ] ──▶ Zero-shot comparison: Cross-perplexity baseline & low-resource bias analysis
```

---

## 6. Implementation Roadmap (Quick Reference)
* [x] **Phase 1: Literature Synthesis** — Review and distill the 4 core papers (Completed).
* [ ] **Phase 2: Benchmark Expansion** — Generate multi-LLM Kazakh synthetic data (Qwen-2.5, LLaMA-3.1) and apply Kazakh perturbation scripts.
* [ ] **Phase 3: Model Architecture** — Build explicit Morphology Encoder + KazRoBERTa Fusion.
* [ ] **Phase 4: Robust Training** — Implement supervised contrastive learning across original & perturbed pairs.
