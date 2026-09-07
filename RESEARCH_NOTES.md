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

## 3. Paper 3: Binoculars (ICML 2024) — *[Upcoming]*
* **Title:** *Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text*
* **Focus:** Training-free zero-shot cross-perplexity baseline and calibration for low-resource languages.

---

## 4. Paper 4: MultiSocial / GenAIDetect (2025) — *[Upcoming]*
* **Title:** *MultiSocial: Multilingual Benchmark of Machine-Generated Text Detection of Social-Media Texts*
* **Focus:** Social media short-text detection, register shift, and low-resource multilingual evaluation.

---

## 5. Implementation Roadmap (Quick Reference)
* [ ] **Phase 1: Literature Synthesis** — Review and distill the 4 core papers (In progress).
* [ ] **Phase 2: Benchmark Expansion** — Generate multi-LLM Kazakh synthetic data (Qwen-2.5, LLaMA-3.1) and apply Kazakh perturbation scripts.
* [ ] **Phase 3: Model Architecture** — Build explicit Morphology Encoder + KazRoBERTa Fusion.
* [ ] **Phase 4: Robust Training** — Implement supervised contrastive learning across original & perturbed pairs.
