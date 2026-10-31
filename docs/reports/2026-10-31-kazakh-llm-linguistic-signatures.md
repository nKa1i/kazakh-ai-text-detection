# Comparative Morphosyntactic Signatures of Kazakh Machine-Generated Text Across Sovereign, Reasoning, and Multilingual LLMs

- **Date:** 2026-10-31
- **Corpus Size:** 96 multi-sentence documents (12 per AI family, 24 Authentic Human controls)
- **Analysis Engine:** Auto-FST v2 (`DynamicFSTOverlay`) + `GeneratorPatternProber`

## Executive Statistical Summary

Across all evaluated sovereign (`ISSAI-KazLLM-70B`, `Sherkala-8B`), analytical reasoning (`DeepSeek-R1`), frontier multilingual (`GPT-4o`, `Qwen-2.5-72B`), and held-out cross-lingual calque (`Llama-3.3-70B-Calque`) generators, authentic Kazakh human prose exhibits statistically significant structural divergence:

- **Morpheme Chain Entropy (`chain_len_entropy`):** Human prose achieves higher productive chain entropy than synthetic text ($t = +4.91$, $p = 9.305e-07 < 0.001$, $\Delta = +0.2299$ bits).
- **Analytic Copula Crutch Density (`copula_ratio`):** Human prose avoids bureaucratic analytic copulas (`болып табылады`, `болып келеді`, `қамтамасыз етеді`) compared to synthetic text ($t = -14.41$, $p = 4.467e-47 < 0.001$, $\Delta = -6.1640$ per 100 words).
- **Synthetic Verbal Predication (`synthetic_verb_ratio`):** Human prose deploys direct synthetic finite predicates at substantially higher rates ($t = +16.05$, $p = 5.423e-58$).
- **Discourse Framing Rigidity (`discourse_rigidity_score`):** Synthetic generators systematically over-insert stereotyped clause transitions ($t = -8.69$, $p = 3.558e-18$).
- **Head-Final SOV Compliance (`sov_compliance_score`):** Cross-lingual and reasoning LLMs leak medial auxiliaries and matrix calques ($t = +4.18$, $p = 2.874e-05$).

---

## 1. Cross-Model Morphological & Syntactic Profile Table

Values report sample mean with 95% confidence interval half-width across evaluated multi-sentence Kazakh documents.

| Generator Family | Chain Entropy (bits) | Chain Var | Copula / 100w | Synth. Verb / 100w | Discourse Rigidity | SOV Compliance | Loan Harmony | Lexical TTR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ISSAI-KazLLM-70B** | 1.65 +/- 0.19 | 0.95 +/- 0.22 | 1.75 +/- 0.88 | 6.61 +/- 1.39 | 0.28 +/- 0.11 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 0.99 +/- 0.01 |
| **Sherkala-8B** | 1.73 +/- 0.10 | 0.86 +/- 0.10 | 7.30 +/- 0.89 | 1.75 +/- 0.88 | 0.09 +/- 0.06 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 0.98 +/- 0.02 |
| **DeepSeek-R1** | 1.60 +/- 0.18 | 0.80 +/- 0.21 | 2.35 +/- 1.62 | 7.97 +/- 1.53 | 0.46 +/- 0.11 | 0.93 +/- 0.05 | 0.97 +/- 0.04 | 0.97 +/- 0.01 |
| **GPT-4o** | 1.83 +/- 0.08 | 0.99 +/- 0.10 | 7.62 +/- 0.84 | 1.25 +/- 0.88 | 0.06 +/- 0.06 | 0.94 +/- 0.07 | 1.00 +/- 0.00 | 0.99 +/- 0.01 |
| **Qwen-2.5-72B** | 1.79 +/- 0.14 | 0.99 +/- 0.21 | 7.80 +/- 1.30 | 2.90 +/- 1.41 | 0.11 +/- 0.07 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.01 |
| **Llama-3.3-70B-Calque** | 1.69 +/- 0.08 | 0.90 +/- 0.11 | 10.16 +/- 1.01 | 0.25 +/- 0.48 | 0.72 +/- 0.08 | 0.82 +/- 0.07 | 1.00 +/- 0.00 | 0.96 +/- 0.04 |
| **Authentic Human** | 1.95 +/- 0.07 | 1.15 +/- 0.13 | 0.00 +/- 0.00 | 12.36 +/- 0.73 | 0.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 0.98 +/- 0.01 |

### Publication LaTeX Table Snippet

```latex
\begin{tabular}{lcccccc}
\hline
Generator Family & $H_{\text{chain}}$ & Copula/100w & SynthVerb/100w & Rigidity & SOV & LoanHarm \\
\hline
ISSAI-KazLLM-70B & 1.65 & 1.75 & 6.61 & 0.28 & 1.00 & 1.00 \\
Sherkala-8B & 1.73 & 7.30 & 1.75 & 0.09 & 1.00 & 1.00 \\
DeepSeek-R1 & 1.60 & 2.35 & 7.97 & 0.46 & 0.93 & 0.97 \\
GPT-4o & 1.83 & 7.62 & 1.25 & 0.06 & 0.94 & 1.00 \\
Qwen-2.5-72B & 1.79 & 7.80 & 2.90 & 0.11 & 1.00 & 1.00 \\
Llama-3.3-70B-Calque & 1.69 & 10.16 & 0.25 & 0.72 & 0.82 & 1.00 \\
Authentic Human & 1.95 & 0.00 & 12.36 & 0.00 & 1.00 & 1.00 \\
\hline
\end{tabular}
```

---

## 2. Primary Linguistic Weaknesses Identified per LLM Family

### ISSAI-KazLLM-70B (Sovereign Kazakh Policy LLM)

- **Structural Diagnosis:** Preserves sovereign Kazakh SOV clause finality and moderates analytic copula crutches relative to multilingual peers, but exhibits compressed morpheme chain entropy due to repetitive administrative nominalizations (-лық/-лік) and postpositional frames (шеңберінде, бойынша).
- **Dominant Z-Score Deviations vs. Human Baseline:** `discourse_rigidity_score` (mean=0.278, z=+3.10), `synthetic_verb_ratio` (mean=6.612, z=-2.94), `copula_ratio` (mean=1.749, z=+2.31)

### Sherkala-8B (Sovereign Kazakh Encyclopedic LLM)

- **Structural Diagnosis:** Maintains grammatical vowel harmony and SOV verb placement, yet over-relies on encyclopedic analytic copula constructions (болып табылады, септігін тигізеді, негіз болады), sharply depleting direct synthetic finite verb predication.
- **Dominant Z-Score Deviations vs. Human Baseline:** `copula_ratio` (mean=7.305, z=+10.85), `synthetic_verb_ratio` (mean=1.754, z=-5.43), `chain_len_entropy` (mean=1.729, z=-1.20)

### DeepSeek-R1 (Analytical Reasoning LLM)

- **Structural Diagnosis:** Displays the strongest chain-length compression and high discourse framing rigidity driven by deductive transition formulas (Осыған байланысты, Логикалық қорытындыларға сәйкес, көрсеткендей), alongside occasional impersonal calques (деп есептеледі) and hybrid technical stems.
- **Dominant Z-Score Deviations vs. Human Baseline:** `discourse_rigidity_score` (mean=0.463, z=+5.41), `copula_ratio` (mean=2.346, z=+3.22), `synthetic_verb_ratio` (mean=7.966, z=-2.25)

### GPT-4o (Frontier Multilingual LLM)

- **Structural Diagnosis:** Achieves high surface lexical variety (TTR) and fluent inflection, but substitutes native Kazakh synthetic predicates with English-calqued light-verb templates (жол ашуда, маңызға ие, мүмкіндік береді), lowering synthetic verb density and nominalizing clause endings.
- **Dominant Z-Score Deviations vs. Human Baseline:** `copula_ratio` (mean=7.618, z=+11.34), `synthetic_verb_ratio` (mean=1.251, z=-5.69), `burstiness` (mean=0.104, z=-1.22)

### Qwen-2.5-72B (Open-Weights Multilingual LLM)

- **Structural Diagnosis:** Generates uniform sentence lengths (low burstiness) and high analytic light-verb saturation (кепілдік береді, жағдай жасайды, болып табылады) typical of macroeconomic and diplomatic translationese templates.
- **Dominant Z-Score Deviations vs. Human Baseline:** `copula_ratio` (mean=7.802, z=+11.62), `synthetic_verb_ratio` (mean=2.901, z=-4.84), `burstiness` (mean=0.082, z=-1.46)

### Llama-3.3-70B-Calque (Unseen Cross-Lingual Calque LLM)

- **Structural Diagnosis:** Demonstrates severe cross-lingual translationese: extreme analytic copula/passive crutch density (болып келеді, мән беріледі, назарға алынады, қол жеткізуге болады деп күтілуде), peak discourse transition rigidity (Өз кезегінде, Осының арқасында, Тұтастай алғанда), and degraded SOV compliance.
- **Dominant Z-Score Deviations vs. Human Baseline:** `copula_ratio` (mean=10.165, z=+15.25), `discourse_rigidity_score` (mean=0.722, z=+8.65), `synthetic_verb_ratio` (mean=0.245, z=-6.21)

---

## 3. Unseen Detector Generalization Rationale

1. **Subword Tokenization Bottleneck on Agglutinative Tails:** Byte-Pair Encoding (BPE) tokenizers split deep Kazakh 1st/2nd-person possessive-case-converb stacks (`-кер-ші-ліг-іміз-ді`, `-тас-тар-ымыз-бен`, `-ған-быз`) into low-probability multi-token sequences. Autoregressive decoding systematically favors high-prior 1-2 affix 3rd-person nominal templates (`-лық/-лік`, `-лар-ы`), depressing `chain_len_entropy` regardless of the specific LLM vendor.
2. **Cross-Lingual Copula & Light-Verb Transfer:** Because pretraining corpora remain dominated by Indo-European (English/Russian) and Analytic (Chinese) copular predication (`is a key factor`, `plays a role`, `provides an opportunity`, `is expected that`), unseen LLMs consistently translate predicates into Kazakh analytic crutches (`болып табылады`, `болып келеді`, `мүмкіндік береді`, `мән беріледі`, `деп күтілуде`) rather than direct synthetic inflection.
3. **RLHF Discourse Template Anchoring:** Instruction-tuned decoders align paragraph openings to explicit transition signposts (`Осыған байланысты`, `Өз кезегінде`, `Қорытындылай келе`, `Аталмыш ... барысында`), creating elevated `discourse_rigidity_score` even when seen lexical markers from specific model families are held out under Leave-One-Generator-Out (LOGO) evaluation.
4. **Orthogonal Open-Set Defense:** Because binary unseen detection triggers on these invariant morphosyntactic deficits (`chain_len_entropy` collapse + `copula_ratio` inflation + impersonal passive saturation + calque leakage) rather than memorizing surface topic words, newly released or fine-tuned generators such as `Llama-3.3-70B-Calque` remain reliably caught while open-set centroid distance flags them as Out-of-Distribution.
