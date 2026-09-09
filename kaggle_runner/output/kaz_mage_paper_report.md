# Kaz-MAGE: 2x2 Cross-Domain & Multi-Generator Benchmark Paper Report

## 1. Executive Benchmark Summary

Comprehensive evaluation of 6 detector architectures across 4 quadrants on 6,000 paired Kazakh texts:
- **Q1**: Seen Domain x Seen Generator (Kaspi Reviews x Sherkala-7B)
- **Q2**: Seen Domain x Unseen Generator (Kaspi Reviews x Qwen-2.5-7B)
- **Q3**: Unseen Domain x Seen Generator (News/Wiki x Sherkala-7B)
- **Q4 (Wild)**: Unseen Domain x Unseen Generator (News/Wiki x Qwen-2.5-7B)

| Model | Q1 (Seen x Seen) | Q2 (Unseen Gen) | Q3 (Unseen Dom) | Q4 Wild (Unseen Both) | $\Delta \text{AUC}_{\text{gen}}$ | $\Delta \text{AUC}_{\text{dom}}$ | $\Delta \text{AUC}_{\text{wild}}$ |
|---|---|---|---|---|---|---|---|
| **Model 1 (KazRoBERTa Pure)** | 0.7621 (74.9%) | 0.9537 (90.5%) | 0.6640 (67.0%) | **0.8810 (84.8%)** | `-0.1916` | `+0.0981` | `**-0.1189**` |
| **Model 2 (KazRoBERTa Hybrid FST)** | 0.9625 (90.0%) | 0.9964 (97.1%) | 0.5118 (44.2%) | **0.9505 (87.2%)** | `-0.0339` | `+0.4507` | `**+0.0120**` |
| **Model 3 (KazRoBERTa + SupCon Single-Stream)** | 1.0000 (100.0%) | 1.0000 (100.0%) | 0.6260 (61.0%) | **0.9731 (91.5%)** | `+0.0000` | `+0.3740` | `**+0.0269**` |
| **Model 4 (Dual-Stream CE-Only)** | 1.0000 (100.0%) | 1.0000 (100.0%) | 0.5979 (59.2%) | **0.8142 (72.8%)** | `+0.0000` | `+0.4021` | `**+0.1858**` |
| **Model 5 (Morpho-SupCon Full Architecture)** | 1.0000 (100.0%) | 1.0000 (100.0%) | 0.6143 (60.5%) | **0.8811 (79.7%)** | `+0.0000` | `+0.3857` | `**+0.1189**` |
| **Model 5-Adv (Invariant Contrastive Defense)** | 1.0000 (100.0%) | 1.0000 (100.0%) | 0.6449 (61.5%) | **0.9114 (82.4%)** | `+0.0000` | `+0.3551` | `**+0.0886**` |

## 2. Dynamic Gate Routing Dynamics across Domains

Empirical routing weight $\mathbf{g}$ distribution across informal consumer reviews vs. formal long-form prose:

| Model | Reviews $\bar{g}$ | News $\bar{g}$ | Wikipedia $\bar{g}$ | $\Delta g_{\text{news}}$ | $\Delta g_{\text{wiki}}$ | Routing Adaptation |
|---|---|---|---|---|---|---|
| **Model 4 (Dual-Stream CE-Only)** | `0.526` | `0.517` | `0.518` | `-0.009` | `-0.007` | Balanced routing |
| **Model 5 (Morpho-SupCon Full Architecture)** | `0.511` | `0.504` | `0.506` | `-0.008` | `-0.006` | Balanced routing |
| **Model 5-Adv (Invariant Contrastive Defense)** | `0.489` | `0.476` | `0.477` | `-0.014` | `-0.013` | Balanced routing |

## 3. Individual Model Diagnostic Breakdowns

### Model 1 (KazRoBERTa Pure)

| Quadrant | Domain Type | Generator Type | Domain | Generator | ROC-AUC (95% CI) | Accuracy (%) | Macro F1 (%) | Optimal $\tau^*$ |
|---|---|---|---|---|---|---|---|---|
| **Q1** | Seen | Seen | Reviews | Sherkala-7B | 0.7621 [0.7413, 0.7847] | 75.2% | 74.9% | 0.9857 |
| **Q2** | Seen | Unseen | Reviews | Qwen-2.5-7B | 0.9537 [0.9456, 0.9621] | 90.5% | 90.5% | 0.9904 |
| **Q3** | Unseen | Seen | News/Wiki | Sherkala-7B | 0.6640 [0.6387, 0.6874] | 69.2% | 67.0% | 0.9970 |
| **Q4** | Unseen | Unseen | News/Wiki | Qwen-2.5-7B | 0.8810 [0.8647, 0.8981] | 84.9% | 84.8% | 0.9817 |

- **Generator Degradation (Q1 -> Q2)**: `delta_AUC = -0.1916`
- **Domain Degradation (Q1 -> Q3)**: `delta_AUC = +0.0981`
- **Wild Degradation (Q1 -> Q4)**: `delta_AUC = -0.1189`

### Model 2 (KazRoBERTa Hybrid FST)

| Quadrant | Domain Type | Generator Type | Domain | Generator | ROC-AUC (95% CI) | Accuracy (%) | Macro F1 (%) | Optimal $\tau^*$ |
|---|---|---|---|---|---|---|---|---|
| **Q1** | Seen | Seen | Reviews | Sherkala-7B | 0.9625 [0.9560, 0.9689] | 90.0% | 90.0% | 0.0018 |
| **Q2** | Seen | Unseen | Reviews | Qwen-2.5-7B | 0.9964 [0.9951, 0.9975] | 97.1% | 97.1% | 0.0056 |
| **Q3** | Unseen | Seen | News/Wiki | Sherkala-7B | 0.5118 [0.4842, 0.5350] | 54.9% | 44.2% | 0.0003 |
| **Q4** | Unseen | Unseen | News/Wiki | Qwen-2.5-7B | 0.9505 [0.9418, 0.9580] | 87.2% | 87.2% | 0.0002 |

- **Generator Degradation (Q1 -> Q2)**: `delta_AUC = -0.0339`
- **Domain Degradation (Q1 -> Q3)**: `delta_AUC = +0.4507`
- **Wild Degradation (Q1 -> Q4)**: `delta_AUC = +0.0120`

### Model 3 (KazRoBERTa + SupCon Single-Stream)

| Quadrant | Domain Type | Generator Type | Domain | Generator | ROC-AUC (95% CI) | Accuracy (%) | Macro F1 (%) | Optimal $\tau^*$ |
|---|---|---|---|---|---|---|---|---|
| **Q1** | Seen | Seen | Reviews | Sherkala-7B | 1.0000 [1.0000, 1.0000] | 100.0% | 100.0% | 0.4600 |
| **Q2** | Seen | Unseen | Reviews | Qwen-2.5-7B | 1.0000 [1.0000, 1.0000] | 100.0% | 100.0% | 0.7699 |
| **Q3** | Unseen | Seen | News/Wiki | Sherkala-7B | 0.6260 [0.5991, 0.6500] | 61.2% | 61.0% | 0.0015 |
| **Q4** | Unseen | Unseen | News/Wiki | Qwen-2.5-7B | 0.9731 [0.9675, 0.9781] | 91.5% | 91.5% | 0.0025 |

- **Generator Degradation (Q1 -> Q2)**: `delta_AUC = +0.0000`
- **Domain Degradation (Q1 -> Q3)**: `delta_AUC = +0.3740`
- **Wild Degradation (Q1 -> Q4)**: `delta_AUC = +0.0269`

### Model 4 (Dual-Stream CE-Only)

| Quadrant | Domain Type | Generator Type | Domain | Generator | ROC-AUC (95% CI) | Accuracy (%) | Macro F1 (%) | Optimal $\tau^*$ |
|---|---|---|---|---|---|---|---|---|
| **Q1** | Seen | Seen | Reviews | Sherkala-7B | 1.0000 [1.0000, 1.0000] | 100.0% | 100.0% | 0.0090 |
| **Q2** | Seen | Unseen | Reviews | Qwen-2.5-7B | 1.0000 [1.0000, 1.0000] | 100.0% | 100.0% | 0.0810 |
| **Q3** | Unseen | Seen | News/Wiki | Sherkala-7B | 0.5979 [0.5729, 0.6211] | 59.7% | 59.2% | 0.0008 |
| **Q4** | Unseen | Unseen | News/Wiki | Qwen-2.5-7B | 0.8142 [0.7961, 0.8317] | 73.0% | 72.8% | 0.0007 |

- **Generator Degradation (Q1 -> Q2)**: `delta_AUC = +0.0000`
- **Domain Degradation (Q1 -> Q3)**: `delta_AUC = +0.4021`
- **Wild Degradation (Q1 -> Q4)**: `delta_AUC = +0.1858`
- **Dynamic Gate Routing**: Reviews `0.526`, News `0.517`, Wiki `0.518`

### Model 5 (Morpho-SupCon Full Architecture)

| Quadrant | Domain Type | Generator Type | Domain | Generator | ROC-AUC (95% CI) | Accuracy (%) | Macro F1 (%) | Optimal $\tau^*$ |
|---|---|---|---|---|---|---|---|---|
| **Q1** | Seen | Seen | Reviews | Sherkala-7B | 1.0000 [1.0000, 1.0000] | 100.0% | 100.0% | 0.0652 |
| **Q2** | Seen | Unseen | Reviews | Qwen-2.5-7B | 1.0000 [1.0000, 1.0000] | 100.0% | 100.0% | 0.5395 |
| **Q3** | Unseen | Seen | News/Wiki | Sherkala-7B | 0.6143 [0.5895, 0.6375] | 61.2% | 60.5% | 0.0022 |
| **Q4** | Unseen | Unseen | News/Wiki | Qwen-2.5-7B | 0.8811 [0.8668, 0.8946] | 79.7% | 79.7% | 0.0024 |

- **Generator Degradation (Q1 -> Q2)**: `delta_AUC = +0.0000`
- **Domain Degradation (Q1 -> Q3)**: `delta_AUC = +0.3857`
- **Wild Degradation (Q1 -> Q4)**: `delta_AUC = +0.1189`
- **Dynamic Gate Routing**: Reviews `0.511`, News `0.504`, Wiki `0.506`

### Model 5-Adv (Invariant Contrastive Defense)

| Quadrant | Domain Type | Generator Type | Domain | Generator | ROC-AUC (95% CI) | Accuracy (%) | Macro F1 (%) | Optimal $\tau^*$ |
|---|---|---|---|---|---|---|---|---|
| **Q1** | Seen | Seen | Reviews | Sherkala-7B | 1.0000 [1.0000, 1.0000] | 100.0% | 100.0% | 0.6052 |
| **Q2** | Seen | Unseen | Reviews | Qwen-2.5-7B | 1.0000 [1.0000, 1.0000] | 100.0% | 100.0% | 0.9957 |
| **Q3** | Unseen | Seen | News/Wiki | Sherkala-7B | 0.6449 [0.6205, 0.6682] | 62.8% | 61.5% | 0.0003 |
| **Q4** | Unseen | Unseen | News/Wiki | Qwen-2.5-7B | 0.9114 [0.8995, 0.9221] | 82.4% | 82.4% | 0.0003 |

- **Generator Degradation (Q1 -> Q2)**: `delta_AUC = +0.0000`
- **Domain Degradation (Q1 -> Q3)**: `delta_AUC = +0.3551`
- **Wild Degradation (Q1 -> Q4)**: `delta_AUC = +0.0886`
- **Dynamic Gate Routing**: Reviews `0.489`, News `0.476`, Wiki `0.477`
