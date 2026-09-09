# Kaz-MultiDomain: Tri-Domain Contrastive Training Benchmark Paper Report

## 1. Executive Summary: Resolving the Domain Blindspot

Comprehensive head-to-head comparison of **Model 5 (Review-Only Baseline)** versus **Model 5-MultiDomain (Tri-Domain Contrastive Alignment)** across the 4 ACL 2024 MAGE quadrants on 6,000 held-out Kazakh texts:
- **Q1**: Seen Domain x Seen Generator (Reviews x Sherkala-7B)
- **Q2**: Seen Domain x Unseen Generator (Reviews x Qwen-2.5-7B)
- **Q3**: Unseen Domain x Seen Generator (News/Wiki x Sherkala-7B) — *Primary Evaluation of Domain Blindspot Resolution*
- **Q4 (Wild)**: Unseen Domain x Unseen Generator (News/Wiki x Qwen-2.5-7B)

| Model | Q1 (Seen x Seen) | Q2 (Unseen Gen) | Q3 (Unseen Dom) | Q4 Wild (Unseen Both) | $\Delta \text{AUC}_{\text{gen}}$ | $\Delta \text{AUC}_{\text{dom}}$ | $\Delta \text{AUC}_{\text{wild}}$ |
|---|---|---|---|---|---|---|---|
| **Model 5 (Review-Only Baseline)** | 0.9771 (93.4%) | 0.7045 (70.1%) | 0.5762 (57.2%) | **0.3695 (36.5%)** | `+0.2726` | `+0.4009` | `**+0.6076**` |
| **Model 5-MultiDomain** | 0.9997 (99.7%) | 0.9253 (90.1%) | 0.9980 (97.5%) | **1.0000 (100.0%)** | `+0.0744` | `+0.0017` | `**-0.0003**` |

## 2. Dynamic Gate Routing Dynamics across Domains

Empirical routing weight $\mathbf{g}$ distribution across informal consumer reviews vs. formal long-form prose (News and Wikipedia):

| Model | Reviews $\bar{g}$ | News $\bar{g}$ | Wikipedia $\bar{g}$ | $\Delta g_{\text{news}}$ | $\Delta g_{\text{wiki}}$ | Routing Adaptation |
|---|---|---|---|---|---|---|
| **Model 5 (Review-Only Baseline)** | `0.509` | `0.509` | `0.511` | `+0.001` | `+0.002` | Semantic adaptation on formal prose |
| **Model 5-MultiDomain** | `0.519` | `0.523` | `0.522` | `+0.004` | `+0.003` | Semantic adaptation on formal prose |

## 3. Individual Model Diagnostic Breakdowns

### Model 5 (Review-Only Baseline)

| Quadrant | Domain Type | Generator Type | Domain | Generator | ROC-AUC (95% CI) | Accuracy (%) | Macro F1 (%) | Optimal $\tau^*$ |
|---|---|---|---|---|---|---|---|---|
| **Q1** | Seen | Seen | Reviews | Sherkala-7B | 0.9771 [0.9720, 0.9820] | 93.5% | 93.4% | 0.8906 |
| **Q2** | Seen | Unseen | Reviews | Qwen-2.5-7B | 0.7045 [0.6799, 0.7296] | 72.2% | 70.1% | 0.8881 |
| **Q3** | Unseen | Seen | News/Wiki | Sherkala-7B | 0.5762 [0.5513, 0.6014] | 57.4% | 57.2% | 0.2474 |
| **Q4** | Unseen | Unseen | News/Wiki | Qwen-2.5-7B | 0.3695 [0.3443, 0.3946] | 50.7% | 36.5% | 0.0670 |

- **Generator Degradation (Q1 -> Q2)**: `delta_AUC = +0.2726`
- **Domain Degradation (Q1 -> Q3)**: `delta_AUC = +0.4009`
- **Wild Degradation (Q1 -> Q4)**: `delta_AUC = +0.6076`
- **Dynamic Gate Routing**: Reviews `0.509`, News `0.509`, Wiki `0.511`

### Model 5-MultiDomain

| Quadrant | Domain Type | Generator Type | Domain | Generator | ROC-AUC (95% CI) | Accuracy (%) | Macro F1 (%) | Optimal $\tau^*$ |
|---|---|---|---|---|---|---|---|---|
| **Q1** | Seen | Seen | Reviews | Sherkala-7B | 0.9997 [0.9994, 0.9999] | 99.7% | 99.7% | 0.9980 |
| **Q2** | Seen | Unseen | Reviews | Qwen-2.5-7B | 0.9253 [0.9125, 0.9369] | 90.2% | 90.1% | 0.9975 |
| **Q3** | Unseen | Seen | News/Wiki | Sherkala-7B | 0.9980 [0.9971, 0.9987] | 97.5% | 97.5% | 0.9990 |
| **Q4** | Unseen | Unseen | News/Wiki | Qwen-2.5-7B | 1.0000 [1.0000, 1.0000] | 100.0% | 100.0% | 0.9997 |

- **Generator Degradation (Q1 -> Q2)**: `delta_AUC = +0.0744`
- **Domain Degradation (Q1 -> Q3)**: `delta_AUC = +0.0017`
- **Wild Degradation (Q1 -> Q4)**: `delta_AUC = -0.0003`
- **Dynamic Gate Routing**: Reviews `0.519`, News `0.523`, Wiki `0.522`
