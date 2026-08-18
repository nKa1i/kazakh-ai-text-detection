---
language:
- kk
license: mit
task_categories:
- text-classification
pretty_name: KazAI-Detect Benchmark
tags:
- ai-text-detection
- kazakh
- low-resource-nlp
- kazsandra
- fst-morphology
dataset_info:
  features:
  - name: text
    dtype: string
  - name: domain
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': Human
          '1': AI
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test.csv
  - split: ood_test
    path: data/ood_test.csv
---

# 🇰🇿 KazAI-Detect: Kazakh AI-Generated Text Detection Benchmark

**KazAI-Detect** is the first multi-domain benchmark dataset designed specifically for evaluating and detecting AI-generated text in the Kazakh language.

## 📌 Dataset Overview

* **Languages:** Kazakh (`kk`)
* **Task:** Binary Text Classification (`0: Human`, `1: AI`)
* **Domains:** 
  1. *Consumer Reviews* (sourced from authentic KazSAnDRA user reviews)
  2. *Formal News* (Informburo, Egemen Qazaqstan)
  3. *Academic & Encyclopedic* (Kazakh Wikipedia)
* **Synthetic Generation:** AI counterparts generated using **Sherkala-7B** (Kazakh-adapted LLaMA) with domain-aligned prompts matching register, vocabulary, and length.

---

## 📊 Dataset Structure

### Data Fields
* `text` (*string*): The raw Kazakh text instance.
* `domain` (*string*): The genre/register (`consumer_reviews`, `news`, `wikipedia`).
* `label` (*int*): Classification target (`0`: Human, `1`: AI-Generated).

### Data Splits
| Split | Description | Samples |
| :--- | :--- | :---: |
| `test` | In-domain held-out consumer reviews | 67 |
| `ood_test` | Out-of-Distribution multi-domain benchmark (Reviews, News, Wikipedia) | 1,317 |

---

## 🚀 Quick Start & Usage

### 1. Using Hugging Face `datasets`
```python
from datasets import load_dataset

# Load the KazAI-Detect benchmark
dataset = load_dataset("nKa1i/kazakh-ai-detect")

# Inspect a sample
print(dataset['test'][0])
# {'text': 'Каспи маған өте қатты ұнайды...', 'domain': 'consumer_reviews', 'label': 0}
```

### 2. Using Pandas
```python
import pandas as pd

df_test = pd.read_csv("https://raw.githubusercontent.com/dauletanekesh/kazakh-ai-text-detection/main/dataset_package/data/ood_test.csv")
print(df_test.head())
```

---

## 📜 Citation

If you use this dataset in your research, please cite our Springer LNCS publication:

```bibtex
@inproceedings{anekesh2026detecting,
  title={Detecting AI-Generated User Reviews in Kazakh: A Study on BERT Model Performance and False Positive Reduction},
  author={Anekesh, Daulet and Ualiyeva, Irina},
  booktitle={Proceedings of the Analysis of Images, Social Networks and Texts (AIST 2026)},
  series={Lecture Notes in Computer Science (LNCS)},
  publisher={Springer},
  year={2026}
}
```

---

## ⚖️ License
This dataset is distributed under the [MIT License](LICENSE).
