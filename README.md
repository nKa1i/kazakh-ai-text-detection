# Kazakh AI-Generated Text Detection

A benchmark study comparing multilingual and Kazakh-specific BERT-based models for detecting AI-generated text in the Kazakh language, with and without Finite-State Transducer (FST) morphological analysis.

## Overview

This is, to our knowledge, the **first study on AI-generated text detection specifically for the Kazakh language**. We benchmark 8 models across two conditions — pure (raw text) and FST-augmented — on a domain-aligned dataset built from real KazSAnDRA user reviews.

## Models

Models were selected in chronological order, from early multilingual to modern Kazakh-specific:

| Model | Year | Type |
|---|---|---|
| mBERT | 2018 | Multilingual (Google, 104 languages) |
| XLM-R | 2019 | Multilingual (Meta, 100 languages) |
| KazRoBERTa | ~2022 | Monolingual Kazakh (Beeline Kazakhstan) |
| KazBERT | ~2023 | Monolingual Kazakh |

Each model is evaluated in two modes:
- **Pure** — trained on raw text
- **FST** — trained on morphologically segmented text (stems + suffixes)

## Dataset

- **Human text:** KazSAnDRA reviews (real Kazakh user reviews)
- **AI text:** Generated using Sherkala LLM, seeded from real KazSAnDRA reviews to match domain, length, and register
- **Test set:** 4000 samples, balanced 50/50, leakage-free (separate seed pools for train and test)

## Results

| Model | Mode | Accuracy | F1 | False Positives (Short) |
|---|---|---|---|---|
| mBERT | Pure | 95.42% | 95.46 | 58 |
| mBERT | FST | 93.62% | 93.71 | 70 |
| XLM-R | Pure | 95.45% | 95.52 | 67 |
| XLM-R | FST | 94.35% | 94.50 | 83 |
| KazBERT | Pure | 94.59% | 94.63 | 63 |
| KazBERT | FST | 94.56% | 94.63 | 65 |
| **KazRoBERTa** | **Pure** | **96.10%** | **96.15** | 53 |
| KazRoBERTa | FST | 96.07% | 96.07 | **36** |

## Key Findings

- **Newer = better:** Clear accuracy improvement from mBERT (95.4%) → KazRoBERTa (96.1%), confirming that domain-specific pretraining matters
- **FST does not improve accuracy** on domain-aligned data — raw text carries sufficient signal
- **FST reduces false positives:** KazRoBERTa FST produces 36 false positives vs 53 for Pure (−32%) — useful for precision-critical applications like fake review detection or academic integrity checks
- **Deployment recommendation:**
  - Use **KazRoBERTa Pure** for accuracy-first use cases
  - Use **KazRoBERTa FST** for precision-first use cases (minimizing false accusations)

## Demo

To run the AI-text detector demo locally, you need Docker and Docker Compose installed.

### Prerequisites
- Docker
- Docker Compose

### One-time Setup
1. Download or place the KazRoBERTa Pure model weights into the `data/pure_KazRoBERTa/` directory.

### Running the Demo
1. Build and start the services:
   ```bash
   docker compose up --build
   ```
2. Open your browser and navigate to the Gradio UI at: **[http://localhost:7860](http://localhost:7860)**

## Project Structure

```
├── api/                                # FastAPI backend for model inference
├── ui/                                 # Gradio frontend UI
├── docker-compose.yml                  # Docker deployment configuration
├── build_native_kazakh_ai_data.ipynb   # AI data generation (KazSAnDRA-seeded)
├── advanced_fst_evaluation.ipynb       # Training + evaluation of all 8 models
├── fst_analyzer.py                     # Kazakh morphological segmenter (FST)
└── data/
    ├── full_evaluation_results.csv     # Full results table
    ├── full_evaluation_results.json
    ├── chart_overall_acc.png
    ├── chart_short_text_acc.png
    ├── chart_false_positives.png
    └── chart_fp_reduction.png
```

## Requirements

- Python 3.9+
- PyTorch
- Transformers (HuggingFace)
- LM Studio (for AI data generation, local inference)

## License

MIT
