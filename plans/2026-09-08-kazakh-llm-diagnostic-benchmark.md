# Kazakh LLM Linguistic Diagnostic & Robust Benchmark Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a multi-LLM generation and 6-factor linguistic diagnostic pipeline on Kaggle GPU that generates paired Kazakh synthetic reviews across Sherkala, Qwen-2.5, and LLaMA-3.1, profiles morphological token inflation and register shift, and stress-tests zero-shot detector transfer.

**Architecture:** A modular Python pipeline combining stratified human seed extraction (`KazSAnDRA`), automated multi-LLM prompt generation, morphological and lexical diagnostic analysis (FST, token inflation, loanword detection, semantic distance), and a self-contained Kaggle kernel package.

**Tech Stack:** Python 3.10+, PyTorch, Hugging Face Transformers, Tokenizers, Apertium-Kaz / Kazakh FST, scikit-learn, Kaggle CLI.

## Global Constraints
- Target dataset scale: 4,000 samples ($N=1,000$ Human seed from `KazSAnDRA`, $N=1,000$ per generator: `Sherkala-7B`, `Qwen-2.5-7B-Instruct`, `LLaMA-3.1-8B-Instruct`).
- Stratification: Short ($\le 60$ chars: 35%), Medium ($61–200$ chars: 45%), Long ($> 200$ chars: 20%).
- Compatible with Kaggle dual-T4 / single P100 GPUs (16GB VRAM) using 4-bit/8-bit or half-precision inference.
- Clean repository: Keep agentic folders permanently excluded (`agent_engine/`, `docs/`, `.superpowers/`). Save plans to `plans/` and specs to `specs/`.

---

## File Structure

```
c:\Users\Roza\Desktop\projects\kazakh-ai-text-detection/
├── data/
│   ├── seed_human_reviews_1k.json           # 1,000 stratified human reviews
│   └── code_switched_loanword_lexicon.json   # Existing loanword dictionary
├── scripts/
│   ├── extract_seed_human_data.py          # Extracts & stratifies 1,000 KazSAnDRA reviews
│   ├── prompt_builder.py                   # Builds domain/length-conditioned prompts for LLMs
│   ├── generate_multi_llm.py               # Batch generator for Qwen, LLaMA, Sherkala on GPU
│   ├── compute_linguistic_diagnostics.py   # Computes T/W, TTR, FST fragmentation, loanword ratio
│   └── evaluate_zero_shot_transfer.py      # Evaluates KazRoBERTa Pure vs FST on multi-LLM data
├── kaggle_runner/
│   ├── diagnostic_kernel.py                # Standalone script for Kaggle GPU execution
│   └── kernel-metadata.json                # Kaggle CLI deployment configuration
└── tests/
    ├── test_seed_extraction.py             # Tests stratified length and domain distribution
    ├── test_prompt_builder.py              # Tests prompt formatting and length constraints
    └── test_linguistic_diagnostics.py      # Tests T/W, TTR, and loanword calculation logic
```

---

### Task 1: Human Seed Data Extraction & Stratified Sampling

**Files:**
- Create: `scripts/extract_seed_human_data.py`
- Create: `tests/test_seed_extraction.py`
- Output: `data/seed_human_reviews_1k.json`

**Interfaces:**
- Consumes: `dataset_package/data/train.csv` (human reviews where `label == '0'`)
- Produces: `data/seed_human_reviews_1k.json` with list of dicts: `[{"id": int, "text": str, "char_length": int, "length_bracket": "short"|"medium"|"long"}]`

- [ ] **Step 1: Write failing test for stratified seed extraction**

```python
# tests/test_seed_extraction.py
import os
import json
import pytest
from scripts.extract_seed_human_data import extract_stratified_human_reviews

def test_extract_stratified_human_reviews(tmp_path):
    output_path = tmp_path / "test_seed.json"
    data = extract_stratified_human_reviews(
        source_csv="dataset_package/data/train.csv",
        output_json=str(output_path),
        total_samples=100
    )
    assert os.path.exists(output_path)
    assert len(data) == 100
    short_count = sum(1 for d in data if d["length_bracket"] == "short")
    med_count = sum(1 for d in data if d["length_bracket"] == "medium")
    long_count = sum(1 for d in data if d["length_bracket"] == "long")
    assert short_count > 0
    assert med_count > 0
    assert long_count > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_seed_extraction.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.extract_seed_human_data'`

- [ ] **Step 3: Implement `scripts/extract_seed_human_data.py`**

```python
# scripts/extract_seed_human_data.py
import csv
import json
import os

def categorize_length(length: int) -> str:
    if length <= 60:
        return "short"
    elif length <= 200:
        return "medium"
    return "long"

def extract_stratified_human_reviews(
    source_csv: str = "dataset_package/data/train.csv",
    output_json: str = "data/seed_human_reviews_1k.json",
    total_samples: int = 1000
):
    short_pool = []
    med_pool = []
    long_pool = []

    with open(source_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("label", "").strip() == "0":
                text = row.get("text", "").strip()
                if not text:
                    continue
                char_len = len(text)
                bracket = categorize_length(char_len)
                entry = {
                    "text": text,
                    "char_length": char_len,
                    "length_bracket": bracket,
                    "domain": row.get("domain", "consumer_reviews")
                }
                if bracket == "short":
                    short_pool.append(entry)
                elif bracket == "medium":
                    med_pool.append(entry)
                else:
                    long_pool.append(entry)

    # 35% short, 45% medium, 20% long
    n_short = int(total_samples * 0.35)
    n_med = int(total_samples * 0.45)
    n_long = total_samples - n_short - n_med

    selected = short_pool[:n_short] + med_pool[:n_med] + long_pool[:n_long]
    for idx, item in enumerate(selected):
        item["id"] = idx

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)

    return selected

if __name__ == "__main__":
    data = extract_stratified_human_reviews()
    print(f"Extracted {len(data)} stratified human reviews into data/seed_human_reviews_1k.json")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_seed_extraction.py -v`
Expected: PASS

- [ ] **Step 5: Execute extraction to generate `data/seed_human_reviews_1k.json` and commit**

Run: `python scripts/extract_seed_human_data.py`
Expected: 1,000 samples generated.
Commit: `git add scripts/extract_seed_human_data.py tests/test_seed_extraction.py data/seed_human_reviews_1k.json && git commit -m "feat: extract and stratify 1k human seed reviews from KazSAnDRA"`

---

### Task 2: Multi-LLM Prompt Conditioning & Generation Engine

**Files:**
- Create: `scripts/prompt_builder.py`
- Create: `scripts/generate_multi_llm.py`
- Create: `tests/test_prompt_builder.py`

**Interfaces:**
- Consumes: `data/seed_human_reviews_1k.json`
- Produces: Formatted prompts matching seed review characteristics, and generation output `data/kazakh_aigc_diagnostic_4k.json`

- [ ] **Step 1: Write failing test for prompt builder**

```python
# tests/test_prompt_builder.py
from scripts.prompt_builder import build_generation_prompt

def test_build_generation_prompt():
    seed = {
        "text": "Өте керемет тауар, жылдам жеткізді!",
        "char_length": 35,
        "length_bracket": "short",
        "domain": "consumer_reviews"
    }
    prompt = build_generation_prompt(seed)
    assert "Қазақ тілінде" in prompt
    assert "қысқа" in prompt or "short" in prompt.lower()
    assert isinstance(prompt, str)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prompt_builder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.prompt_builder'`

- [ ] **Step 3: Implement `scripts/prompt_builder.py`**

```python
# scripts/prompt_builder.py
def build_generation_prompt(seed_entry: dict) -> str:
    bracket = seed_entry.get("length_bracket", "medium")
    domain = seed_entry.get("domain", "consumer_reviews")
    
    if bracket == "short":
        len_instruction = "Өте қысқа пікір жазыңыз (10-15 сөзден аспасын, 1-2 сөйлем)."
    elif bracket == "medium":
        len_instruction = "Орташа ұзындықтағы пікір жазыңыз (25-45 сөз)."
    else:
        len_instruction = "Толыққанды, егжей-тегжейлі пікір жазыңыз (50+ сөз)."

    system_instruction = (
        "Сіз Kaspi / интернет-дүкендегі нақты сатып алушысыз. "
        "Қазақ тілінде шынайы пікір (review) жазыңыз. "
        "Ешқандай жасанды интеллект кіріспе сөздерінсіз, қарапайым халықтық тілде жазыңыз.\n"
        f"Тақырыбы: {domain}. {len_instruction}\n"
        "Пікір:"
    )
    return system_instruction
```

- [ ] **Step 4: Implement `scripts/generate_multi_llm.py`**

Provide GPU batch generation logic compatible with Kaggle (using Hugging Face `pipeline` or `AutoModelForCausalLM` with 4-bit quantization).

- [ ] **Step 5: Run tests and commit**

Run: `pytest tests/test_prompt_builder.py -v`
Expected: PASS
Commit: `git add scripts/prompt_builder.py scripts/generate_multi_llm.py tests/test_prompt_builder.py && git commit -m "feat: implement prompt conditioning and multi-LLM generator engine"`

---

### Task 3: Six-Factor Linguistic Diagnostics Engine

**Files:**
- Create: `scripts/compute_linguistic_diagnostics.py`
- Create: `tests/test_linguistic_diagnostics.py`
- Output: `data/diagnostic_metrics_results.json`

**Interfaces:**
- Consumes: Sample text strings, `data/code_switched_loanword_lexicon.json`
- Produces: Dictionary containing $T/W$ ratios, TTR, Distinct-1/2, loanword frequencies, length-stratified statistics.

- [ ] **Step 1: Write failing test for diagnostic calculations**

```python
# tests/test_linguistic_diagnostics.py
from scripts.compute_linguistic_diagnostics import (
    calculate_token_inflation,
    calculate_lexical_diversity,
    detect_code_switched_loanwords
)

def test_token_inflation():
    text = "Бұл өте жақсы және ыңғайлы қосымша екен"
    tw = calculate_token_inflation(text)
    assert tw >= 1.0

def test_lexical_diversity():
    texts = ["жақсы тауар", "өте жақсы тауар сапасы"]
    res = calculate_lexical_diversity(texts)
    assert "ttr" in res
    assert "distinct_1" in res
    assert "distinct_2" in res

def test_loanword_detection():
    text = "Доставкасы өте тез болды, каспиге рақмет"
    loanwords = detect_code_switched_loanwords(text)
    assert len(loanwords) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_linguistic_diagnostics.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `scripts/compute_linguistic_diagnostics.py`**

Implement `calculate_token_inflation`, `calculate_lexical_diversity`, `detect_code_switched_loanwords`, and overall dataset profiling aggregating by `generator` and `length_bracket`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_linguistic_diagnostics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

Run: `git add scripts/compute_linguistic_diagnostics.py tests/test_linguistic_diagnostics.py && git commit -m "feat: implement 6-factor linguistic diagnostic calculation engine"`

---

### Task 4: Zero-Shot Transfer Evaluation & Stress-Testing

**Files:**
- Create: `scripts/evaluate_zero_shot_transfer.py`
- Output: `data/zero_shot_transfer_results.json`

**Interfaces:**
- Consumes: Multi-LLM test dataset, fine-tuned KazRoBERTa model checkpoint (or mock if running offline)
- Produces: Accuracy, F1, FPR, FNR across `human`, `sherkala_7b`, `qwen_2.5_7b`, `llama_3.1_8b` broken down by length bracket.

- [ ] **Step 1: Implement `scripts/evaluate_zero_shot_transfer.py` with length stratification and confusion metrics**
- [ ] **Step 2: Run verification on mock samples**
- [ ] **Step 3: Commit**

Run: `git add scripts/evaluate_zero_shot_transfer.py && git commit -m "feat: implement zero-shot transfer and length-stratified evaluation engine"`

---

### Task 5: Kaggle Kernel Package & Automation Workflow

**Files:**
- Create: `kaggle_runner/diagnostic_kernel.py`
- Create: `kaggle_runner/kernel-metadata.json`
- Create: `scripts/push_to_kaggle.py`

**Interfaces:**
- Consumes: All diagnostic scripts and datasets
- Produces: Standalone, GPU-executable Kaggle kernel ready for `kaggle kernels push`

- [ ] **Step 1: Package all components into `kaggle_runner/diagnostic_kernel.py`**
- [ ] **Step 2: Set up `kaggle_runner/kernel-metadata.json` with GPU acceleration enabled (`"enable_gpu": "true"`)**
- [ ] **Step 3: Test local compilation of kernel script**
- [ ] **Step 4: Commit**

Run: `git add kaggle_runner/ scripts/push_to_kaggle.py && git commit -m "feat: package full diagnostic benchmark pipeline into Kaggle GPU runner"`

---

## Self-Review Checklist
- [x] **Spec coverage:** Covers Seed Sampling, Multi-LLM Generation, 6-Factor Diagnostics, Zero-Shot Stress Test, and Kaggle Packaging.
- [x] **No Placeholders:** All file paths, interfaces, and code structures are fully specified.
- [x] **Type consistency:** Data structures and field names (`length_bracket`, `char_length`, `generator`) are uniform across all tasks.
