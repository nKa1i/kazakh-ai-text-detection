---
name: kazakh-verification-suite
description: Standardized verification suite for Kazakh AI text detection, Auto-FST v2, hybrid span benchmarks, and Kaggle NLI packaging. Use before making git commits or after modifying morphological analyzers, datasets, or UI components.
---

# Kazakh Verification Suite

Run this standardized workflow to verify zero regressions across all modeling, benchmarking, and packaging pipelines.

## 1. Environment & Pre-requisites
Always run commands using the project's primary Python 3.12 binary from the repository root:

```powershell
cd C:\Users\Roza\Desktop\projects\kazakh-ai-text-detection
$env:PYTHONIOENCODING="utf-8"
$py = "C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe"
```

## 2. Sync HuggingFace Space Mirror
Keep `src/auto_fst/` and `hf_space/auto_fst/` in exact synchronization:

```powershell
& $py -c "import shutil, os; [shutil.copy2(os.path.join('src/auto_fst', f), os.path.join('hf_space/auto_fst', f)) for f in os.listdir('src/auto_fst') if f.endswith('.py')]; print('Synced Auto-FST to hf_space')"
```

## 3. Auto-FST Morphological Regression Suite
Run all Auto-FST unit tests (40 tests):

```powershell
& $py -m unittest discover -s tests -p "test_auto_fst_*.py"
```

Evaluate exact segmentation accuracy on the 123-item gold benchmark (target: >= 95% stem, >= 94% exact segmentation):

```powershell
& $py scripts/evaluate_autofst_benchmark.py
```

## 4. Benchmark & Knowledge Corpus Verification
Run the gold benchmark, hybrid mixed spans, and knowledge corpus expansion suites:

```powershell
& $py -m unittest tests.test_autofst_gold_benchmark tests.test_hybrid_mixed_benchmark tests.test_expand_knowledge_corpus
```

Generate the 100-document span benchmark and verify Macro F1 >= 0.75:

```powershell
& $py scripts/build_hybrid_mixed_benchmark.py
```

## 5. Kaggle Runner & NLI Packaging Verification
Ensure Kaggle kernels package cleanly with the 152-passage embedded corpus:

```powershell
& $py -m unittest tests.test_kaggle_runner_package tests.test_package_nli_kaggle_kernel
```
