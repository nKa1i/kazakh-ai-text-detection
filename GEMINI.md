# Kazakh AI Text Detection & Verification: Project Rules

## 1. Python Environment & Encoding Rules
- **Python Binary**: Always execute commands using the full scientific Python 3.12 binary:
  `C:\Users\Roza\AppData\Local\Programs\Python\Python312\python.exe`
  (Do NOT use bare `python` which defaults to `Python314` lacking `pytest`, `scikit-learn`, `scipy`, `torch`, and `python-pptx`).
- **Working Directory**: Always ensure commands run from the project root:
  `cd C:\Users\Roza\Desktop\projects\kazakh-ai-text-detection`
- **PowerShell Encoding**: Always prefix Python one-liners in PowerShell with `$env:PYTHONIOENCODING="utf-8"` to prevent Windows `GBK` (`cp936`) crashes on Kazakh Cyrillic characters (`қ, ғ, ұ, ү, ә, і, ң, ө, һ`), or write test scripts using explicit UTF-8 output.

## 2. Morphological Modeling & Anti-Shortcut Guardrails
- **Algorithmic Generalization**: All morphophonological rules in `src/auto_fst/` must generalize algorithmically across open-vocabulary Kazakh stems.
- **Anti-Overfitting Constraint**: Never hardcode evaluation benchmark tokens into lookup dictionaries (`KNOWN_STEM_RESTORATION`, etc.) to artificially pass tests. Every rule must be tested and verified against held-out unseen stems.
- **Benchmark Thresholds**: `DynamicFSTOverlay` and `RussianLayoutNormalizer` must maintain $\ge$ 95% stem accuracy, $\ge$ 95% boundary F1, and $\ge$ 94% exact segmentation accuracy on `data/kaz_autofst_gold_benchmark.jsonl`.

## 3. Dual-Tree Synchronization
- Any edits to `src/auto_fst/*.py` must be kept in exact parity with `hf_space/auto_fst/*.py`.
- Any edits to `ui/*.py` must be kept in exact parity with `hf_space/ui/*.py`.

## 4. Documentation & Code Integrity
- **Zero Decorative Emojis**: Strictly zero decorative emojis across all source code, tests, docstrings, git commit messages, and agent responses.
- **Test-Driven Development (TDD)**: Always write or identify the failing test first, run it to verify failure, implement the fix, and re-run to confirm clean pass.
- **Evidence Before Assertion**: Always cite terminal outputs and metric numbers before claiming completion.
