# Kaggle Remote Batch Generation & Dataset Harvesting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the Kaggle remote environment `bitsandbytes` quantization dependency, re-package the 72-article Kazakh knowledge corpus, deploy Kernel Version 23 to Kaggle Dual Tesla T4 GPUs (`dauletanekesh/kazakh-gpu-runner-nb`), verify healthy GPU startup (`KernelWorkerStatus.RUNNING`), and establish the monitoring and data harvesting protocol for generating 3,000 Kazakh-FEVER claims and 1,000 social media posts.

**Architecture:**
1. Self-healing environment bootstrap in `kaggle_runner/generate_fever_and_social_kernel.py`: Dynamically installs and upgrades `bitsandbytes>=0.46.1` and `accelerate` at GPU initialization before loading model quantization configs.
2. In-kernel automated corpus deployment: Embeds the 72-article gzip+base64 corpus payload and unpacks it cleanly into `/kaggle/working/kazakh_knowledge_corpus.jsonl`.
3. Automated push and remote dispatch via `scripts/push_to_kaggle.py` and the Kaggle CLI.
4. Monitoring handshake confirming startup without `ImportError`, transitioning to batch inference with per-record file flushing (`f.flush()`).

**Tech Stack:** Python 3.12, Kaggle CLI, PyTorch, Hugging Face Transformers (`Qwen/Qwen2.5-7B-Instruct`), BitsAndBytes (4-bit NF4), `unittest`.

## Global Constraints

- Under no circumstances shall `aist2026/paper.tex` be touched or modified (0 diff lines).
- Zero decorative emojis across all code, datasets, comments, documentation, and papers.
- All Python file operations must explicitly specify `encoding='utf-8'`.
- All code files and documents must contain complete, production-ready content with zero placeholders ("TODO", "TBD").
- Maintain backward compatibility with existing tests (417+ passing tests).

---

### Task 1: Environment Auto-Upgrade & bitsandbytes Fix in Kernel Script

**Files:**
- Modify: `kaggle_runner/generate_fever_and_social_kernel.py:253-278`
- Test: `tests/test_kaggle_runner_package.py`

**Interfaces:**
- Consumes: `LLMRunner.initialize_model()` method.
- Produces: Robust dynamic dependency resolution:
  - Invokes `pip install -q -U "bitsandbytes>=0.46.1" accelerate` when running on GPU / non-dry-run.
  - Clears pre-loaded references to `bitsandbytes` and `transformers` in `sys.modules` to guarantee the upgraded library is imported.
  - Retains instant return for `dry_run=True`.

- [ ] **Step 1: Write unit test in `tests/test_kaggle_runner_package.py` verifying bootstrap behavior**
- [ ] **Step 2: Run test to observe behavior**
- [ ] **Step 3: Update `kaggle_runner/generate_fever_and_social_kernel.py` with dynamic bootstrap**
- [ ] **Step 4: Run unit tests to verify pass**
- [ ] **Step 5: Commit changes**

---

### Task 2: Package and Push Kernel Version 23 to Kaggle

**Files:**
- Modify: `kaggle_runner/generate_fever_and_social_kernel.py` (corpus payload embedded)
- Modify: `kaggle_runner/kernel-metadata.json`
- Script: `scripts/push_to_kaggle.py`

**Interfaces:**
- Consumes: `kaggle_runner/kazakh_knowledge_corpus.jsonl` (72 articles).
- Produces:
  - Gzip + Base64 embedded payload updated in `generate_fever_and_social_kernel.py`.
  - Kernel pushed to Kaggle via `python scripts/push_to_kaggle.py`.
  - Confirmation of Version 23 creation.

- [ ] **Step 1: Run dry-run packaging check with `python scripts/push_to_kaggle.py --dry-run`**
- [ ] **Step 2: Execute real push with `python scripts/push_to_kaggle.py`**
- [ ] **Step 3: Verify Kaggle CLI response indicates successful push**
- [ ] **Step 4: Commit updated packaged kernel state if applicable**

---

### Task 3: Remote GPU Execution Monitoring & Verification Handshake

**Files:**
- Inspect: `kaggle_runner/output/kazakh-gpu-runner-nb.log`
- CLI: `kaggle kernels status dauletanekesh/kazakh-gpu-runner-nb`
- CLI: `kaggle kernels output dauletanekesh/kazakh-gpu-runner-nb`

**Interfaces:**
- Consumes: Kaggle API status endpoint.
- Produces:
  - Verification of transition: `QUEUED` -> `RUNNING`.
  - Verification that `bitsandbytes>=0.46.1` installation succeeds.
  - Verification that model `Qwen/Qwen2.5-7B-Instruct` is loaded onto GPU and generation begins.

- [ ] **Step 1: Poll `kaggle kernels status dauletanekesh/kazakh-gpu-runner-nb` until status is `RUNNING`**
- [ ] **Step 2: Pull initial execution log to confirm successful model loading without `ImportError`**
- [ ] **Step 3: Verify generation loop starts incrementing counts and flushing to disk**

---

### Task 4: Downstream Harvesting & Verification Benchmark Readiness

**Files:**
- Verify: `scripts/verify_and_filter_claims.py`
- Verify: `scripts/run_retrieval_benchmark.py`
- Verify: `scripts/run_verification_benchmark.py`
- Verify: `papers/kazakh_fever_conference/`

**Interfaces:**
- Consumes: Output datasets `kazakh_fever_3k_generated.jsonl` and `kazakh_social_media_ai_1k.jsonl`.
- Produces:
  - Validated 3,000 FEVER claims (1,000 SUPPORTS, 1,000 REFUTES, 1,000 Hard NEI).
  - Empirical metrics for Recall@1, 3, 5, MRR, strict FEVER score, and 2D Trust Matrix.
  - Final populated conference paper tables.

- [ ] **Step 1: Verify all downstream processing scripts work with dry-run candidate outputs**
- [ ] **Step 2: Run full regression test suite (417+ tests passing)**
- [ ] **Step 3: Document expected execution timeline and harvest commands for the user**
