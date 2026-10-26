# Proposal Defense Presentation (5-Element Structure) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify the complete 21-slide Master's Thesis Proposal Defense presentation (`Kazakh_AI_Detection_Thesis_Proposal_Defense_Oct29.pptx`) for candidate Daulet Anekesh (大雷) under supervisor Prof. Bin Guo (郭斌教授), formatted in the approved 5-element laboratory structure with authentic institutional branding, 15 publication figures, and the horizontal timeline diagram (2026.10–2027.06).

**Architecture:** A Python automation pipeline: (1) `scripts/generate_timeline_figure.py` generates `fig15_timeline_2026_2027.png` matching Screenshot 2; (2) `scripts/build_proposal_defense_presentation.py` directly clones `C:\Users\Roza\Downloads\Aliya PPT.pptx`, strips commercial watermarks, preserves native branding, configures the 5-element running tracker bar, builds all 21 slides, and mirrors to `C:\Users\Roza\Desktop\Kazakh_AI_Detection_Thesis_Proposal_Defense_Oct29.pptx`; (3) `tests/test_proposal_defense_presentation.py` validates slide count, headers, figure paths, and tables.

**Tech Stack:** Python 3.12, `python-pptx`, `matplotlib` (Agg backend), `unittest`.

## Global Constraints

- Strictly zero decorative emojis across all slides, code, tables, and documentation.
- Widescreen 16:9 format ($13.333 \times 7.500$ inches) strictly preserved.
- Native branding preserved: KazNU logo, NPU logo group, watermark, and top-left NPU pentagon badge.
- Running top tracker bar configured for 5 sections: `研究背景`, `相关工作`, `研究内容`, `研究创新`, `进度安排`.
- Timeline diagram must cover `2026年10月-2027年6月` with 5 alternating red/blue circles along a horizontal arrow matching Screenshot 2.
- All Python file operations must explicitly specify `encoding='utf-8'`.
- Mirror output directly to `C:\Users\Roza\Desktop\Kazakh_AI_Detection_Thesis_Proposal_Defense_Oct29.pptx`.

---

### Task 1: Generate High-Resolution Timeline Figure (`fig15_timeline_2026_2027.png`)

**Files:**
- Create: `scripts/generate_timeline_figure.py`
- Output: `presentation_figures/fig15_timeline_2026_2027.png`

**Interfaces:**
- Produces: `presentation_figures/fig15_timeline_2026_2027.png` at 300 DPI, dimensions approx. 12 $\times$ 4.5 inches.

- [ ] **Step 1: Write `scripts/generate_timeline_figure.py`**
  Renders horizontal arrow with 5 alternating circles:
  1: Red (Below) `2026.10-2026.11` 开题报告答辩与修改完善
  2: Blue (Above) `2026.11-2027.01` 长文档滑动窗口检测与跨领域优化
  3: Red (Below) `2027.02-2027.03` 会议论文录用修改与补充消融实验
  4: Blue (Above) `2027.04-2027.05` 撰写硕士学位论文各章节初稿与预答辩
  5: Red (Below) `2027.05-2027.06` 论文盲审修改、硕士学位论文正式答辩及毕业

- [ ] **Step 2: Run script to generate the figure**
  Run: `python scripts/generate_timeline_figure.py`
  Verify file exists and is > 20 KB.

---

### Task 2: Automated Regression Test Suite

**Files:**
- Create: `tests/test_proposal_defense_presentation.py`

**Interfaces:**
- Consumes: `Kazakh_AI_Detection_Thesis_Proposal_Defense_Oct29.pptx`
- Asserts: 21 slides, 5 sections, slide titles, embedded figures, desktop copy exists.

- [ ] **Step 1: Write `tests/test_proposal_defense_presentation.py`**
- [ ] **Step 2: Run test to verify initial failure (presentation not yet built)**
  Run: `python -m unittest tests/test_proposal_defense_presentation.py`

---

### Task 3: Implement Builder Script (`scripts/build_proposal_defense_presentation.py`)

**Files:**
- Create: `scripts/build_proposal_defense_presentation.py`

**Interfaces:**
- Clones: `C:\Users\Roza\Downloads\Aliya PPT.pptx`
- Generates: `Kazakh_AI_Detection_Thesis_Proposal_Defense_Oct29.pptx`
- Mirrors: `C:\Users\Roza\Desktop\Kazakh_AI_Detection_Thesis_Proposal_Defense_Oct29.pptx`

- [ ] **Step 1: Implement template cloning, watermark stripping, and 5-element tracker repositioning**
- [ ] **Step 2: Implement slide populator functions for all 21 slides**
- [ ] **Step 3: Execute builder script**
  Run: `python scripts/build_proposal_defense_presentation.py`

---

### Task 4: Verification and Desktop Delivery

- [ ] **Step 1: Run automated regression tests**
  Run: `python -m unittest tests/test_proposal_defense_presentation.py`
- [ ] **Step 2: Verify desktop file existence and size**
- [ ] **Step 3: Git commit all changes**
