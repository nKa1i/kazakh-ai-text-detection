# Master's Thesis Proposal Defense Presentation (5-Element Structure) Design Spec

- **Date**: 2026-10-26
- **Event**: Master's Dissertation Title Selection Report Defense (西北工业大学硕士学位论文选题报告答辩)
- **Defense Date**: 2026年10月29日 (October 29, 2026)
- **Candidate**: 大雷 (DAULET ANEKESH, Student No.: `2025291043`)
- **Supervisor**: Prof. Bin Guo (郭斌 教授)
- **Institution**: School of Computer Science, Northwestern Polytechnical University (NPU / 西北工业大学计算机学院) & Al-Farabi Kazakh National University (KazNU)
- **Degree**: Master of Science in Computer Science and Technology (计算机科学与技术)
- **Template Baseline**: `C:\Users\Roza\Downloads\Aliya PPT.pptx` (16:9 widescreen format, authentic institutional branding)

---

## 1. Architectural Overview & Design Decisions

### 1.1 Consolidation to 5 Elements (Matching Professor Guo's Lab Standard)
Following reference materials from Professor Guo's lab (reference proposal presentation from fellow lab student), the slide deck is organized strictly into **5 core elements**:
1. **研究背景 (Research Background & Significance)**
2. **相关工作 (Related Work & SOTA Gaps)**
3. **研究内容 (Research Content & System Architecture)**
4. **研究创新 (Research Innovations & Technical Challenges)**
5. **进展安排 (Progress, Preliminary Results, Working Conditions & Timeline)**

### 1.2 Running Section Tracker Bar
- The top navigation tracker bar across content slides is updated from 6 items to 5 items:
  `研究背景` | `相关工作` | `研究内容` | `研究创新` | `进度安排`
- The active section has a high-contrast Navy pill (`#1E3A8A`) with white text, while inactive sections are muted slate.

### 1.3 Timeline Visualization (Slide 19, matching Screenshot 2)
- Header: `2026年10月-2027年6月` (Bold Navy with vertical accent bar)
- A directional horizontal arrow pointing right
- 5 numbered circles (alternating red `#C00000` and blue `#2E75B6`):
  - **Circle 1 (Red, Below Axis)**: `2026.10 - 2026.11`
    - 开题报告答辩与修改完善
    - 确立论文研究方案与技术路线细则
  - **Circle 2 (Blue, Above Axis)**: `2026.11 - 2027.01`
    - 长文档滑动窗口检测与跨领域优化
    - 完成4-Tab Gradio云端原型平台全面部署与内测
  - **Circle 3 (Red, Below Axis)**: `2027.02 - 2027.03`
    - 会议论文录用修改与补充消融实验
    - 扩展Kazakh-FEVER事实核验基准评测
  - **Circle 4 (Blue, Above Axis)**: `2027.04 - 2027.05`
    - 撰写硕士学位论文各章节初稿与导师审阅
    - 参加学院组织的硕士学位论文预答辩
  - **Circle 5 (Red, Below Axis)**: `2027.05 - 2027.06`
    - 论文双盲评审、盲审意见修改与定稿
    - 硕士学位论文正式答辩与办理毕业手续

---

## 2. Complete Slide Mapping (21 Slides)

1. **Slide 1: 封面 (Cover Page)**:
   - Full bilingual title: *Morphologically-Grounded AI-Generated Text Detection and Factual Verification for Low-Resource Agglutinative Languages: A Comprehensive Framework for Kazakh* / 基于形态学感知的低资源黏着语AI生成文本检测与事实核查研究
   - Subtitle: 硕士学位论文选题报告答辩 / Master's Thesis Proposal Defense
   - Metadata: 汇报人：大雷 (DAULET ANEKESH, 学号 2025291043) | 导师：郭斌 教授 | 答辩日期：2026年10月29日 | 西北工业大学计算机学院
2. **Slide 2: 目录 (Contents)**:
   - 5 structured academic chapters with bilingual subtitles.
3. **Slide 3: 01 研究背景 - 语言学特性与低资源AI生成文本威胁**:
   - `fig01_subword_vs_fst.png` (Subword fragmentation vs 83-rule FST); 4 stat cards.
4. **Slide 4: 01 研究背景 - 现有检测系统的三大致命病态分析**:
   - `fig02_domain_collapse.png` (-41.79% OOD domain collapse); 3 failure mode cards.
5. **Slide 5: 02 相关工作 - SOTA检测范式对比与低资源黏着语局限**:
   - Academic comparison table (5 paradigms $\times$ 5 dimensions).
6. **Slide 6: 02 相关工作 - 自动化事实核查与“捷径学习”缺陷**:
   - FEVER pipeline (Retrieval $\to$ NLI); 2-char negation suffix reversal; Hard NEI protocol.
7. **Slide 7: 03 研究内容 - 总体研究框架与三层神经-符号协同架构**:
   - `fig03_tripartite_framework.png` (Tripartite system architecture).
8. **Slide 8: 03 研究内容 - 研究内容一：形态感知的双流门控融合机制**:
   - `fig04_morpho_gate_arch.png` (Kaz-RoBERTa + BiLSTM + dynamic gating $\mathbf{g}$ + $\mathcal{L}_{\text{SupCon}}$).
9. **Slide 9: 03 研究内容 - 研究内容二：缩写保护的滑动窗口长文档检测**:
   - `fig05_chunking_topk_flow.png` (Sliding window + Top-$K$ worst-chunk pooling).
10. **Slide 10: 03 研究内容 - 研究内容三：证据驱动的事实核查与四象限信任空间**:
    - `fig06_trust_matrix_pipeline.png` (Hybrid BM25/M3 retrieval + 3-way NLI + $(T_{\text{fact}}, T_{\text{gen}})$ trust plane).
11. **Slide 11: 04 研究创新 - 问题创新与方法创新 (两大核心维度)**:
    - Problem Innovation & Methodological Innovation cards matching Chinese reference slide 21.
12. **Slide 12: 04 研究创新 - 核心技术难点与应对解决方案**:
    - Comparison table mapping morphological vocabulary explosion, generator overfitting, and retrieval shortcut learning to mathematical solutions.
13. **Slide 13: 05 进展安排 - 数据集构建：Kaz-MAGE与Kazakh-FEVER 3K**:
    - `fig07_dataset_distribution.png`; 10k samples, 3k FEVER claims (500 Hard NEI).
14. **Slide 14: 05 进展安排 - 阶段性成果一：双流检测模型评测与泛化性能**:
    - `fig08_roc_curves_kaz_mage.png`; Macro ROC-AUC 0.9980, Kaspi OOD 99.80%, Qwen-2.5-7B zero-shot 100%.
15. **Slide 15: 05 进展安排 - 阶段性成果二：长文档篡改定位与显存开销**:
    - `fig09_long_doc_injection.png`; 100% tamper localization; $\mathcal{O}(L_{\text{chunk}})$ linear memory scaling; latency < 85ms.
16. **Slide 16: 05 进展安排 - 阶段性成果三：事实核验矩阵与消融验证**:
    - `fig10_kaz_fever_confusion_matrix.png` & `fig12_ablation_study_barchart.png`.
17. **Slide 17: 05 进展安排 - 阶段性成果四：学术发表与云端系统原型**:
    - `fig13_gradio_dashboard_panels.png`; **Paper 1 Accepted (AIST 2026, Springer LNCS)**; **Paper 2 Submitted (COLING 2027 ARR Track)**; Hugging Face Spaces deployment.
18. **Slide 18: 05 进展安排 - 工作积累与研究条件落实情况**:
    - Matching reference slide 22: Publication portfolio + GPU cluster (A100 80GB, dual Tesla T4s) + KazNU collaboration.
19. **Slide 19: 05 进展安排 - 进度安排时间轴 (2026.10 – 2027.06)**:
    - High-impact timeline figure matching Screenshot 2 with 5 alternating red/blue circles across Oct 2026 – June 2027.
20. **Slide 20: 05 进展安排 - 预期难点与毕业答辩准备**:
    - 4 risk mitigation cards (dialectal variance, dense retrieval latency, native student protections, pre-defense).
21. **Slide 21: 结语致谢 - THANKS! / 敬请各位专家批评指正！**:
    - Institutional closing slide matching reference slide 24 with metadata for October 29, 2026.

---

## 3. Deliverables

- Figure: `presentation_figures/fig15_timeline_2026_2027.png` (300 DPI, matplotlib).
- Builder Script: `scripts/build_proposal_defense_presentation.py`.
- Automated Test Suite: `tests/test_proposal_defense_presentation.py`.
- Primary Output: `Kazakh_AI_Detection_Thesis_Proposal_Defense_Oct29.pptx`.
- Desktop Mirror: `C:\Users\Roza\Desktop\Kazakh_AI_Detection_Thesis_Proposal_Defense_Oct29.pptx`.
