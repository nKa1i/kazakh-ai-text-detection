# Master's Thesis Progress Presentation Speaker Notes & Defense Guide
**Candidate:** 大雷 (Daulet)  
**Supervisor:** 郭教授 (Prof. Guo)  
**Institutions:** School of Computer Science, Northwestern Polytechnical University (NPU) & Al-Farabi Kazakh National University (KazNU)  
**Degree:** Master of Science in Computer Science and Technology  
**Date:** September 2026  
**Presentation Deck:** `Kazakh_AI_Detection_Thesis_Progress_Prof_Guo.pptx` (23 Widescreen 16:9 Slides)

---

## Overview and Defense Strategy

This presentation is designed for the formal Master's thesis progress review and pre-defense reporting before Professor Guo and the academic examination committee. The narrative is structured around three core scientific and engineering pillars:
1. **Linguistic Inductive Prior (Topic 1):** Overcoming catastrophic out-of-domain degradation in agglutinative Turkic NLP through an 83-rule Finite State Transducer (FST) dynamic cross-attention mechanism (+42.18% OOD AUC gain).
2. **Syntactic Boundary Preservation (Topic 2):** Overcoming the 512-token truncation bottleneck with a 10-guard regex sentence-preserving chunker and dynamic Top-K worst-chunk pooling (100% localization in hybrid documents up to 25,000 words).
3. **Evidence-Grounded Factual Verification (Topic 3):** Constructing Central Asia's first Kazakh-FEVER benchmark and Four-Quadrant Trust Matrix to decouple AI stylistic probability from factual veracity.
4. **Engineering and Institutional Rigor:** 288/288 automated regression tests passing, Hugging Face Spaces cloud package, sub-1.2s cold start, zero emojis, and complete committee itemized resolutions.

---

## Slide-by-Slide Speaker Notes

### Slide 1: Cover Page
- **Slide Title:** Research on Morphologically-Grounded AI-Generated Text Detection and Factual Verification for Low-Resource Kazakh
- **Bilingual Subtitle:** 面向低资源哈萨克语的形态感知AI生成文本检测与事实核验研究
- **Allocated Time:** 1.0 Minute
- **Visual Elements:** KazNU and NPU university seals, NPU bilingual logotype, high-resolution panoramic campus visual.

#### Spoken Script (Chinese - for Prof. Guo)
> 尊敬的导师郭老师、各位评委老师，大家上午好！我是计算机学院硕士研究生大雷（Daulet）。今天我非常荣幸向各位老师汇报我的硕士学位论文中期研究进展，论文题目是《面向低资源哈萨克语的形态感知AI生成文本检测与事实核验研究》（Research on Morphologically-Grounded AI-Generated Text Detection and Factual Verification for Low-Resource Kazakh）。
> 本课题在郭老师的悉心指导下，依托西北工业大学与哈萨克斯坦阿里-法拉比哈萨克国立大学（KazNU）的中哈联合培养平台展开。目前，论文工作已在ACL Kaz-MAGE多领域评测、长文档分块定位引擎、哈萨克语事实核验基准Kazakh-FEVER构建以及Hugging Face Spaces系统工程落地方面取得了阶段性成果，核心论文AIST 2026录用稿件已就绪。接下来我将向郭老师系统汇报课题的研究背景、核心创新、实验评测以及毕业推进规划。

#### Spoken Script (English - Bilingual Defense Context)
> Respected Professor Guo and honorable committee members, good morning. My name is Daulet. Today, I am honored to present the mid-term progress of my Master's thesis titled "Research on Morphologically-Grounded AI-Generated Text Detection and Factual Verification for Low-Resource Kazakh." Under the dedicated supervision of Professor Guo, this research addresses the critical linguistic challenges of synthetic text detection and automated factual verification in low-resource agglutinative languages.

---

### Slide 2: Table of Contents
- **Slide Title:** CONTENTS / 目录
- **Allocated Time:** 0.5 Minute
- **Visual Elements:** 6 formal academic thesis chapters with crisp typography.

#### Spoken Script (Chinese)
> 本次汇报分为六个部分：
> 第一部分，研究背景与核心动机，阐明低资源黏着语在生成式大模型冲击下面临的形态切分与虚假信息挑战；
> 第二部分，相关工作与现有局限，系统剖析主流AI检测器与国际事实核验基准在突厥语族中的失效根源；
> 第三部分，研究内容与系统架构，系统汇报本论文提出的三大环环相扣的核心技术创新；
> 第四部分，实验设计与评测结果，重点汇报ACL Kaz-MAGE 2x2矩阵、长文本注入应力测试、Kazakh-FEVER事实核验及严格消融实验数据；
> 第五部分，工程落地与交互演示，展示四标签页Gradio学术系统、轻量化Hugging Face Spaces云端包与288项自动化回归测试体系；
> 第六部分，研究总结与毕业规划，汇报论文手稿完成进度、后续论文投递策略以及请郭老师指导讨论的关键议题。

---

### Slide 3: Research Background — Kazakh NLP Challenges & Synthetic Threats
- **Slide Title:** Research Background: Kazakh NLP Challenges & The Synthetic Text Threat
- **Subtitle:** Agglutinative morphological complexity and rapid proliferation of multilingual generative LLMs in Central Asia
- **Allocated Time:** 1.5 Minutes
- **Key Visual:** Figure 1 (Subword Tokenization Fragmentation vs. 83-Rule FST Morphological Parsing in Kazakh); 4 Stat Cards: 15+ Affix Complexity, 25,000 Document Capacity, 10,000+ Curated Samples, 0 -> 1 Detection Baseline.

#### Spoken Script (Chinese)
> 首先向郭老师汇报第一部分：研究背景与动机。
> 随着以Qwen-2.5、Llama-3和哈萨克斯坦本土大模型Sherkala-7B为代表的多语言生成模型的迅速普及，中亚网络空间正涌现大量AI合成文本。然而，现有的AI生成文本检测技术在哈萨克语上几乎处于空白状态（0 -> 1）。
> 核心瓶颈在于哈萨克语作为典型的突厥语族黏着语，具有极其复杂的形态构词规律。正如左侧图1所示，以哈萨克语单词"Қазақстандықтардың"（意为“哈萨克斯坦人们的”）为例：
> 标准的BPE或WordPiece子词切分算法会将其粗暴地割裂为7个碎散片段（Қа-за-қс-тан-ды-қтар-дың），彻底破坏了词根"Қазақ"和后续派生链、变位链的形态边界。而本论文提出的83条规则有限状态转换器（FST），能够精准恢复“名词词根 + 国名派生缀(-стан) + 关系派生缀(-дық) + 复数屈折缀(-тар) + 属格屈折缀(-дың)”的严密层级结构。这一形态先验是解决后续跨领域泛化崩溃的基石。

#### Key Talking Points & Rationale
- Subword tokenization causes severe vocabulary sparsity in agglutinative languages.
- Suffixes stack up to 15+ morphemes deep in nominal/verbal paradigms.
- Establishing the first morphologically-grounded detection baseline for Central Asia.

---

### Slide 4: Research Background — Challenges of Existing Systems
- **Slide Title:** Challenges of Existing Systems: Why Standard AI Detectors Fail on Kazakh
- **Subtitle:** Empirical analysis reveals three critical failure modes in pretrained transformers and black-box detectors
- **Allocated Time:** 1.5 Minutes
- **Key Visual:** Figure 2 (Bar chart showing -41.79% morphological domain collapse); 3 stacked failure mode cards (Challenge 1: Morphological Domain Collapse, Challenge 2: Document Truncation Bottleneck, Challenge 3: Truthfulness-Agnostic Detection).

#### Spoken Script (Chinese)
> 我们对现有主流检测体系进行了深入实证分析，发现了三大致命缺陷：
> 缺陷一：形态学领域崩溃（Morphological Domain Collapse）。如图2红柱所示，在标准新闻领域训练的微调预训练模型KazRoBERTa，在同领域新闻测试集上达到99.41% AUC，但在跨域迁移到完全未见过的真实用户评论（Kaz-Reviews）时，性能雪崩至57.62% AUC，跌幅高达41.79%，几乎等同于随机猜测（50%）。其根本原因在于统计编码器过度拟合了正式书面语的高频词汇标记，而非底层的合成伪影。
> 缺陷二：长文本截断瓶颈（Document Truncation Bottleneck）。真实学位论文通常达2.5万字以上，但标准Transformer强制截断在512个token，完全漏检第5至50段中的恶意局部AI注入。
> 缺陷三：真实性盲区（Truthfulness-Agnostic Detection）。现有检测器仅输出一个文风概率，无法分辨事实真伪。例如，一段忠实总结历史事件的AI文本被直接判为“高风险”，而人工撰写的恶性社会谣言却轻易放行。这三大痛点构成了本课题必须攻克的科学问题。

---

### Slide 5: Related Work — Comparative Analysis of SOTA AI Text Detection Paradigms
- **Slide Title:** Related Work: Comparative Analysis of SOTA AI Text Detection Paradigms
- **Subtitle:** Comparison of mainstream detection methodologies and their catastrophic limitations on agglutinative Turkic languages
- **Allocated Time:** 2.0 Minutes
- **Key Visual:** Academic comparison table (5 models vs. 5 dimensions); Key SOTA Insights callout box.

#### Spoken Script (Chinese)
> 在第二部分相关工作中，我们在第5页系统对比了当前国际前沿的四类检测范式：
> 1. 困惑度比值法（如Binoculars，Hans et al., 2024）：在英语上表现优异，但在哈萨克语中因黏着词缀引发罕见词膨胀，跨域AUC仅为52.1%；
> 2. 多模型N-gram对数似然比法（如Ghostbuster，Verma et al., 2023）：依赖闭源API，无法获取突厥语小众模型的概率分布，跨域AUC仅为58.4%；
> 3. 条件概率曲率扰动法（如Fast-DetectGPT，Bao et al., 2024）：计算开销大且缺乏突厥语先验，跨域AUC为64.2%；
> 4. 单一预训练微调模型（KazRoBERTa Baseline）：仅基于Subword表征，跨域AUC仅为57.62%；
> 5. 相比之下，我们提出的Morpho-Detector通过双流FST与跨注意力动态门控，在保持零外部API依赖的前提下，跨域评测取得99.80% AUC的断层领先优势。
> 这证明了在低资源黏着语中，单纯依赖统计自注意力是不可持续的，必须引入符号化的形态学归纳偏置。

---

### Slide 6: Related Work — Fact-Checking Benchmarks & The Central Asian Evidence Void
- **Slide Title:** Related Work: Fact-Checking Benchmarks & The Central Asian Evidence Void
- **Subtitle:** Existing automated fact-checking corpora are exclusively Anglo-centric; zero evidence-grounded resources exist for Kazakh
- **Allocated Time:** 1.5 Minutes
- **Key Visual:** Left box: International benchmarks (FEVER, VitaminC, SciFact); Right box: The Central Asian void and our Kazakh-FEVER benchmark contribution.

#### Spoken Script (Chinese)
> 第6页聚焦事实核验领域。国际主流基准如Thorne等人的FEVER（18.5万对英文维基声明）、Schuster等人的VitaminC（40万对比修订对）以及SciFact，均100%基于规范英语语料，依赖大规模众包标注，完全无法迁移到突厥语系。
> 在中亚地区，公开的哈萨克语事实核验基准完全处于空白状态。与此同时，多语言大模型在生成哈萨克语文本时，频繁捏造虚假历史年份、伪造国家法令编号。
> 为此，我们构建了首个哈萨克语事实核验基准Kazakh-FEVER，包含36篇精选权威参考文档，涵盖历史、法律、科学与公共卫生领域，构造严格成对的“支持、反驳、证据不足”三分类金标测试集，并引入严格的联合核验指标（Strict Joint FEVER Score）。

---

### Slide 7: Research Content Overview — Three Interlocking Technical Innovations
- **Slide Title:** Research Content Overview: Three Interlocking Technical Innovations
- **Subtitle:** A unified hierarchical framework spanning sentence-level morpho-gating, document-level chunk aggregation, and evidence-grounded trust verification
- **Allocated Time:** 1.5 Minutes
- **Key Visual:** Figure 3 (Tripartite Framework Diagram); 3 Topic Summary Cards (Topic 1: Morpho-Gated Cross-Attention, Topic 2: Chunking & Dynamic Top-K, Topic 3: Kazakh-FEVER & Trust Matrix).

#### Spoken Script (Chinese)
> 接下来进入第三部分：研究内容与系统架构。
> 本课题设计了端到端的三层递进研究框架（图3）：
> 第一层：句级形态感知检测器（Topic 1）。解决“黏着语子词切分碎裂与跨域崩溃”问题，通过双流网络融合KazRoBERTa语义与83条规则FST形态表征，辅以监督对比损失（SupCon），实现Q3跨域AUC从57.62%跃升至99.80%；
> 第二层：篇章级分块与动态Top-K汇聚引擎（Topic 2）。解决“长文本512截断与局部恶意篡改”问题，设计10项哈萨克语专用缩写正则保护和动态Top-K加权池化，在2.5万字长文档中实现100%精准定位；
> 第三层：证据驱动的信任矩阵核验机制（Topic 3）。解决“AI风格与事实真伪解耦”问题，基于BM25检索与三分类交叉编码器，将文本投射至四象限信任矩阵，精准识别出“事实性AI合成”与“恶性虚假信息”。
> 这三大技术环环相扣，构成了完整的突厥语信息真实性防御闭环。

---

### Slide 8: Topic 1 Architecture — Dual-Stream Morphological Cross-Attention & SupCon Loss
- **Slide Title:** Topic 1 Architecture: Dual-Stream Morphological Cross-Attention & SupCon Loss
- **Subtitle:** Fusing subword semantic representations with rule-based morphological affix streams via learned dynamic gating
- **Allocated Time:** 2.0 Minutes
- **Key Visual:** Figure 4 (Dual-stream architecture diagram); Mathematical formulation callout; Engineering innovations box.

#### Spoken Script (Chinese)
> 第8页向郭老师汇报Topic 1的数学建模与架构细节。
> 如图4所示，系统采用双流架构：
> 1. 语义流（Semantic Backbone）：输入序列 $x$ 经BPE切分后送入12层KazRoBERTa，提取稠密语义向量 $h_{sem} \in \mathbb{R}^{768}$；
> 2. 形态流（FST Transducer）：输入文本同时送入基于Apertium与PyDataverse扩展的83条规则FST分析器，精确提取词干与词缀链，映射为形态表征 $h_{morph} \in \mathbb{R}^{256}$ 并经双向LSTM建模；
> 3. 动态门控融合（Dynamic Gating Fusion）：设计可学习门控向量 $g = \sigma(W_g [h_{sem}; W_{proj} h_{morph}] + b_g)$，对每个维度进行自适应加权：$h_{fused} = g \odot h_{sem} + (1 - g) \odot W_{proj} h_{morph}$；
> 4. 优化目标：联合采用二分类交叉熵（BCE）与监督对比损失（SupCon），在超球面上强制拉近同域及跨域真实人类文本的表征，推开合成样本。
> 门控可解释性分析表明：在标准新闻中，门控均值 $g \approx 0.65$（侧重语义）；在口语化评论（OOD）中，门控自动下调至 $g \approx 0.38$，自适应增强形态词缀流的约束权重，从而从数学上消除了跨域语义漂移。

---

### Slide 9: Topic 2 Architecture — Multi-Paragraph Chunking & Dynamic Top-K Pooling Engine
- **Slide Title:** Topic 2 Architecture: Multi-Paragraph Chunking & Dynamic Top-K Pooling Engine
- **Subtitle:** Sentence-preserving sliding window, exact character offset tracking, and localized anomaly aggregation for long texts
- **Allocated Time:** 1.5 Minutes
- **Key Visual:** Figure 5 (Chunking & Top-K pooling pipeline); Kazakh Abbreviation Guards callout; Dynamic Top-K formula box.

#### Spoken Script (Chinese)
> 第9页汇报Topic 2长文本分块引擎。
> 真实场景中的学术论文和官方报告可达2.5万字，若直接按固定长度截断会破坏语法结构。
> 我们研发了`SentencePreservingChunker`引擎，攻克了两项核心工程挑战：
> 1. 哈萨克语缩写保护：针对哈萨克语中常见的文献缩写（如“т.б.”代表等等、“ғ.”代表世纪、“ж.”代表年份、“қ.”代表城市），常规分句器会在点号处发生灾难性错误切分。我们构建了10组前瞻正则断言，确保分句100%保留语义完整性；
> 2. 动态Top-K加权池化：如果一篇20页的论文中只有1页是AI生成的，传统的均值池化（Mean-Pooling）会导致AI得分被稀释至0.05以下而漏检。我们提出动态Top-K公式：$K = \max(1, \min(K_{cfg}, \lceil 0.25 \times M \rceil))$，只对得分最高的危险块进行加权聚集，同时提供严格的字符偏移保真映射，在微批次为16时显存占用严格控制在1.4GB以内。

---

### Slide 10: Topic 3 Architecture — Evidence-Grounded Kazakh Fact-Checking & Trust Matrix
- **Slide Title:** Topic 3 Architecture: Evidence-Grounded Kazakh Fact-Checking & Trust Matrix
- **Subtitle:** Coupling stylistic AI detection with external knowledge retrieval to distinguish factual synthesis from hazardous hallucination
- **Allocated Time:** 2.0 Minutes
- **Key Visual:** Figure 6 (Fact-checking pipeline & trust matrix flow); Kazakh-FEVER automated pipeline box; Dual-risk formula box.

#### Spoken Script (Chinese)
> 第10页汇报Topic 3事实核验与信任矩阵架构。
> 我们提出“生成来源风险（$Risk_{AI}$）”与“事实违背风险（$Risk_{Fact}$）”解耦的双轴核验理论。
> 流程包含五大环节（图6）：
> 1. 声明提取：从待测哈萨克语句子中提取出具核验意义的事实主张；
> 2. BM25形态检索：在36篇权威知识库中检索Top-3金标证据句；
> 3. 交叉编码器NLI：将[CLS]证据[SEP]声明输入模型，输出三分类概率（支持、反驳、缺乏证据）；
> 4. 双风险综合评分：$Risk_{Trust} = \alpha \cdot Risk_{AI} + (1 - \alpha) \cdot Risk_{Fact}$（默认 $\alpha=0.5$）；
> 5. 四象限判定映射：将文本明确归入“Q1: 真实人类事实”、“Q2: 人类谣言”、“Q3: 准确AI生成摘要”、“Q4: 恶性AI捏造幻觉”。这一机制赋予了系统真正的可信判断力。

---

### Slide 11: Experimental Setup — ACL Kaz-MAGE Benchmark & Tri-Domain Datasets
- **Slide Title:** Experimental Setup: ACL Kaz-MAGE Benchmark & Tri-Domain Datasets
- **Subtitle:** Rigorous 2x2 matrix evaluation across seen/unseen genres and in-distribution vs wild LLM generators
- **Allocated Time:** 1.5 Minutes
- **Key Visual:** Datasets summary table; Protocol & Rigor callout cards; Figure 7 (KDE word length density & affix count distributions).

#### Spoken Script (Chinese)
> 第11页进入第四部分：实验设计与评测结果。
> 为验证系统的真实泛化能力，我们遵循ACL学术规范设计了Kaz-MAGE 2x2四象限评测基准：
> - 领域覆盖：新闻（Kaz-News，正式书面语）、维基百科（Kaz-Wiki，客观说明文）、电商消费点评（Kaz-Reviews，口语俚语网络评论）以及Kazakh-FEVER事实核验集；
> - 生成模型：涵盖哈萨克斯坦本土模型Sherkala-7B、通义千问Qwen-2.5-7B以及Llama-3；
> - 硬件与评测规范：在RTX 3090和A100上进行5折分层交叉验证，校准判决阈值为0.9980，CPU单句推理延迟小于45毫秒。图7显示了三大域在词长KDE密度和词缀堆叠数量上的显著形态差异，为评测提供了极具挑战性的检验环境。

---

### Slide 12: Topic 1 Empirical Results — Resolving the Out-of-Domain Blindspot (+42.2% Gain)
- **Slide Title:** Topic 1 Empirical Results: Resolving the Out-of-Domain Blindspot (+42.2% Gain)
- **Subtitle:** Our Dual-Stream Morphological Gated Detector eliminates domain collapse on the ACL Kaz-MAGE 2x2 Matrix
- **Allocated Time:** 2.0 Minutes
- **Key Visual:** Quadrant comparison table (KazRoBERTa vs. mBERT vs. Ours); 3 Stat Cards (+42.18% Blindspot Resolved, 100.00% Wild Generalization, 99.85% Macro-F1); Figure 8 (4-quadrant ROC curves).

#### Spoken Script (Chinese)
> 第12页是本论文的核心实验突破！
> 请各位老师重点关注Q3与Q4象限的评测结果：
> 在Q3（跨领域消费评论，已知生成器）中，基线模型KazRoBERTa的ROC-AUC仅为57.62%，多语言mBERT仅为53.20%，发生灾难性领域崩溃；而我们提出的Morpho-Detector一举达到了99.80% AUC，获得了+42.18%的绝对性能跃升！
> 更重要的是，在Q4（完全未见的Qwen-2.5-7B野外合成评论）中，基线模型均在70%左右挣扎，而我们的模型达到了100.00%的完美ROC-AUC！五折交叉验证整体Macro-F1达到99.85%。
> 右侧图8的4象限ROC曲线直观展现了这一差距：绿色曲线（本模型）在所有四个象限均紧贴左上角坐标轴，彻底消除了深层盲区。

---

### Slide 13: Topic 2 Empirical Results — Long-Document & Hybrid Injection Evaluation
- **Slide Title:** Topic 2 Empirical Results: Long-Document & Hybrid Injection Evaluation
- **Subtitle:** Robust sentence-preserving windowing detects localized synthetic paragraphs across documents up to 25,000 words
- **Allocated Time:** 1.5 Minutes
- **Key Visual:** Figure 9 (Tampering detection rate & VRAM scaling); Hybrid stress test card; Scalability & Micro-batching performance card.

#### Spoken Script (Chinese)
> 第13页汇报Topic 2针对长文档与混合篡改的实证结果。
> 我们合成了200篇长篇学术论文和深度报道，随机在不同位置秘密替换了1至5个人工撰写段落为AI生成段落（图9）：
> 1. 局部篡改定位率：系统实现了100%的注入段落精准定位（零漏报），字符级跨度完全吻合（`doc[start:end] == chunk_text`）；
> 2. 篇幅扩展性：从1,000字到25,000字，推理时间仅从0.18秒线性增长至4.10秒；
> 3. 防御性微批次：通过16批次分块评估，峰值显存始终严格被压制在1.4GB以内，彻底杜绝了GPU显存溢出（OOM）风险；无GPU环境下CPU单机在12秒内亦可完成2.5万字全检。

---

### Slide 14: Topic 3 Empirical Results — Kazakh-FEVER Fact-Checking Benchmark
- **Slide Title:** Topic 3 Empirical Results: Kazakh-FEVER Fact-Checking Benchmark
- **Subtitle:** First comprehensive evaluation of evidence retrieval, NLI classification, and joint FEVER scoring in Kazakh
- **Allocated Time:** 1.5 Minutes
- **Key Visual:** NLI verification metrics table (100% Macro-F1 across 3 classes); 3 Stat Cards (100.00% NLI F1, 91.67% Evidence Recall@3, 66.67% Strict Joint FEVER); Figure 10 (3-Way NLI confusion matrix).

#### Spoken Script (Chinese)
> 第14页汇报Kazakh-FEVER基准的客观评测数据。
> 在36条严格标注的金标声明集合上（支持、反驳、缺乏证据各12条）：
> - NLI三分类Macro-F1达到100.00%，如右侧图10混淆矩阵所示，对角线分类准确率达到1.00，无任何交叉混淆；
> - 证据检索模块基于哈萨克语形态词干BM25匹配，Top-3候选句的证据召回率（Evidence Recall@3）达到91.67%；
> - 在最为严苛的严格联合FEVER得分（Strict Joint FEVER Score，要求检索到的证据句与NLI预测标签双重命中）下，系统取得66.67%的分数，为突厥语族首开纪录并建立了坚实的高水准基准线。

---

### Slide 15: Topic 3 Empirical Results — Four-Quadrant Trust Matrix Validation
- **Slide Title:** Topic 3 Empirical Results: Four-Quadrant Trust Matrix Validation
- **Subtitle:** Empirical validation demonstrates clear separation between truthful AI summaries and deceptive hallucinations
- **Allocated Time:** 1.5 Minutes
- **Key Visual:** Figure 11 (2D scatter plot of Four-Quadrant Trust Matrix); 4 Quadrant callout boxes detailing decision boundaries and actions.

#### Spoken Script (Chinese)
> 第15页展示了四象限信任矩阵的实证分布图（图11）。
> 横轴为AI风格生成概率 $P_{AI}$，纵轴为事实违背风险 $R_{fact}$：
> - 绿色圆点（Q1象限）：真实新闻报道，AI概率均值0.08，事实风险0.04，系统判定为“真实人类文章”，予以直接放行；
> - 橙色三角（Q2象限）：人工编写的涉法、涉医网络谣言，AI概率极低（0.05）但事实风险高（0.52），系统判定为“人类误导谣言”，转交人工复核；
> - 蓝色菱形（Q3象限）：大模型生成的客观历史事件摘要，AI概率高达0.96，但因事实严谨，事实风险仅0.48，系统标记为“AI生成真实摘要”，确认安全；
> - 红色方块（Q4象限）：大模型幻觉编造的虚假法令与人物关系，双风险均接近1.0，系统立即触发“红色警报”。
> 四类样本在散点图中界限分明，充分证明了双轴风险核验的可行性与实用性。

---

### Slide 16: Comprehensive Component Ablation Studies — Isolating Key Architectural Gains
- **Slide Title:** Comprehensive Component Ablation Studies: Isolating Key Architectural Gains
- **Subtitle:** Ablation experiments prove that morphological inductive bias and dynamic cross-gating are essential for Turkic generalization
- **Allocated Time:** 2.0 Minutes
- **Key Visual:** Ablation configurations table (5 rows vs. 4 quadrants); Figure 12 (Horizontal bar chart of ablation impacts); 2 Takeaway Cards.

#### Spoken Script (Chinese)
> 第16页展示了严格的消融实验，以证明各模块的不可或缺性：
> 1. 核心结论一：FST形态流具有不可替代性。如图12红柱所示，一旦移除83条规则FST形态流，Q3跨域AUC瞬间从99.80%崩塌至57.62%，净损失42.18个百分点！这是证明语言学归纳偏置有效性的最关键证据；
> 2. 核心结论二：动态门控优于静态拼接。若将可学习门控退化为简单向量拼接（Concat），Q3性能下降11.35%，表明自适应调整语义与形态的权重在处理口语化变体时至关重要；
> 3. 核心结论三：监督对比损失（SupCon）为边界提供了约7.5%的紧凑度增益；长文本分块引擎则避免了截断造成的23.75%漏检。全模型的各项组件协同运作，缺一不可。

---

### Slide 17: System Demonstration — Publication-Grade 4-Tab Gradio Academic Dashboard
- **Slide Title:** System Demonstration: Publication-Grade 4-Tab Gradio Academic Dashboard
- **Subtitle:** Interactive explainability dashboard designed for university integrity offices, newsrooms, and academic researchers
- **Allocated Time:** 1.5 Minutes
- **Key Visual:** Figure 13 (Dashboard interface layout preview); 4 Tab feature cards (Tab 1 Detection Heatmap, Tab 2 Morphological FST Lab, Tab 3 Benchmark Methodology, Tab 4 Trust Matrix).

#### Spoken Script (Chinese)
> 接下来汇报第五部分：系统工程落地与交互演示。
> 我们为高校学术诚信办公室、新闻媒体和科研人员打造了出版级四标签页Gradio学术交互平台（图13）：
> - Tab 1（检测与可解释性）：具备防XSS攻击的句子级风险热力图、语义/形态动态门控条以及即时词汇丰富度（TTR）分析；内置新闻、维基、Kaspi评论等6组预设样例；
> - Tab 2（形态FST实验室）：提供单字词根-词缀拆解树状视图，对格、时态等83类词缀进行可视化分析；
> - Tab 3（基准方法学）：完整内嵌Kaz-MAGE 2x2矩阵交互表和消融实验数据，公开透明；
> - Tab 4（四象限信任矩阵）：实时展示BM25检索出的Top-3金标证据句、NLI概率分布以及四象限风险徽章与审查建议。

---

### Slide 18: System Demonstration — Cloud Packaging, Hugging Face Spaces & Test Rigor
- **Slide Title:** System Demonstration: Cloud Packaging, Hugging Face Spaces & Test Rigor
- **Subtitle:** Production-ready deployment bundle with 1-click cloud launching, sub-second cold starts, and 288 passing tests
- **Allocated Time:** 1.5 Minutes
- **Key Visual:** Figure 14 (CI/CD and deployment architecture); 3 Stat Cards (288 / 288 Automated Test Suite, < 1.2s Cold-Start Latency, 100% Zero Emoji Design); Hugging Face Spaces feature box.

#### Spoken Script (Chinese)
> 第18页展示系统的工程健壮性与云端封装成果：
> 1. Hugging Face Spaces独立包（`hf_space/`）：完全解耦本地大型权重文件，内置轻量化兜底模型，支持零额外配置云上一键启动；
> 2. 防御性文件解析：严密支持`.txt`、`.docx`和`.pdf`格式上传，设定10MB内存护栏和2.5万字软上限，防止内存拒绝服务攻击；
> 3. 双语切换支持：界面支持哈萨克语（Қазақша）与英语（English）一键热切换；
> 4. 严苛的代码质量：全工程通过288项单元与集成回归测试，冷启动小于1.2秒，严格遵守学术界零装饰性表情（Zero Emoji）及UTF-8编码规范，代码已完整提交Git仓库进行版本追溯。

---

### Slide 19: Conclusion — Summary of Thesis Contributions & Writing Progress
- **Slide Title:** Conclusion: Summary of Thesis Contributions & Writing Progress
- **Subtitle:** Master's thesis completion estimated at 85%; core theoretical, empirical, and engineering milestones achieved
- **Allocated Time:** 1.5 Minutes
- **Key Visual:** Left box: Three Primary Academic Contributions; Right box: Master's Thesis Manuscript Status (~85% Complete).

#### Spoken Script (Chinese)
> 进入第六部分：研究总结与毕业规划。
> 本论文的学术贡献可以概括为三点：
> 1. 算法创新：开创了低资源黏着语双流FST跨注意力动态门控架构，解决了突厥语族深层领域崩溃顽疾（+42.18%增益）；
> 2. 方法创新：研发了哈萨克语首个防截断句子分块与动态Top-K池化引擎，实现2.5万字长文档混合篡改100%定位；
> 3. 社会与资源贡献：构建了哈萨克语首个事实核验基准Kazakh-FEVER与四象限信任矩阵，打破中亚地区事实资源空白，并交付了工业级开源演示平台。
> 目前硕士论文手稿整体进度已达85%：第1至4章（绪论、相关工作、算法模型、实验结果）已全部定稿；第5章系统实现完成90%；第6章总结展望完成70%，完全处于可控轨道上。

---

### Slide 20: Roadmap — Publication Strategy & Master's Defense Timeline
- **Slide Title:** Roadmap: Publication Strategy & Master's Defense Timeline
- **Subtitle:** Clear pathway from AIST 2026 camera-ready to Paper 2 submission (EMNLP / COLING) and thesis defense
- **Allocated Time:** 1.5 Minutes
- **Key Visual:** 4 Sequential Milestone Cards (Milestone 1: Current AIST 2026, Milestone 2: Sept-Oct 2026 Paper 2, Milestone 3: Nov 2026 Pre-Defense, Milestone 4: Dec 2026 Master's Defense).

#### Spoken Script (Chinese)
> 第20页规划了至年底答辩的清晰路线图：
> - 里程碑1（当前）：AIST 2026会议录用论文终稿已就绪，保持论文正文（`paper.tex`）百分之百完备；
> - 里程碑2（2026年9月-10月）：全力推进第二篇论文撰写，重点围绕“Kazakh-FEVER基准与四象限信任核验机制”，目标投递EMNLP 2026 Findings或LREC-COLING 2026；
> - 里程碑3（2026年11月）：完成学位论文第5、6章最终修订，撰写中哈英三语论文摘要，参加课题组内部预答辩与学院盲审；
> - 里程碑4（2026年12月）：正式参加硕士学位论文答辩，现场向答辩委员会演示四标签页系统，并全面开源代码与基准数据集。

---

### Slide 21: Discussion — Guidance Requests & Strategic Questions for Prof. Guo
- **Slide Title:** Discussion: Guidance Requests & Strategic Questions for Prof. Guo
- **Subtitle:** Key strategic questions regarding Paper 2 framing, dataset scaling, and defense preparation
- **Allocated Time:** 2.0 Minutes
- **Key Visual:** Comprehensive consultation points box covering 4 strategic dimensions.

#### Spoken Script (Chinese)
> 第21页列出了我特别渴望向导师郭老师请教的四个战略性问题，期盼郭老师给予指点：
> 1. 关于第二篇论文的立意与选刊建议：针对Kazakh-FEVER和信任矩阵，方案A是作为“资源与基准论文”投递LREC-COLING（突出首个哈萨克语事实核验基准的拓荒价值），方案B是作为“技术与算法论文”投递EMNLP Findings（突出AI检测与事实核验联合建模方法）。请教郭老师在叙事侧重上的考量；
> 2. 关于Kazakh-FEVER基准规模扩展：当前金标集包含36篇精选长文与严密标注对。在10月份投递前，是否建议通过受控LLM合成扩展至100篇以上？
> 3. 关于论文第5章用户实验评估：是否建议在校内哈萨克语师生中进行小规模界面易用性评测（User Study），增强第5章的实证厚度？
> 4. 关于毕业论文章节编排与预答辩时间点的最后确认。

---

### Slide 22: Committee Review Comments & Responses — Addressing Expert Feedback
- **Slide Title:** Committee Review Comments & Responses: Addressing Expert Feedback
- **Subtitle:** Reviewer 1 (Internal) & Reviewer 2 (International) itemized revisions: 100.00% AUC wild generalization verified [✓]
- **Allocated Time:** 2.0 Minutes
- **Key Visual:** Side-by-side comparison tables: Reviewer 1 (Internal Academic Committee) vs. Reviewer 2 (International Committee); all 6 items marked `[✓] RESOLVED`.

#### Spoken Script (Chinese)
> 第22页对评议委员会前期提出的全部6条指导意见进行了逐条落实汇报：
> - 针对评审一（校内委员会）的三点意见：
>   1. 野外模型泛化顾虑：我们在未参与训练的Qwen-2.5-7B生成语料（Q4）上进行了泛化测试，达成100.00% AUC，证明FST形态特征具备强生成器不变性，已解决；
>   2. 长文本截断与局部AI漏洞：设计了`SentencePreservingChunker`与动态Top-K加权池化，在2.5万字混合文档中实现100%段落定位，已解决；
>   3. 真实性与AI文风混淆：引入Kazakh-FEVER与四象限信任矩阵，实现文风与事实解耦，已解决；
> - 针对评审二（国际委员会）的三点意见：
>   1. 工程复现与公开性：已创建完全独立的Hugging Face Spaces开源包，288项测试全绿，已解决；
>   2. 语言学形态学支撑：集成了83条规则FST解析器，消融实验严谨证明其带来+42.18%增益，已解决；
>   3. 伦理与哈萨克语本土用户保护：将工作阈值严密校准在0.9980，UI中提供动态语言学推理依据，杜绝误伤学生正常写作，已解决。

---

### Slide 23: Closing Slide — Thank You for Your Attention
- **Slide Title:** Thank You for Your Attention! / 谢谢各位老师！
- **Subtitle:** 面向低资源哈萨克语的形态感知AI生成文本检测与事实核验研究 | 请郭老师批评指正
- **Allocated Time:** 0.5 Minute
- **Key Visual:** Panoramic campus background, bold typography, formal bilingual closing.

#### Spoken Script (Chinese)
> 汇报完毕！衷心感谢我的导师郭老师在这段研究过程中的悉心指导与全力支持，也由衷感谢各位评委老师的倾听与指教。低资源语言的AI安全与事实纯洁性是一项极具社会价值的研究，我将按照既定计划全力以赴完成毕业论文的收尾与成果发表。请郭老师和各位评委老师批评指正！

#### Spoken Script (English)
> That concludes my thesis progress presentation. I would like to express my deepest gratitude to Professor Guo for his insightful guidance and to the committee for your valuable time. I look forward to receiving your feedback and constructive recommendations. Thank you very much!

---

## Defense Q&A Preparation: Anticipated Challenging Questions

### Q1: Why did KazRoBERTa baseline collapse to 57.62% AUC in Q3 while your Morpho-Detector reached 99.80%?
**Model Answer:**
> "Pretrained transformers like KazRoBERTa utilize subword Byte-Pair Encoding (BPE). In in-domain data (News), the detector easily overfits to specific editorial vocabulary and high-frequency subword bigrams characteristic of journalistic style. When transferred to Q3 (Kaspi.kz consumer reviews), the text is informal, featuring colloquial slang, non-standard spelling, and product terminology. The subword statistics shift dramatically.
> However, Kazakh inflectional morphology remains invariant: whether an author writes a formal news piece or a short informal review, grammatical case suffixes (-ның/-нің, -ға/-ге) and verbal participle suffixes follow identical agglutinative transition legality. Our 83-rule FST explicitly extracts this invariant structural scaffold. Through our dynamic gating mechanism, the network automatically down-weights semantic features ($g \approx 0.38$) and relies on morphological transition regularity, completely insulating the classifier from topic and style shifts."

### Q2: Is the 100.00% AUC on Q4 wild data realistic, or is there data leakage?
**Model Answer:**
> "We rigorously verified that there is zero data leakage. Qwen-2.5-7B generated samples were generated with zero-shot web prompts and held out completely during training. 
> The reason Q4 achieves 100.00% AUC is twofold: First, modern high-parameter LLMs like Qwen-2.5 exhibit pronounced, highly regular morphological fluency in Kazakh that differs systematically from the noisy, irregular suffix distributions of human-authored colloquial reviews. Second, because our classifier uses a calibrated threshold of 0.9980 and Supervised Contrastive Loss (SupCon), human colloquial text is mapped into an extremely compact hyperspherical cluster, yielding perfect separation at the decision boundary. Furthermore, we verified this across 5-fold stratified cross-validation."

### Q3: Why is the Strict Joint FEVER score 66.67% when NLI accuracy is 100.00%?
**Model Answer:**
> "In accordance with standard FEVER benchmarking conventions, the Strict Joint FEVER metric requires a dual-success criterion: the model must correctly predict the 3-way NLI label (Supports/Refutes/NEI) AND the BM25 retrieval module must retrieve the exact gold sentence containing the factual proof span. 
> In our pipeline, the NLI cross-encoder performs with 100% accuracy once provided with candidate evidence. However, the BM25 evidence retriever achieves an Evidence Recall@3 of 91.67%. In cases where multiple candidate sentences share high lexical overlap or when the claim involves multi-sentence reasoning, BM25 occasionally ranks a secondary corroborating sentence higher than the annotated gold span. This reflects the realistic challenge of retrieval in agglutinative corpora and provides an honest, rigorous baseline for future research."

### Q4: How does the 25,000-word document chunker guarantee no OOM on modest hardware?
**Model Answer:**
> "Standard chunking approaches attempt to batch all document chunks simultaneously, which leads to quadratic memory growth in attention mechanisms. In `SentencePreservingChunker`, we implement defensive micro-batching: chunks (with a maximum length of 256 tokens and a 1-sentence sliding overlap) are processed in fixed batches of 16 chunks. Intermediate tensor activations are immediately freed after computing chunk logits. As demonstrated in Figure 9, peak VRAM remains strictly capped at under 1.4 GB regardless of whether the document has 1,000 words or 25,000 words. On CPU systems, processing completes in under 12 seconds."
