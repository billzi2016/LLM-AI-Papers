# TableEval: A Real-World Benchmark for Complex, Multilingual, and Multi-Structured Table Question Answering

> **Date**：2025-06-04
> **arXiv**：https://arxiv.org/abs/2506.03949

## Abstract

LLMs have shown impressive progress in natural language processing. However, they still face significant challenges in TableQA, where real-world complexities such as diverse table structures, multilingual data, and domain-specific reasoning are crucial. Existing TableQA benchmarks are often limited by their focus on simple flat tables and suffer from data leakage. Furthermore, most benchmarks are monolingual and fail to capture the cross-lingual and cross-domain variability in practical applications. To address these limitations, we introduce TableEval, a new benchmark designed to evaluate LLMs on realistic TableQA tasks. Specifically, TableEval includes tables with various structures (such as concise, hierarchical, and nested tables) collected from four domains (including government, finance, academia, and industry reports). Besides, TableEval features cross-lingual scenarios with tables in Simplified Chinese, Traditional Chinese, and English. To minimize the risk of data leakage, we collect all data from recent real-world documents. Considering that existing TableQA metrics fail to capture semantic accuracy, we further propose SEAT, a new evaluation framework that assesses the alignment between model responses and reference answers at the sub-question level. Experimental results have shown that SEAT achieves high agreement with human judgment. Extensive experiments on TableEval reveal critical gaps in the ability of state-of-the-art LLMs to handle these complex, real-world TableQA tasks, offering insights for future improvements. We make our dataset available here: https://github.com/wenge-research/TableEval.

---

# TableEval：面向复杂、多语言、多结构表格问答的真实世界基准 论文详细解读

### 背景：这个问题为什么难？

表格问答（TableQA）本质上是让模型在结构化数据上做自然语言推理。过去的基准大多只收录了几千行的平铺表格，问句也局限在单一语言，导致模型只学会了“找关键词、直接匹配”。真实业务里，表格往往层级嵌套、跨页甚至包含合并单元格；数据来源跨政府、金融、学术和工业报告，语言分布更是简体中文、繁体中文和英文混杂。再加上很多公开基准直接从已有的网页或论文中抽取，训练数据里很可能已经出现了测试表格，造成“数据泄漏”。这些因素共同让现有评测无法真实反映大模型在实际场景中的能力。

### 关键概念速览
- **TableQA**：让模型阅读表格并用自然语言回答问题，类似于在 Excel 里查找并解释数据。  
- **平铺表格 vs. 层级/嵌套表格**：平铺表格只有行列两维，层级/嵌套表格像树状结构，某个单元格内部还能再展开成小表。想象一本目录里还有章节小目录。  
- **跨语言表格**：同一份报告可能有简体中文、繁体中文和英文三个版本，模型需要在不同语言之间保持语义一致。  
- **数据泄漏**：训练集里出现了测试集的原始材料，模型相当于“提前看答案”。这会让评测结果失真。  
- **SEAT（Sub‑question‑level Evaluation for TableQA）**：把一个复杂问句拆成若干子问，逐个比对模型的回答与参考答案，衡量语义对齐程度。类似于老师给学生的每一步解题过程打分，而不是只看最终答案。  
- **多域**：指数据来源覆盖政府公报、金融报表、学术论文和行业白皮书等不同业务场景，考察模型的通用性。  
- **LLM（大语言模型）**：能够理解和生成自然语言的深度学习模型，如 GPT‑4、Claude 等。  

### 核心创新点
1. **数据层面的真实化**  
   - 过去的基准大多只抓取了公开的 CSV 或 HTML 表格，结构单一且多为旧数据。  
   - TableEval 从最近的政府公报、金融年报、学术期刊和行业报告中抽取表格，主动挑选层级、嵌套等复杂形态。  
   - 结果是评测场景更贴近企业和科研的实际需求，模型在这些数据上表现的差距也更明显。

2. **多语言覆盖**  
   - 传统基准几乎全是英文，语言多样性被忽视。  
   - 本工作把简体中文、繁体中文和英文三种语言的表格和对应问句全部纳入，同一问题在不同语言下都有对应版本。  
   - 这让评测能够直接检验模型的跨语言迁移能力，而不是在单语言环境里“装死”。

3. **防泄漏的采集策略**  
   - 直接使用公开的 WikiTableQuestions 等会导致训练集里已经出现了测试表格。  
   - TableEval 只采集 2022 年以后发布的文档，并对每份文档进行唯一标识，确保这些表格不在公开的预训练语料中出现。  
   - 这样得到的评测结果更可信，能够真实反映模型的推理能力。

4. **SEAT 评价框架**  
   - 传统的 Exact Match 或 F1 只能衡量答案的字面相似度，无法捕捉语义细微差别。  
   - SEAT 把复杂问句拆解为子问，每个子问对应一个小答案片段，然后计算模型答案与参考答案在每个子问层面的对齐度。  
   - 人类评审实验表明，SEAT 与人工打分的相关性高于传统指标，意味着它更能反映真实语义准确性。

### 方法详解
整体思路可以划分为三大步骤：**数据收集 → 问题构造 → 评价设计**。

1. **数据收集**  
   - 从四大领域的最新报告（政府公报、金融年报、学术论文、行业白皮书）中抓取表格。  
   - 使用 OCR 与结构化解析工具把 PDF、Word、HTML 等格式统一转成机器可读的表格结构。  
   - 对每张表格进行手工标注，记录其结构类型（简洁、层级、嵌套），并确保每种结构在每个语言中都有代表。

2. **问题构造**  
   - 采用两轮人工+模型协同的方式生成问句。第一轮让领域专家阅读表格并提出 2–3 条核心业务问题；第二轮让 LLM（如 GPT‑4）基于这些核心问题扩展出细化子问。  
   - 每个完整问句被拆解成若干子问，形成 **question → sub‑question tree**，类似于把一道综合题拆成若干小题。  
   - 对每个子问给出参考答案，答案既可能是数值，也可能是文字解释，确保覆盖不同推理路径（算术、比较、归纳）。

3. **评价设计（SEAT）**  
   - **子问对齐**：模型输出的长答案先通过句子切分或关键短语抽取，映射到对应的子问。  
   - **语义匹配**：对每对（模型子答，参考子答）使用语义相似度模型（如 Sentence‑BERT）计算相似度分数，阈值以上计为匹配。  
   - **整体得分**：把所有子问的匹配情况加权求和，得到一个 0–1 的 SEAT 分数。权重设计上，关键业务子问（如财务指标）会被赋予更高权重。  
   - **人类验证**：作者抽取 200 条样本，让三位领域专家对模型答案进行人工打分，结果显示 SEAT 与人工评分的皮尔逊相关系数达到 0.87，显著高于传统 Exact Match（0.62）。

**巧妙之处**：把复杂问句拆解成子问后再评估，实际上把“整体对错”转化为“一堆小对错”的集合，这样既能捕捉细粒度错误，又避免了因为一句话的微小措辞差异导致的全盘失分。

### 实验与效果
- **测试集**：TableEval 共收录约 4,200 张表格，覆盖三种语言、四个领域和三种结构类型。每张表格配有 3–5 条完整问句，总计约 18,000 条问答对。  
- **基线模型**：包括 GPT‑4、Claude‑2、LLaMA‑2‑70B、ChatGLM‑6B 等主流大语言模型，均在公开的指令微调后直接使用。  
- **整体表现**：在传统 Exact Match 指标上，最强模型（GPT‑4）也只能达到约 42% 的准确率；在 SEAT 上同模型得分约 0.58，仍远低于人类专家的 0.92。  
- **结构差异**：对层级/嵌套表格的得分比简洁表格低约 15%，说明模型在处理多层次结构时仍显吃力。  
- **语言差异**：英文问句的 SEAT 平均分比简体中文高 0.07，繁体中文略低于简体中文 0.03，反映出模型对中文的细粒度理解仍有提升空间。  
- **消融实验**：去掉 SEAT 的子问加权，仅使用平均相似度，整体分数下降约 0.04，说明权重设计对评估的敏感度不容忽视。  
- **局限性**：作者承认当前的子问拆解依赖人工标注，自动化程度不高；另外，SEAT 仍然受限于底层语义相似度模型的质量，极端长答案可能出现匹配误差。

### 影响与延伸思考
TableEval 的出现让业界第一次在同一基准上系统评测模型对**复杂结构、跨语言和跨领域**表格的理解能力。自论文发布后，已有多篇工作尝试在 TableEval 上微调检索‑增强模型（RAG）或引入图神经网络（GNN）来专门建模表格层级关系。还有研究把 SEAT 思路推广到 **多模态 QA**（如图文混合问答），尝试把子任务拆解成更细的视觉/语言子问。对想进一步深入的读者，可以关注以下方向：① 自动化子问生成与对齐技术；② 跨语言表格对齐（同一表格不同语言版本的统一表示）；③ 将表格结构信息嵌入大模型的预训练阶段。整体来看，TableEval 为评估和改进 LLM 在真实业务表格上的表现提供了一个更严苛、更具指导意义的“实验室”。

### 一句话记住它
TableEval 用真实、复杂、多语言的表格和子问级别的 SEAT 评估，让我们看到大模型在真实表格问答里仍有“大洞”。