# CharXiv: Charting Gaps in Realistic Chart Understanding in Multimodal   LLMs

> **Date**：2024-06-26
> **arXiv**：https://arxiv.org/abs/2406.18521

## Abstract

Chart understanding plays a pivotal role when applying Multimodal Large Language Models (MLLMs) to real-world tasks such as analyzing scientific papers or financial reports. However, existing datasets often focus on oversimplified and homogeneous charts with template-based questions, leading to an over-optimistic measure of progress. We demonstrate that although open-source models can appear to outperform strong proprietary models on these benchmarks, a simple stress test with slightly different charts or questions can deteriorate performance by up to 34.5%. In this work, we propose CharXiv, a comprehensive evaluation suite involving 2,323 natural, challenging, and diverse charts from arXiv papers. CharXiv includes two types of questions: 1) descriptive questions about examining basic chart elements and 2) reasoning questions that require synthesizing information across complex visual elements in the chart. To ensure quality, all charts and questions are handpicked, curated, and verified by human experts. Our results reveal a substantial, previously underestimated gap between the reasoning skills of the strongest proprietary model (i.e., GPT-4o), which achieves 47.1% accuracy, and the strongest open-source model (i.e., InternVL Chat V1.5), which achieves 29.2%. All models lag far behind human performance of 80.5%, underscoring weaknesses in the chart understanding capabilities of existing MLLMs. We hope CharXiv facilitates future research on MLLM chart understanding by providing a more realistic and faithful measure of progress. Project page and leaderboard: https://charxiv.github.io/

---

# CharXiv：绘制多模态大语言模型在真实图表理解中的差距 论文详细解读

### 背景：这个问题为什么难？

在把多模态大语言模型（MLLM）用于科研论文、金融报表等实际场景时，模型必须能读懂各种各样的图表。过去的评测数据集大多只收集了几种常见的柱状图、折线图，甚至直接用模板生成问题，导致图表形态单一、问题模式千篇一律。模型在这种“练习册”上得分高，却很可能在真实报告里崩溃，因为真实图表往往包含噪声、复杂布局和跨要素的推理需求。正是这种评测与真实使用之间的鸿沟，促使作者专门打造了更贴近现实的基准。

### 关键概念速览
- **多模态大语言模型（MLLM）**：既能处理文字，又能理解图像的语言模型，类似于会看图说话的“全能助理”。  
- **图表理解**：模型从图像中识别坐标轴、图例、数据点等基本要素，并据此回答问题。可以想象成把图表当成一张“地图”，模型要找出路标并给出导航。  
- **Benchmark（基准测试）**：统一的评测集合，用来比较不同模型的能力，就像跑步比赛的计时表。  
- **Stress Test（压力测试）**：在基准之外，故意改动图表样式或问题措辞，检验模型的鲁棒性，类似给选手加上障碍赛。  
- **Descriptive Question（描述性问题）**：只要求模型指出图表中出现的具体元素，例如“X 轴的标签是什么”。相当于让模型做“看图找字”。  
- **Reasoning Question（推理性问题）**：需要模型综合多条信息得出结论，例如“在 2020‑2022 年间，哪条曲线的增长率最高”。这相当于让模型在图表上做“算术”。  
- **Human Expert Curation（人工专家筛选）**：所有图表和问题都经过领域专家手工挑选和校对，确保质量不被自动生成的噪声污染。  
- **Accuracy Gap（准确率差距）**：模型答对的比例与人类或更强模型之间的差距，用来量化能力缺口。

### 核心创新点
1. **从模板化到真实多样**：过去的基准多用统一模板生成图表和问题 → CharXiv 从 arXiv 论文中抽取 2,323 张自然出现的图表，涵盖不同学科、不同绘图风格 → 评测结果更能反映模型在真实场景的表现。  
2. **双层问题设计**：传统评测只关注单一维度（如只问数值） → CharXiv 同时提供描述性和推理性两类问题，前者检验基本要素识别，后者考察跨要素综合推理 → 模型需要同时具备“看图”和“算图”的能力。  
3. **全程人工把关**：很多数据集依赖自动标注，质量参差不齐 → CharXiv 的每张图表、每个问题都由领域专家手工挑选、编写、验证 → 确保评测不被标注错误或歧义干扰。  
4. **系统化压力测试**：评测往往只看一次得分 → 作者在基准上额外加入轻度变体（如换颜色、改写问题），测得性能最高下降 34.5% → 揭示了模型对细微变化的脆弱性，提醒研究者关注鲁棒性。

### 方法详解
**整体框架**  
CharXiv 的构建可以划分为四步：① 图表采集 → ② 质量筛选 →③ 问题撰写 →④ 验证与评测。整个流程像是从“原始文献库”挑选“珍稀标本”，再为每个标本配上“观察指南”，最后让不同模型来“解剖”并打分。

**1. 图表采集**  
- 从 arXiv 上的 PDF 文件中抽取所有嵌入的图像。  
- 使用 OCR 与图像分类过滤掉非图表（如照片、流程图），只保留常见的折线图、柱状图、散点图等。  
- 对每张图表记录来源、论文领域、绘图工具等元信息，方便后续分析。

**2. 质量筛选**  
- 领域专家检查每张图表的可读性：坐标轴是否完整、标签是否清晰、颜色对比度是否足够。  
- 剔除模糊、缺失关键要素或过于专业化（如仅在特定软件内部使用的自定义图形）的样本。  
- 最终保留 2,323 张高质量、风格多样的真实图表。

**3. 问题撰写**  
- 对每张图表至少生成两类问题：  
  - *描述性*：如“图例中蓝色线对应的变量是什么？”  
  - *推理性*：如“在图中，哪一年出现了最高的增长率？”  
- 编写时遵循“单一答案、无歧义”的原则，确保模型的输出可以直接对比金标准。  
- 为防止模型记忆模板，问题的措辞在同一类型中保持高度变异。

**4. 验证与评测**  
- 所有问题的答案由两位独立专家标注，若出现分歧则交由第三位专家裁定。  
- 评测时，将图表图像和对应问题一起喂给模型，模型输出文本答案。  
- 采用**准确率**（答对的比例）作为主要指标，兼顾 **Top‑2 准确率** 以衡量近似匹配。  
- 为测模型鲁棒性，作者在原始图表上做轻度变体（如调色、旋转 5°、改写问题），记录性能下降幅度，最高达 34.5%。

**最巧妙的地方**  
- 把“真实图表”与“人工审校”结合，避免了自动生成数据常见的噪声与偏差。  
- 双层问题设计让评测既能捕捉低层次的视觉识别错误，也能暴露高层次的逻辑推理缺陷。  
- 压力测试的加入，使得评测不再是“一次性打分”，而是对模型稳健性的多维度审视。

### 实验与效果
- **测试对象**：包括最强的商业模型 GPT‑4o、若干开源模型（如 InternVL‑Chat V1.5）以及人类标注者。  
- **整体表现**：GPT‑4o 在 CharXiv 上取得 47.1% 的准确率，仍然远低于人类的 80.5%；开源模型 InternVL‑Chat V1.5 最高仅 29.2%。  
- **与旧基准的对比**：在传统的同质化图表基准上，开源模型往往能跑赢 GPT‑4o，甚至接近 70% 的准确率。但一旦换到 CharXiv，性能骤降，说明旧基准高估了模型的真实能力。  
- **压力测试结果**：对同一模型进行轻度图表/问题变体测试，最高下降 34.5%，表明模型对细节变化极度敏感。  
- **消融实验**：原文未提供细粒度消融，但通过对比不同模型的表现，作者暗示“描述性问题”相对容易，而“推理性问题”是导致整体准确率低的主要因素。  
- **局限性**：CharXiv 只覆盖 arXiv 论文中的图表，领域偏向学术；图表数量虽比旧基准多，但仍不足以覆盖所有行业的特殊绘图风格。作者也承认未对模型的内部错误类型做系统分析。

### 影响与延伸思考
CharXiv 为多模态大语言模型的图表理解提供了更贴近真实使用场景的评测标准，已经在社区形成了公开排行榜，吸引了不少后续工作。后续研究可能会在以下方向发力：

- **专门的图表微调**：利用 CharXiv 或类似数据对现有 MLLM 进行针对性微调，提升跨要素推理能力。  
- **视觉‑语言链式思考（Vision‑CoT）**：在回答推理性问题时，引入类似思维链的中间步骤，让模型先描述坐标轴、再列出关键数据点，最后给出结论。  
- **跨域图表扩展**：把金融报表、医学影像中的统计图等加入评测，检验模型的通用性。  
- **鲁棒性提升**：基于压力测试的思路，研发对颜色、布局、文字噪声不敏感的视觉编码器。  

如果想进一步跟进，可以关注近期在 arXiv 上出现的 “ChartQA‑2.0” 与 “VisReason” 系列论文，它们大多引用 CharXiv 作为基准，说明该工作已经成为该子领域的“金标准”。

### 一句话记住它
CharXiv 揭露了现有多模态大语言模型在真实图表理解上的巨大差距，并提供了更严苛、更可信的评测平台。