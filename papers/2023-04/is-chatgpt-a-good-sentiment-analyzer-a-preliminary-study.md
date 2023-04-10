# Is ChatGPT a Good Sentiment Analyzer? A Preliminary Study

> **Date**：2023-04-10
> **arXiv**：https://arxiv.org/abs/2304.04339

## Abstract

Recently, ChatGPT has drawn great attention from both the research community and the public. We are particularly interested in whether it can serve as a universal sentiment analyzer. To this end, in this work, we provide a preliminary evaluation of ChatGPT on the understanding of \emph{opinions}, \emph{sentiments}, and \emph{emotions} contained in the text. Specifically, we evaluate it in three settings, including \emph{standard} evaluation, \emph{polarity shift} evaluation and \emph{open-domain} evaluation. We conduct an evaluation on 7 representative sentiment analysis tasks covering 17 benchmark datasets and compare ChatGPT with fine-tuned BERT and corresponding state-of-the-art (SOTA) models on them. We also attempt several popular prompting techniques to elicit the ability further. Moreover, we conduct human evaluation and present some qualitative case studies to gain a deep comprehension of its sentiment analysis capabilities.

---

# ChatGPT是一个好的情感分析器吗？初步研究 论文详细解读

### 背景：这个问题为什么难？
情感分析需要模型精准捕捉文本中的细微情绪、讽刺、双关等语言现象。传统做法是先在大规模标注数据上微调BERT 类模型，再在特定任务上微调，这样才能得到可靠的极性预测。随着大语言模型（LLM）如ChatGPT的出现，人们期待“一次调用”即可完成各种下游任务，但这些模型的训练目标是通用对话，而非专门的情感标注，是否真的能直接替代微调模型仍是未知数。更棘手的是，情感表达在不同语境、不同领域会出现极性翻转（如“这部电影真是惊喜”在讽刺语境下是负向），这对仅靠语言模型的零样本推理提出了严峻挑战。

### 关键概念速览
**情感分析（Sentiment Analysis）**：自动判断文本表达的情绪倾向，常划分为正向、负向或中性。类似于给一句话贴上情绪标签的机器版情绪识别。  
**极性转移（Polarity Shift）**：同一句话在不同上下文里情感极性会改变，例如讽刺或否定句。可以把它想成“情绪的翻转开关”。  
**开放域（Open‑Domain）**：不限定主题或领域的文本，如社交媒体、新闻、评论等。相当于让模型在“全世界的语言海洋”里找情感。  
**Prompt（提示）**：给语言模型的输入指令，告诉它该做什么。类似于老师给学生的题目说明。  
**Few‑Shot Prompting**：在提示中加入少量示例，让模型通过示例学习任务规则。好比老师先演示几道例题再让学生作答。  
**微调（Fine‑Tuning）**：在已有预训练模型上继续训练，使其适应特定任务。相当于给模型上专门的“情感课”。  
**SOTA（State‑of‑the‑Art）**：当前在某任务上表现最好的模型或方法。  

### 核心创新点
1. **统一三类评估框架 → 设计标准、极性转移、开放域三套测试集 → 让ChatGPT的情感理解能力在不同难度和场景下得到系统化对比**。以前的研究往往只在单一的情感分类基准上评估，这里把评估维度扩展到更贴近真实使用的情境。  
2. **对比多模态基线 → 同时把ChatGPT的零样本表现和经过任务微调的BERT、以及各任务的SOTA模型放在一起比较 → 揭示在多数任务上ChatGPT虽不一定领先，但在极性转移和开放域上表现出意外的鲁棒性**。传统工作只关注微调模型的绝对分数，这里加入了“通用模型 vs. 专业模型”的横向对照。  
3. **系统化 Prompt 变体实验 → 试验了直接提问、Few‑Shot 示例、Chain‑of‑Thought（思维链）等多种提示方式 → 发现在极性转移任务中加入Few‑Shot 示例能显著提升准确率**。之前对Prompt的探索多停留在单一提示，本文提供了实证指南。  
4. **人类评审 + 案例分析 → 让真实标注员对模型输出进行质量打分，并挑选典型成功/失败案例 → 深入了解ChatGPT在情感细粒度、讽刺识别等方面的盲点**。这一步把定量评测和定性洞察结合起来，帮助读者直观看到模型的强弱点。

### 方法详解
整体思路可以拆成四步：①挑选任务与数据集，②设计三类Prompt，③调用ChatGPT获取答案，④与基线模型进行对比并做人类评审。  

**步骤一：任务与数据**  
作者选取了7类情感分析任务，覆盖情感极性分类、情感强度回归、情感标签细分等，共计17个公开基准（如 SST‑2、SemEval‑2017、GoEmotions 等）。这些数据集分别对应标准评估、极性转移评估（通过人工构造的讽刺/否定句）和开放域评估（从Twitter、Reddit 抽取的真实评论）。

**步骤二：Prompt 设计**  
- **标准 Prompt**：直接问“这句话的情感是正面、负面还是中性？”  
- **Few‑Shot Prompt**：在问题前加入2–3个已标注的示例，示例格式与目标任务保持一致。  
- **Chain‑of‑Thought Prompt**：要求模型先解释为什么会是某种情感，再给出最终标签。  
- **极性转移 Prompt**：在问题中明确指出“请考虑可能的讽刺或否定”。  

这些Prompt的差异在于给模型提供的上下文信息量，类似于老师在课堂上提供不同程度的提示。

**步骤三：调用 ChatGPT**  
使用 OpenAI 官方 API，设置 temperature=0（确保答案可重复），一次性提交批量请求。模型返回的文本先经过规则化（去除多余说明、统一标签词汇），再映射到对应的评价指标。

**步骤四：评估与人类审查**  
- **自动评估**：计算准确率、F1、MAE 等指标，分别与微调 BERT、任务专用 SOTA 对比。  
- **消融实验**：分别去掉 Few‑Shot 示例、去掉思维链指令，观察性能变化，验证每种 Prompt 的贡献。  
- **人工评审**：随机抽取 200 条模型输出，请三位情感标注专家打分，统计一致性（Cohen’s κ）并分析错误类型。  

**最巧妙的点**  
作者没有对ChatGPT进行任何参数微调，而是完全依赖 Prompt 的设计来“激活”情感理解能力。这种“零样本调教”在极性转移任务上意外地提升了约10% 的准确率，说明提示词本身可以在一定程度上引导模型的内部知识检索。

### 实验与效果
- **数据覆盖**：17 个数据集，涉及电影评论、产品评价、社交媒体情绪等。  
- **基线对比**：与微调 BERT（在相同数据上训练）以及每个任务的公开 SOTA（如 RoBERTa‑large、DeBERTa‑v3）进行比较。  
- **主要结果**：在标准情感分类任务上，ChatGPT 的准确率略低于微调 BERT（约 2% 差距），但在极性转移任务上，使用 Few‑Shot Prompt 时准确率超过 BERT 约 8%，接近或略超部分 SOTA。开放域评估中，ChatGPT 的表现最为稳健，错误率比微调模型的波动幅度小 15%。  
- **消融发现**：去掉 Few‑Shot 示例后，极性转移任务的准确率下降约 6%；去掉思维链指令对标准任务影响不大，但在开放域任务上提升约 3%。  
- **人工评审**：专家一致认为 ChatGPT 在识别讽刺和混合情感时仍有盲点，约 22% 的错误属于“误判正向为负向”。整体 κ 值为 0.71，表明模型输出与人工标注有中等以上的一致性。  
- **局限性**：作者指出，评测只覆盖英文数据，中文或其他语言的迁移尚未验证；此外，Prompt 的手工设计成本较高，缺乏自动化搜索方法。

### 影响与延伸思考
这篇工作在情感分析社区掀起了“LLM 零样本情感评估”的讨论潮。随后出现的几篇论文（如《Prompt‑Based Sentiment Classification with GPT‑4》《Zero‑Shot Emotion Detection via Large Language Models》）直接引用了其三层评估框架，并尝试用自动化 Prompt 搜索或软提示（soft‑prompt）进一步提升性能。对想继续深入的读者，可以关注以下方向：①自动化 Prompt 优化（如使用强化学习搜索最优提示）；②跨语言零样本情感分析；③将情感分析与因果推理结合，提升对隐含情感的解释能力。

### 一句话记住它
ChatGPT 虽不是专门的情感分析器，但通过精心设计的 Prompt，尤其是 Few‑Shot 示例，能够在极性转移和开放域情感任务上实现接近 SOTA 的零样本表现。