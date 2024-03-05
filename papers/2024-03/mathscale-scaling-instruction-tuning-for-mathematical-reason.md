# MathScale: Scaling Instruction Tuning for Mathematical Reasoning

> **Date**：2024-03-05
> **arXiv**：https://arxiv.org/abs/2403.02884

## Abstract

Large language models (LLMs) have demonstrated remarkable capabilities in problem-solving. However, their proficiency in solving mathematical problems remains inadequate. We propose MathScale, a simple and scalable method to create high-quality mathematical reasoning data using frontier LLMs (e.g., {\tt GPT-3.5}). Inspired by the cognitive mechanism in human mathematical learning, it first extracts topics and knowledge points from seed math questions and then build a concept graph, which is subsequently used to generate new math questions. MathScale exhibits effective scalability along the size axis of the math dataset that we generate. As a result, we create a mathematical reasoning dataset (MathScaleQA) containing two million math question-answer pairs. To evaluate mathematical reasoning abilities of LLMs comprehensively, we construct {\sc MwpBench}, a benchmark of Math Word Problems, which is a collection of ten datasets (including GSM8K and MATH) covering K-12, college, and competition level math problems. We apply MathScaleQA to fine-tune open-source LLMs (e.g., LLaMA-2 and Mistral), resulting in significantly improved capabilities in mathematical reasoning. Evaluated on {\sc MwpBench}, MathScale-7B achieves state-of-the-art performance across all datasets, surpassing its best peers of equivalent size by 42.9\% in micro average accuracy and 43.7\% in macro average accuracy, respectively.

---

# MathScale：大规模指令微调用于数学推理 论文详细解读

### 背景：这个问题为什么难？

在过去，语言模型虽然在聊天、写作等任务上已经相当强大，但面对数学题目时仍常常卡壳。传统的做法是直接收集已有的数学题库（如 GSM8K、MATH）进行微调，然而这些数据规模有限，且题目类型分布不均，导致模型在新题型或更高难度的推理上表现乏力。更重要的是，数学推理需要模型掌握概念之间的层级关系和解题步骤，这种结构化的知识在普通文本中很少出现，单纯靠大规模通用语料难以获得。于是，如何高效生成覆盖广、质量高的数学推理数据，成为提升模型数学能力的瓶颈。

### 关键概念速览
- **概念图（Concept Graph）**：把数学概念（如“二次方程”“求导”）当作节点，概念之间的前后依赖（比如“求导”需要先懂“函数”）当作连线，形成一张知识网络。类似于学习路线图，帮助模型知道先学什么再学什么。
- **指令微调（Instruction Tuning）**：在大模型上继续训练，让它学会按照特定的指令格式（如“先解释思路，再给答案”）输出。相当于给模型一本使用手册，提升对任务的遵循度。
- **思维链（Chain‑of‑Thought, CoT）**：让模型在回答前把推理过程写出来，像人在解题时写草稿一样，能够显著提升复杂题目的正确率。
- **微平均（Micro Average）/宏平均（Macro Average）**：评价指标的两种聚合方式。微平均把所有样本当成一个大集合计数，宏平均先对每个子数据集算准确率再取平均，前者偏重样本多的任务，后者更公平地衡量每个任务的表现。
- **种子题（Seed Questions）**：最初收集的一小批高质量数学题，用来抽取概念并启动数据生成流程。相当于“原始材料”，后面的合成全部围绕它展开。
- **前沿大模型（Frontier LLM）**：指在生成阶段使用的最强模型（如 GPT‑3.5），它的高质量输出为后续的开源模型提供“老师”样本。

### 核心创新点
1. **从概念图出发的数学题生成**  
   - 之前的合成方法多是直接给大模型一个模板，让它随意生成题目，结果往往缺乏系统性和覆盖度。  
   - 这篇论文先把种子题拆解成概念和知识点，构建概念图，然后在图上采样子结构作为生成提示。  
   - 这样生成的题目在概念层面更均衡，能够覆盖从小学到竞赛的完整知识谱，显著提升数据的多样性和可控性。

2. **利用前沿模型进行大规模高质量数据合成**  
   - 传统做法往往直接用已有的公开数据进行微调，数据量受限。  
   - 这里把 GPT‑3.5 当作“自动老师”，在概念图的约束下让它生成题目和详细解答，并通过自动过滤保留高质量样本。  
   - 结果得到 200 万对问答，规模比公开数据集大数十倍，同时保持了专业的解题步骤。

3. **统一的数学推理基准 MwpBench**  
   - 过去的评测往往只看单一数据集，难以判断模型的通用推理能力。  
   - 作者把十个已有的数学文字题库（包括 K‑12、大学、竞赛层次）整合成 MwpBench，提供微平均和宏平均两套指标。  
   - 这让不同模型的比较更公平，也帮助发现模型在特定难度层次上的薄弱环节。

4. **在开源模型上实现同等规模的 SOTA**  
   - 以往只有闭源的大模型（如 GPT‑4）在数学推理上领先。  
   - 通过在 MathScaleQA 上进行指令微调，LLaMA‑2‑7B、Mistral‑7B 等开源模型的表现跃升至同等规模模型的最前列，微平均提升 42.9%，宏平均提升 43.7%。  
   - 证明了“数据+微调”组合可以弥补模型规模的不足。

### 方法详解
**整体框架**  
这篇论文的工作流可以划分为四大步骤：①种子题抽取概念，②构建概念图，③基于概念图合成新题，④指令微调开源模型。整个过程像是先搭建一座知识地图，再让最强的老师在地图指引下批量写教材，最后把教材教给学生模型。

**1. 概念抽取与图构建**  
- 从公开的数学题库中挑选约几千道高质量题目作为种子。  
- 使用预训练的语言模型（或手工规则）把每道题的题干和解答拆解成若干“知识点”。例如，“求解二次方程 ax²+bx+c=0”会被标记为“二次方程”“判别式”“根的公式”。  
- 将这些知识点视作节点，依据教材中的先后顺序或数学逻辑添加有向边，形成概念图。边的方向表示“先学 A，才能学 B”。这一步类似于把教材章节目录转成网络结构。

**2. 图驱动的题目采样**  
- 在概念图上随机或策略性地抽取子图。子图的大小决定了目标题目的难度：子图越大、包含的概念越多，题目越综合。  
- 把抽取的子图转化为提示（prompt），交给 GPT‑3.5。提示会明确要求模型：①围绕子图中的概念设计一道文字题，②给出逐步解答（CoT），③保持答案唯一且可检验。  
- 通过多次采样，同一子图可以生成多道变体，极大提升数据的多样性。

**3. 质量过滤与数据整理**  
- 自动检查生成的答案是否符合数学常识（例如使用符号计算库验证数值结果），不符合的被剔除。  
- 对解答的思维链进行长度和完整性检查，确保每一步都有明确的推理依据。  
- 最终得到的 2 M 对问答被统一格式化为指令微调所需的 “问题 → 思考过程 → 最终答案” 三段式。

**4. 指令微调**  
- 选取开源的基础模型（LLaMA‑2‑7B、Mistral‑7B 等），在 MathScaleQA 上进行指令微调。训练目标是让模型在收到类似 “请解答以下数学题并写出思考过程” 的指令时，能够输出符合 CoT 风格的完整解答。  
- 为了提升泛化，还在微调过程中混入少量真实数据集（如 GSM8K）作为对齐参考。  
- 训练结束后得到的模型被称为 MathScale‑7B（对应 7 B 参数规模的模型），随后在 MwpBench 上进行评测。

**巧妙之处**  
- **概念图的可控性**：通过显式的概念约束，生成过程不再是“黑箱”，研究者可以精准调节题目难度和覆盖范围。  
- **前沿模型的教师角色**：利用 GPT‑3.5 的强推理能力，省去了人工编写大规模教材的成本，同时保持了解题步骤的高质量。  
- **指令微调+CoT 双管齐下**：微调让模型学会遵循指令，CoT 让它在内部形成可检验的思维链，两者相辅相成，显著提升了数学推理的可靠性。

### 实验与效果
- **评测基准**：MwpBench 包含十个子数据集，覆盖 K‑12（如 GSM8K、MAWPS）、大学水平（如 Algebra 2000）以及国际数学竞赛题目（如 IMO）。提供微平均和宏平均两套准确率指标。  
- **主要结果**：在 MwpBench 上，MathScale‑7B 的微平均准确率比同参数规模的最佳对手高出 42.9%，宏平均高出 43.7%。这意味着无论是样本多的常规题目，还是样本少的高难度题目，MathScale‑7B 都实现了显著提升。  
- **基线对比**：与 LLaMA‑2‑7B、Mistral‑7B（未微调）以及公开的开源微调模型（如 Alpaca、Vicuna）相比，提升幅度均在 30% 以上。与闭源的 GPT‑3.5（直接使用）相比，差距进一步缩小。  
- **消融实验**：作者分别去掉概念图约束、去除 CoT 生成、只用真实数据不做合成，发现每一项都对最终性能有明显贡献。尤其是去掉概念图后，微平均下降约 15%，说明概念图是提升数据质量和覆盖度的关键。  
- **局限性**：数据合成依赖 GPT‑3.5 的输出质量，若前沿模型出现系统性偏差，合成数据也会受影响。概念抽取仍然基于规则和模型混合，可能遗漏一些高级概念的细微差别。评测仅限于文字题，未覆盖图形、几何构造等非文字形式的数学推理。

### 影响与延伸思考
- **领域影响**：MathScale 的概念图驱动合成思路在随后的一批工作中被广泛引用，例如利用知识图谱生成编程题、化学方程式等领域的合成数据。它证明了“结构化知识 + 强模型生成”可以在不增加模型规模的情况下显著提升特定推理能力。  
- **后续工作**：已有研究尝试把概念图与检索结合，让模型在生成时动态查找相关概念（如 GraphPrompt、Retrieval‑augmented Math）。还有人把这种方法推广到多模态数学（如几何图形）或跨学科推理（物理、统计）。  
- **进一步探索**：如果想深入，可以关注以下方向：①更自动化的概念抽取技术（利用图神经网络），②在合成阶段加入对抗训练提升数据多样性，③将概念图与人类教师的交互式标注结合，形成闭环的持续学习系统。

### 一句话记住它
用概念图约束前沿模型批量生成数学题，再用指令微调把这些“教材”教给开源模型，轻松让 7 B 参数的模型在数学推理上跑出 SOTA。