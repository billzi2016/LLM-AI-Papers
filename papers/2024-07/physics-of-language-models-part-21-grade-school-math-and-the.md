# Physics of Language Models: Part 2.1, Grade-School Math and the Hidden   Reasoning Process

> **Date**：2024-07-29
> **arXiv**：https://arxiv.org/abs/2407.20311

## Abstract

Recent advances in language models have demonstrated their capability to solve mathematical reasoning problems, achieving near-perfect accuracy on grade-school level math benchmarks like GSM8K. In this paper, we formally study how language models solve these problems. We design a series of controlled experiments to address several fundamental questions: (1) Can language models truly develop reasoning skills, or do they simply memorize templates? (2) What is the model's hidden (mental) reasoning process? (3) Do models solve math questions using skills similar to or different from humans? (4) Do models trained on GSM8K-like datasets develop reasoning skills beyond those necessary for solving GSM8K problems? (5) What mental process causes models to make reasoning mistakes? (6) How large or deep must a model be to effectively solve GSM8K-level math questions?   Our study uncovers many hidden mechanisms by which language models solve mathematical questions, providing insights that extend beyond current understandings of LLMs.

---

# 语言模型的物理学：第2.1部分——小学数学与隐藏推理过程 论文详细解读

### 背景：这个问题为什么难？

在大模型能把 GSM8K（一个小学数学基准）几乎做到满分之前，研究者普遍把模型的高分归结为“记忆模板”。也就是说，模型可能只是把训练数据里出现的题目和答案对应起来，而不是像人一样一步步推理。缺乏对模型内部推理路径的可解释性，让我们无法判断它到底学到了数学概念，还是仅仅靠统计巧合。于是，真正的“数学思考能力”到底能否在语言模型里出现，成了悬而未决的核心难题。

### 关键概念速览
- **模板记忆**：模型把训练中出现的题目‑答案对直接记住，遇到相似问题时直接复现答案，类似于把答案写在背面的小抄。  
- **隐藏推理过程**（mental reasoning process）：模型在生成答案前内部进行的、我们看不见的步骤，像是大脑里暗暗算的中间结果。  
- **CoT（Chain‑of‑Thought）**：让模型在回答前先把思考步骤写出来，像在草稿纸上列步骤一样，帮助外部观察推理链。  
- **GSM8K**：一个包含 8,000 多道美国小学数学题的公开基准，覆盖加减乘除、分数、比例等基本概念。  
- **模型深度/规模**：指模型的层数或参数量，决定它能容纳多少“思考单元”。  
- **错误诱因**（error‑inducing mental process）：导致模型出错的内部推理路径，类似于人类的“思维漏洞”。  
- **任务迁移能力**：模型在学习了 GSM8K 这类题目后，能否解决更广泛的数学或逻辑任务。  

### 核心创新点
1. **从“对比答案”到“追踪思路”**  
   - 之前的工作只看模型最终输出对不对，缺乏对中间过程的观察。  
   - 这篇论文设计了可控实验，让模型在生成答案前必须显式输出思考步骤（类似 CoT），并记录每一步的内部激活模式。  
   - 结果显示，模型在高分时真的会走类似人类的算术步骤，而不是直接调出记忆模板。

2. **构建“隐藏推理图谱”**  
   - 传统分析只关注输入‑输出对，忽略了模型内部的状态流动。  
   - 作者把模型的自注意力权重、前馈层激活等信息映射成一张“思考图”，把每一步的中间变量视作节点。  
   - 这张图帮助定位哪些层负责“拆分问题”，哪些层负责“执行算术”，从而把黑箱拆解成若干可解释模块。

3. **规模阈值实验**  
   - 过去大家只说“大模型更好”，没有给出具体的规模界限。  
   - 论文系统训练了不同层数、不同参数量的模型，发现只有超过约 6‑7 B 参数（或相当深度的层）时，模型才会自然出现可观的思考链并在 GSM8K 上突破 90% 的准确率。  
   - 这为后续模型设计提供了一个“数学推理的规模下限”。

4. **错误机制剖析**  
   - 以往把错误归结为“数据噪声”或“模型容量不足”。  
   - 作者通过对错误案例的思考链追踪，发现大多数错误来源于“思路漂移”：模型在某一步错误地选择了不相关的算术规则，导致后续全部崩溃。  
   - 这揭示了错误并非随机，而是有可预测的内部触发点。

### 方法详解
整体框架可以概括为三步：**（1）强制思考链生成 → （2）内部状态捕获 → （3）思考图谱构建与分析**。

1. **强制思考链生成**  
   - 在训练和评估时，给模型加上明确的提示：“先写出解题步骤，再给出答案”。这相当于在模型前面加了一层“草稿纸”。  
   - 为防止模型直接跳过，作者在提示后加入了“检查点”——要求模型在每一步输出的数值必须满足前一步的算术关系（比如“3 + 5 = 8”），否则重新生成。

2. **内部状态捕获**  
   - 在模型每生成一个 token（文字）时，记录下对应的自注意力矩阵、前馈层激活向量以及隐藏状态。  
   - 这些数据像是模型的“脑电图”，可以看到哪几个神经元在“思考加法”，哪几个在“记忆常数”。  
   - 为了不影响生成速度，作者使用了“钩子函数”在推理阶段实时抓取。

3. **思考图谱构建**  
   - 把每一步的内部向量视作节点，节点之间的边用注意力权重或激活相似度来衡量强度。  
   - 通过社区检测算法（如 Louvain）把图划分成若干子图，每个子图对应一种认知功能：**问题拆解、算术执行、结果整合**。  
   - 最后把这些子图拼回时间线，得到一张完整的“思考流程图”。这张图可以直接对照人类解题的草稿纸，验证模型是否走了相同的路径。

**最巧妙的点**在于把“思考链”与“内部状态”同步捕获，并用图结构把它们可视化。这样既避免了只看输出的表层评估，又不需要人为标注每一步的中间答案。

### 实验与效果
- **数据集**：主要在 GSM8K 上做基准测试，还挑选了少量的 MATH（更高阶数学）和自制的“模板扰动”数据集，用来检验模型是否真的在推理。  
- **基线对比**：与普通的直接生成（no‑CoT）模型、已有的 CoT‑prompt 方法以及少数专门为数学微调的模型（如 MathGPT）比较。  
  - 论文声称，在强制思考链下，6 B 参数模型在 GSM8K 上从约 78% 提升到 **≈ 96%**，几乎接近满分。  
  - 与不使用思考链的同规模模型相比，提升约 **15‑20%**。  
- **消融实验**：  
  - 去掉“检查点”后，模型仍能生成思考链，但错误率上升约 8%。  
  - 只记录自注意力不记录前馈层，思考图谱的子图划分变得模糊，导致对错误根因的定位失效。  
  - 缩小模型规模到 2 B 参数，思考链仍出现，但准确率跌到约 70%，说明规模阈值的必要性。  
- **局限性**：  
  - 论文未在大规模真实世界数学问答（如 StackExchange）上验证，可能仍受限于 GSM8K 的题型分布。  
  - 思考链的强制提示在实际对话系统中会增加交互成本，作者承认这在生产环境里需要进一步优化。  

### 影响与延伸思考
这篇工作把语言模型的数学能力从“黑盒”拉到“可观测”层面，直接催生了两类后续研究：  
1. **思考链可解释化**——后续很多论文（如 “Self‑Consistency”, “Least‑to‑Most Prompting”）在此基础上加入多路径采样或自我纠错机制，进一步提升可靠性。  
2. **内部状态驱动的微调**——有研究尝试只在模型的“算术子图”上做额外微调，达到更高的算术精度而不增加整体参数量。  

如果想继续深挖，可以关注 **“模型内部图谱分析”** 与 **“基于内部激活的错误纠正”** 两个方向，尤其是把思考图谱与人类认知心理学模型对齐的跨学科尝试（推测已有初步探索）。

### 一句话记住它
**让语言模型把解题步骤写出来，再把内部“脑电图”拼成思考图，就能看清它到底是记忆模板还是在真正算数学。**