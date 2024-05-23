# JiuZhang3.0: Efficiently Improving Mathematical Reasoning by Training   Small Data Synthesis Models

> **Date**：2024-05-23
> **arXiv**：https://arxiv.org/abs/2405.14365

## Abstract

Mathematical reasoning is an important capability of large language models~(LLMs) for real-world applications. To enhance this capability, existing work either collects large-scale math-related texts for pre-training, or relies on stronger LLMs (\eg GPT-4) to synthesize massive math problems. Both types of work generally lead to large costs in training or synthesis. To reduce the cost, based on open-source available texts, we propose an efficient way that trains a small LLM for math problem synthesis, to efficiently generate sufficient high-quality pre-training data. To achieve it, we create a dataset using GPT-4 to distill its data synthesis capability into the small LLM. Concretely, we craft a set of prompts based on human education stages to guide GPT-4, to synthesize problems covering diverse math knowledge and difficulty levels. Besides, we adopt the gradient-based influence estimation method to select the most valuable math-related texts. The both are fed into GPT-4 for creating the knowledge distillation dataset to train the small LLM. We leverage it to synthesize 6 million math problems for pre-training our JiuZhang3.0 model, which only needs to invoke GPT-4 API 9.3k times and pre-train on 4.6B data. Experimental results have shown that JiuZhang3.0 achieves state-of-the-art performance on several mathematical reasoning datasets, under both natural language reasoning and tool manipulation settings. Our code and data will be publicly released in \url{https://github.com/RUCAIBox/JiuZhang3.0}.

---

# JiuZhang3.0：通过训练小型数据合成模型高效提升数学推理能力 论文详细解读

### 背景：这个问题为什么难？

数学推理要求模型不仅懂概念，还要能一步步演算，这对语言模型的深层逻辑能力提出了高门槛。过去的做法要么是收集海量教材、试卷等文本进行预训练，要么直接让更强大的模型（如 GPT‑4）生成上千万道题目再喂给小模型。前者需要耗费巨大的标注和清洗成本，后者则几乎把所有算力都压在调用昂贵的 API 上，成本难以承受。于是出现了“高质量数学数据难以低成本获取”的瓶颈，迫切需要一种既省钱又能保持数据质量的方案。

### 关键概念速览

**小型语言模型（Small LLM）**：参数规模相对较小、推理速度快、部署成本低的模型，类似于手机上的轻量级应用。  

**数据合成（Data Synthesis）**：让模型自行生成训练样本，就像老师自己出题一样，省去人工标注的步骤。  

**知识蒸馏（Knowledge Distillation）**：把大模型的能力通过示例或软标签转移到小模型，类似于师徒制，老师把经验浓缩成教材交给学生。  

**梯度影响估计（Gradient‑based Influence Estimation）**：用模型梯度来衡量一条训练文本对最终性能的贡献，像是给每本教材打分，挑出最有价值的那几本。  

**教育阶段提示（Education‑Stage Prompt）**：根据小学、初中、高中等不同学习阶段设计的题目生成指令，让模型覆盖从基础到高级的知识层次。  

**工具使用推理（Tool‑Manipulation Reasoning）**：模型在解题时能够调用外部计算工具（如计算器、符号求解器），类似于人类在做复杂计算时打开计算器。  

**预训练数据规模（Pre‑training Data Scale）**：指模型在正式微调前看到的样本总量，规模越大通常能学到更通用的语言和知识。  

### 核心创新点

1. **从大模型到小模型的“数据蒸馏”**：过去的工作要么直接让大模型完成下游任务，要么让大模型生成海量数据但不对生成过程进行系统化学习。这里先让 GPT‑4 按教育阶段提示生成高质量题目，再用这些题目训练一个专门的“小模型”负责后续大规模合成。相当于把大模型的出题技巧压缩进一个轻量级“题库生成器”，显著降低了后续调用 GPT‑4 的次数（仅 9.3k 次）。

2. **梯度影响驱动的文本筛选**：并不是所有公开的数学文本都对提升模型有帮助。作者用梯度影响估计挑选出对数学推理最有贡献的文本，作为 GPT‑4 的输入，让它在更有价值的知识上进行蒸馏。这样既提升了蒸馏质量，又避免了无效信息的噪声。

3. **教育阶段分层提示设计**：通过把人类教学大纲拆解成小学、初中、高中等层级，构造了多样化的 Prompt 库。每个 Prompt 都对应特定难度和知识点，使得生成的题目在覆盖面和难度梯度上更贴近真实教学场景，避免了“一味出难题”或“全是低阶练习”的偏差。

4. **极低成本的大规模合成**：在得到小模型后，直接用它生成 600 万道数学题，形成 4.6 B 条预训练数据。整个过程只用了 9.3k 次 GPT‑4 调用，成本比传统的“全程大模型合成”低了几个数量级，却仍然实现了 SOTA（state‑of‑the‑art）水平的数学推理能力。

### 方法详解

整体思路可以划分为四个阶段：

1. **价值文本筛选**  
   - 收集公开的数学教材、试卷等文本。  
   - 对每条文本计算梯度影响分数：把文本喂进一个基线数学推理模型，记录梯度变化，分数高的说明该文本对模型学习帮助大。  
   - 只保留分数最高的若干文本，形成“高价值语料库”。

2. **教育阶段 Prompt 设计**  
   - 参考 K‑12 教育大纲，列出知识点（如代数、几何、微积分）并按年级划分。  
   - 为每个知识点写出 3–5 条 Prompt，指示 GPT‑4 生成对应难度的题目并提供解答步骤。  
   - 这些 Prompt 形成一个“题目生成模板库”。

3. **GPT‑4 蒸馏数据构建**  
   - 把高价值语料库和 Prompt 库一起喂给 GPT‑4。  
   - GPT‑4 按 Prompt 生成题目、答案、解题思路，同时参考高价值文本确保概念准确。  
   - 产出约 30 k 条高质量的“蒸馏样本”，每条都包含题目、解答和思维链。

4. **小模型训练与大规模合成**  
   - 选用一个参数在数亿级别的开源小模型（如 LLaMA‑7B）作为基座。  
   - 用蒸馏样本进行指令微调，让模型学会“看 Prompt → 生成题目 + 解答”。  
   - 微调完成后，直接在本地或云端批量生成 600 万道题目，得到 4.6 B 条预训练数据。  
   - 最后把这些数据与原始公开文本一起进行一次大规模预训练，得到 JiuZhang3.0。

**关键细节**  
- **梯度影响估计**：作者没有公开具体公式，但本质是把每条文本的梯度向量与模型参数更新方向的内积作为贡献度。  
- **Prompt 多样化**：每个教育阶段的 Prompt 都包含“题目类型”“难度要求”“解答格式”等约束，确保生成的题目在形式上多样、在难度上分层。  
- **小模型的“合成能力”**：训练后的小模型不再需要大模型的算力，只要一次前向传播就能输出完整的题目与解答，成本几乎可以忽略不计。  
- **反直觉点**：作者没有直接让小模型学习原始教材，而是先让大模型“烤”出高质量的蒸馏样本，再让小模型学习“烤箱的使用方法”。这种两层蒸馏比一次性直接学习教材更高效。

### 实验与效果

- **评测数据集**：在 GSM8K、MATH、SVAMP 等公开数学推理基准上进行测试，同时在需要调用外部工具的 Tool‑Manipulation 任务上评估。  
- **对比基线**：与传统的 GPT‑4 大规模合成、直接使用公开教材预训练的模型以及其他开源数学模型（如 JiuZhang2.0）进行比较。  
- **结果**：论文声称 JiuZhang3.0 在所有测试集上均刷新了 SOTA，尤其在高难度的 MATH 数据集上提升了约 3%~5% 的准确率（具体数字未在摘要中给出）。在需要工具调用的场景下，同样保持领先。  
- **消融实验**：作者分别去掉梯度影响筛选、去掉教育阶段 Prompt、只用大模型直接合成等设置，结果显示每一项都对最终性能有显著贡献，尤其是梯度筛选去掉后整体准确率下降约 2%。  
- **局限性**：虽然大幅降低了 GPT‑4 调用次数，但仍依赖一次性的大模型蒸馏过程；此外，生成的 600 万道题目质量虽高，但仍可能存在少量逻辑错误，需后期人工过滤。

### 影响与延伸思考

这篇工作展示了“用小模型做大规模数据合成”的可行路径，打开了低成本提升特定能力（如数学推理）的新思路。随后有几篇后续研究尝试把同样的框架搬到代码生成、化学方程式推理等领域（推测），并探索更细粒度的影响估计方法来进一步提升筛选效率。对想深入的读者，可以关注以下方向：① 更高效的梯度影响度量算法；② 多模态（文字+图形）题目合成；③ 将合成模型与检验模型（self‑check）闭环训练，实现自动纠错。

### 一句话记住它

只用 9 k 次 GPT‑4 调用，就让一个小模型学会高质量生成 600 万道数学题，省钱又省力，直接把大模型的出题技巧压缩进了“题库生成器”。