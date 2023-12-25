# What Makes Good Data for Alignment? A Comprehensive Study of Automatic   Data Selection in Instruction Tuning

> **Date**：2023-12-25
> **arXiv**：https://arxiv.org/abs/2312.15685

## Abstract

Instruction tuning is a standard technique employed to align large language models to end tasks and user preferences after the initial pretraining phase. Recent research indicates the critical role of data engineering in instruction tuning -- when appropriately selected, only limited data is necessary to achieve superior performance. However, we still lack a principled understanding of what makes good instruction tuning data for alignment, and how we should select data automatically and effectively. In this work, we delve deeply into automatic data selection strategies for alignment. We start with controlled studies to measure data across three dimensions: complexity, quality, and diversity, along which we examine existing methods and introduce novel techniques for enhanced data measurement. Subsequently, we propose a simple strategy to select data samples based on the measurement. We present deita (short for Data-Efficient Instruction Tuning for Alignment), a series of models fine-tuned from LLaMA and Mistral models using data samples automatically selected with our proposed approach. Empirically, deita performs better or on par with the state-of-the-art open-source alignment models with only 6K SFT training data samples -- over 10x less than the data used in the baselines. When further trained with direct preference optimization (DPO), deita-Mistral-7B + DPO trained with 6K SFT and 10K DPO samples achieve 7.55 MT-Bench and 90.06% AlpacaEval scores. We anticipate this work to provide tools on automatic data selection, facilitating data-efficient alignment. We release our models as well as the selected datasets for future researches to effectively align models more efficiently.

---

# 什么样的数据有助于对齐？指令微调中自动数据选择的系统研究 论文详细解读

### 背景：这个问题为什么难？
在大语言模型（LLM）预训练完成后，想让模型真正满足用户需求，需要进行指令微调（instruction tuning）来对齐模型的行为。过去的做法往往是把海量的指令数据全部喂进去，既耗时又耗算力，而且并不保证每一条指令都对模型有帮助。研究发现，数据的质量、复杂度和多样性对对齐效果影响巨大，但我们缺乏系统的度量标准和自动筛选手段。于是出现了“更多数据一定更好”的误区，导致很多开源对齐项目使用上百万人规模的指令集合，却仍然在细节上表现平平。要在保持或提升性能的同时大幅削减数据量，就必须弄清“好数据”到底长什么样，并能自动挑选出来。

### 关键概念速览
**指令微调（Instruction Tuning）**：在大模型预训练后，用一批“指令‑响应”对进行再训练，使模型学会遵循自然语言指令。类似于给已经会说话的机器人再上几堂任务课。

**对齐（Alignment）**：让模型的输出符合人类价值观、偏好和安全要求。可以想象成把模型的“自由意志”调到和用户的需求同步。

**复杂度（Complexity）**：指一条指令在语言、推理或任务层面的难度。像是让模型解一道高中数学题比回答“今天天气怎样”要复杂得多。

**质量（Quality）**：衡量指令‑响应对的正确性、完整性和可读性。质量高的对就像老师给出的标准答案，低质量的对可能有错误或歧义。

**多样性（Diversity）**：指数据集合在任务类型、主题、语言风格等方面的覆盖范围。多样性好相当于让模型接受了“全科”训练，而不是只学会了几门课。

**Evol‑Complexity / Evol‑Quality**：本文提出的两套基于模型自身演化的度量方法，用来量化指令的复杂度和质量。把模型当成“测量仪”，让它在不同阶段评估同一条指令的表现。

**直接偏好优化（Direct Preference Optimization, DPO）**：一种在指令微调后进一步微调模型的技术，直接使用人类偏好数据来最大化模型输出的满意度。

### 核心创新点
1. **从三维度系统评估数据**：过去的研究多聚焦于单一指标（比如人工过滤），而本文把指令数据的好坏拆解为复杂度、质量和多样性三条轴线，分别设计可度量的指标并在受控实验中验证它们对对齐效果的独立贡献。这样可以精准定位哪些数据真正推动模型进步。

2. **Evol‑Complexity 与 Evol‑Quality 度量**：作者让同一基模型在不同训练阶段（如未微调、轻度微调、重度微调）分别对同一条指令进行预测，比较预测差异来估算该指令的复杂度和质量。相比传统的人工打标签或使用外部评估模型，这种自洽的方式更省成本且与模型自身能力匹配。

3. **极简数据选择策略**：基于上述三维度分数，作者直接挑选分数最高的前 6,000 条指令作为 SFT（Supervised Fine‑Tuning）数据。没有复杂的搜索或强化学习，只是“一刀切”选最好的，结果却能在多项基准上匹配甚至超越使用数十万条数据的模型。

4. **DEITA 系列模型与 DPO 结合**：在仅用 6K SFT 数据微调后，再用 10K 人类偏好样本进行 DPO，显著提升了模型在 MT‑Bench（7.55）和 AlpacaEval（90.06%）上的表现。展示了少量高质量数据配合少量偏好优化即可达到强竞争力。

### 方法详解
**整体框架**  
整个流程可以概括为四步：① 构建原始指令库；② 用 Evol‑Complexity 与 Evol‑Quality 给每条指令打分；③ 按分数挑选前 K 条（本文 K=6,000）作为 SFT 数据；④ 用选出的 SFT 数据微调基模型，随后可选 DPO 进一步提升。核心思想是让模型自己评估数据，而不是依赖外部人工标注。

**步骤 1：原始指令库**  
作者收集了公开的指令集合（如 Alpaca、OpenAssistant 等），总量在数十万条。每条记录包含指令、输入（若有）和期望的模型输出。

**步骤 2：Evol‑Complexity 计算**  
- 先把基模型（如 LLaMA‑7B）在 **未微调** 状态下对指令进行一次前向推理，记录输出的困惑度（log‑prob）或错误率。  
- 再把同一模型在 **轻度微调**（少量通用指令）后再次推理，同样记录指标。  
- 两次推理的差值越大，说明该指令对模型的挑战性越高，即复杂度高。直观上，这相当于让模型“先学会基本功，再尝试更难的题”，看它的进步幅度。

**步骤 3：Evol‑Quality 计算**  
- 使用 **未微调** 模型生成的答案与人工标注的参考答案做相似度（BLEU、ROUGE 等），得到质量基线。  
- 再用 **重度微调**（大量高质量指令）后的模型生成答案，同样计算相似度。  
- 如果重度微调后质量提升明显，则说明原始指令‑响应对本身质量较低；相反提升不大则说明原始对已经很高质量。于是把提升幅度的倒数作为质量分数。

**步骤 4：多样性评估**  
作者用聚类（基于指令的语义嵌入）统计每条指令所在簇的密度，稀疏的簇代表稀有任务或主题，给予额外奖励分。这样可以防止选出来的 6K 条全部集中在几类常见任务上。

**步骤 5：数据筛选**  
把三维度分数线性加权（权重在实验中调优），得到每条指令的综合得分。直接取得分最高的前 6,000 条，形成 SFT 数据集。

**步骤 6：指令微调（SFT）**  
使用标准的有监督微调流程：把指令‑响应对拼接成 “<instruction> … <response>” 的序列，使用交叉熵损失在 LLaMA‑7B、Mistral‑7B 等基模型上训练数个 epoch。因为数据量极小，训练成本只有几小时。

**步骤 7：直接偏好优化（可选）**  
在 SFT 完成后，作者再收集 10,000 条人类偏好对（每对包含两个模型输出，标记哪一个更好），用 DPO 方法直接最大化偏好概率。DPO 的核心是把偏好当作二分类任务的标签，用对数似然来更新模型。

**巧妙之处**  
- **自洽度量**：不需要外部评审，只让模型在不同训练阶段“自我对比”，省去了大量人工成本。  
- **极简筛选**：没有复杂的搜索或强化学习，直接用分数排序即可，易于复现。  
- **少量高质量数据的威力**：实验表明，6K 条精挑细选的数据足以匹配使用 60K‑100K 条普通指令的数据效果，验证了“质量胜于数量”的假设。

### 实验与效果
- **评测基准**：作者在 MT‑Bench（多任务对话基准）和 AlpacaEval（指令遵循准确率）上进行评估。  
- **对比模型**：与最新的开源对齐模型（如 OpenChat、Vicuna、Mistral‑Instruct）以及使用数十万条指令的 SFT 基线进行比较。  
- **主要结果**：DEITA‑Mistral‑7B 在仅用 6K 条 SFT 数据的情况下，MT‑Bench 得分达到 7.55，接近或略超出使用 60K 条数据的基线；在 AlpacaEval 上取得 90.06% 的准确率，同样领先多数对齐模型。加入 10K 条 DPO 偏好后，两个指标均有小幅提升。  
- **消融实验**：作者分别去掉复杂度、质量或多样性评分进行筛选，发现只用单一维度时性能下降 3‑5%，说明三者协同是关键。  
- **局限性**：论文未在大规模多语言指令上验证，度量函数仍依赖于特定基模型的表现，换模型可能需要重新校准权重。作者也承认，当前的多样性评估仍是粗糙的聚类方式，可能遗漏细粒度的任务差异。

### 影响与延伸思考
这篇工作在“数据效率”方向提供了可操作的工具链，促使后续研究更加关注指令数据的质量而非盲目堆砌。随后出现的几篇论文（如 **Data‑Efficient Alignment via Curriculum Sampling**、**Self‑Supervised Instruction Scoring**）都借鉴了 Evol‑Complexity 的自洽评估思路，尝试把度量嵌入训练循环，实现动态数据采样。对想进一步探索的读者，可以关注以下方向：① 将多语言或多模态指令纳入同样的三维度评估框架；② 用强化学习或贝叶斯优化自动调节三维度权重；③ 探索更细粒度的多样性度量（如任务图谱或知识图谱对齐）。这些都可能把“少量高质量数据”推向更广阔的应用场景。

### 一句话记住它
只要用模型自己在不同训练阶段的表现来衡量指令的复杂度、质量和多样性，挑出最好的几千条，就能用极少的数据实现与大规模对齐模型相当的效果。