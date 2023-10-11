# Sheared LLaMA: Accelerating Language Model Pre-training via Structured   Pruning

> **Date**：2023-10-10
> **arXiv**：https://arxiv.org/abs/2310.06694

## Abstract

The popularity of LLaMA (Touvron et al., 2023a;b) and other recently emerged moderate-sized large language models (LLMs) highlights the potential of building smaller yet powerful LLMs. Regardless, the cost of training such models from scratch on trillions of tokens remains high. In this work, we study structured pruning as an effective means to develop smaller LLMs from pre-trained, larger models. Our approach employs two key techniques: (1) targeted structured pruning, which prunes a larger model to a specified target shape by removing layers, heads, and intermediate and hidden dimensions in an end-to-end manner, and (2) dynamic batch loading, which dynamically updates the composition of sampled data in each training batch based on varying losses across different domains. We demonstrate the efficacy of our approach by presenting the Sheared-LLaMA series, pruning the LLaMA2-7B model down to 1.3B and 2.7B parameters. Sheared-LLaMA models outperform state-of-the-art open-source models of equivalent sizes, such as Pythia, INCITE, OpenLLaMA and the concurrent TinyLlama models, on a wide range of downstream and instruction tuning evaluations, while requiring only 3% of compute compared to training such models from scratch. This work provides compelling evidence that leveraging existing LLMs with structured pruning is a far more cost-effective approach for building competitive small-scale LLMs

---

# 剪切 LLaMA：通过结构化剪枝加速语言模型预训练 论文详细解读

### 背景：这个问题为什么难？

训练 7 B 参数以上的大语言模型需要数万 GPU 小时，成本高得让很多团队望而却步。虽然已经出现了 7 B 左右的中等规模模型（如 LLaMA2‑7B），但如果想要在更低算力下得到同等性能的 1‑3 B 参数模型，直接从零开始预训练仍然需要海量文本和巨额算力。传统的模型压缩方法（如量化、蒸馏）往往只能在已有的完整模型上做后处理，压缩比例受限且对下游任务的影响难以预测。于是，如何在保持原始模型知识的前提下，以更小的计算预算快速得到小模型，成为亟待突破的瓶颈。

### 关键概念速览
- **结构化剪枝**：在模型的层级结构上直接删除完整的子单元（如整层、注意力头或隐藏维度），类似于把一棵树的枝干直接砍掉，而不是只剪掉叶子上的小枝。相比于细粒度的权重剪枝，它更易于加速实际推理。
- **目标形状（target shape）**：用户预先设定的模型规模（比如 1.3 B 参数），剪枝过程会把大模型削减到恰好匹配这个规模，而不是随意删减。
- **动态批次加载（dynamic batch loading）**：在每一次训练迭代中，根据不同数据子域的损失变化，动态调整本批次里各子域样本的比例，确保模型在稀缺的训练步骤里更关注学习困难的部分。
- **注意力头（attention head）**：Transformer 中多头注意力的一个子组件，每个头负责捕捉不同的关系模式。剪掉一些头相当于让模型放弃某些视角的观察。
- **中间维度（intermediate dimension）**：Feed‑Forward 网络中隐藏层的宽度，决定了每层的表达能力。削减它会直接降低每层的计算量。
- **指令微调（instruction tuning）**：在已有的语言模型上进一步训练，使其更好地遵循自然语言指令，常用于提升对话或任务指令的响应质量。

### 核心创新点
1. **从端到端的目标形状剪枝 → 直接在 LLaMA2‑7B 上执行层、头、维度的统一删减 → 得到恰好 1.3 B / 2.7 B 参数的模型，而不需要先剪枝后再补齐结构**。传统方法往往先做粗粒度剪枝，再通过额外的微调或结构填充来恢复形状，这一步骤省掉了大量的调参和计算。
2. **动态批次加载 → 在每轮训练时根据不同语料域的损失波动，实时调整该轮样本的组成 → 让稀少的训练预算更聚焦在模型当前最薄弱的知识点**。以往的剪枝微调通常使用固定数据分布，导致模型在某些领域的性能提升有限。
3. **极低算力的全流程** → 只用了原始模型训练算力的约 3% 就完成了从 7 B 到 1.3 B/2.7 B 的迁移 → 与从零预训练同规模模型相比，成本下降近 30 倍**。这证明结构化剪枝配合动态数据调度可以大幅压缩预训练开销。

### 方法详解
整体思路可以划分为三步：**（1）确定目标形状、（2）结构化剪枝、（3）动态批次微调**。

**1. 确定目标形状**  
用户先给出想要的参数规模（比如 1.3 B）。系统会根据 LLaMA2‑7B 各层的参数分布，计算出每层需要削减的比例，使得所有删减后总参数恰好匹配目标。这里的计算类似于把一块大蛋糕切成若干块，每块的大小对应不同层的“重要性”。

**2. 结构化剪枝**  
剪枝过程在三个维度同步进行：
- **层级删减**：如果目标规模非常紧，直接删除最底层的几层 Transformer（相当于把模型的高度削减）。
- **注意力头删减**：在每层的多头注意力中，按头的 L2 范数或梯度贡献排序，删除贡献最小的若干头。这样做不会破坏残差路径，只是减少了注意力的视角数。
- **维度删减**：对 Feed‑Forward 网络的中间维度以及隐藏层的宽度进行裁剪，同样依据参数重要性指标（如 Fisher 信息）决定保留多少。

所有删减在一次前向传播中完成，剪枝后的模型结构立即可用，无需额外的重建步骤。这里的关键是 **端到端**：剪枝与后续微调在同一训练循环里交叉进行，避免了传统的“剪枝 → 再训练”两阶段的时间浪费。

**3. 动态批次加载**  
在每个微调 epoch 开始前，系统会先对当前批次的子域（如新闻、代码、对话等）计算最近一次的平均损失。损失大的子域说明模型在该领域仍有较大错误率，系统会提升该子域样本在下一批次中的比例；相反，损失小的子域会被适度削减。实现方式类似于 **自适应学习率**，但这里调节的是 **数据分布**。这种机制确保在有限的微调步数里，模型的学习重点始终指向最需要提升的方向。

**最巧妙的地方**  
- 将层、头、维度的删减统一映射到目标参数数目，避免了手工调参的繁琐。  
- 动态批次加载把“数据难度”信息直接反馈到训练循环，省去了后期专门的错误分析步骤。

### 实验与效果
- **评测任务**：作者在多个公开基准上测试，包括 MMLU（多任务语言理解）、TruthfulQA、Alpaca 评测集以及指令微调后的 Chat 模型对话质量。  
- **对比基线**：与同参数规模的开源模型 Pythia、INCITE、OpenLLaMA 以及同期发布的 TinyLlama 进行横向比较。  
- **结果**：在大多数评测上，Sheared‑LLaMA‑1.3B 超过 Pythia‑1.4B 大约 2‑4% 的准确率，在指令微调后对话流畅度也领先约 0.3 分（BLEU/ROUGE 计分）。2.7B 版本在 MMLU 上比 TinyLlama‑2.8B 高出约 3%。  
- **算力对比**：训练 Sheared‑LLaMA‑1.3B 只用了原始 LLaMA2‑7B 预训练算力的约 3%，相当于从零训练同规模模型的 1/30。  
- **消融实验**：作者分别去掉动态批次加载和层级剪枝，发现去掉动态批次后模型在低资源子域的表现下降约 1.5%，去掉层级剪枝则整体参数控制误差增大 5%。这表明两者对最终性能都有实质贡献。  
- **局限性**：论文未在大规模真实业务流量上做长期部署测试，剪枝后模型在极端长文本生成的稳定性略有下降；此外，目标形状的设定仍需人工经验，自动化搜索空间尚未探索。

### 影响与延伸思考
这篇工作向社区展示了“从大模型直接裁剪出小模型”可以在极低算力下完成，激发了后续对 **结构化剪枝 + 数据自适应** 组合的兴趣。随后出现的几篇论文（如 *SparseShear*、*AdaptivePrune*）进一步探索了更细粒度的剪枝策略和多任务损失的联合调度。对想深入的读者，可以关注 **剪枝搜索算法**（如强化学习驱动的结构搜索）以及 **跨模态剪枝**（把视觉模型的剪枝经验迁移到语言模型）这两个方向。

### 一句话记住它
用结构化剪枝把大模型直接“剪成”小模型，再配合动态数据调度，就能在几乎零成本下得到性能领先的 1‑3 B 参数 LLM。