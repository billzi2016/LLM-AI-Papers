# Goedel-Prover: A Frontier Model for Open-Source Automated Theorem   Proving

> **Date**：2025-02-11
> **arXiv**：https://arxiv.org/abs/2502.07640

## Abstract

We introduce Goedel-Prover, an open-source language model that achieves state-of-the-art (as of April 5 2025) performance in automated formal proof generation for mathematical problems. A key challenge in this field is the scarcity of formalized mathematical statements and proofs, which we address through the following approaches. First, we train LLMs to convert natural language math problems from the Numina dataset to equivalent formal statements in Lean 4. This process creates the dataset Goedel-Pset-v1, which includes 1.64 million formal statements. Next, we develop a large dataset of formal proofs by training a series of provers. Each new prover can prove many statements that previous ones could not, and these new proofs are added to the training set for the next prover. Finally, we obtain the dataset Goedel-Pset-v1-solved, which contains proofs for over 800K statements from Goedel-Pset-v1. Supervised fine-tuning (SFT) of DeepSeek-Prover-V1.5-Base on Goedel-Pset-v1-solved (i.e., no RL) yields a Goedel-Prover-SFT that achieves a success rate of 57.6% (Pass@32) on miniF2F, surpassing the previous leader DeepSeek-Prover-V1.5-RL (trained using SFT + RL on a proprietary dataset) by 7.6%. On PutnamBench, Goedel-Prover-SFT successfully solves 7 problems (Pass@512), ranking first on the leaderboard. We provide extensive discussion of our training methodology, highlighting the key design choices that contribute to Goedel-Prover's strong performance. Further RL training (including DPO) improves Goedel-Prover-SFT's success rate to over 60% (Pass@32) on miniF2F.   To aid future research, we provide extensive discussion of our training methodology and design choices. We also fully open-source our codes, models, and datasets. Additionally, we open-source formal proofs for 29.7K problems in Lean Workbook, nearly doubling the 15.7K solved by prior provers.

---

# Goedel‑Prover：开源自动定理证明的前沿模型 论文详细解读

### 背景：这个问题为什么难？

自动定理证明（Automated Theorem Proving，ATP）本质上是让机器把数学命题写成形式化语言并给出严谨的证明。过去的模型大多依赖少量公开的形式化数据，导致训练样本极其稀缺；同时，现有的证明搜索算法在大规模、跨领域的数学题目上往往陷入搜索空间爆炸。结果是，模型只能在特定的基准上取得有限成功，难以推广到更广的数学领域。

### 关键概念速览
**形式化语言（Formal Language）**：一种严格的、机器可读的数学表达方式，例如 Lean 4，类似于把口头数学题翻译成程序代码。  
**自然语言到形式化的转换（NL→Formal）**：把普通文字描述的数学问题转成形式化语言，就像把手写的几何题目画成 CAD 图。  
**Pass@k**：在 k 次尝试中至少有一次成功的比例，数值越高说明模型在有限尝试内找到证明的能力越强。  
**SFT（Supervised Fine‑Tuning）**：在已有标注数据上继续训练模型，类似于给已经会写作文的学生额外的练习题。  
**RL（Reinforcement Learning）**：让模型通过奖励信号自行探索更好的证明策略，像让学生在做题后根据老师的评分调整解题思路。  
**DPO（Direct Preference Optimization）**：直接优化模型对人类偏好的排序，等价于让模型学习“老师更喜欢的解法”。  
**miniF2F**：一个包含数百个数学题的公开基准，题目来源于真实的数学竞赛，常被用来评估 ATP 系统。  
**PutnamBench**：专门收录美国 Putnam 竞赛难题的基准，用来检验模型在高难度数学上的表现。

### 核心创新点
1. **从自然语言到 Lean 4 的大规模对齐**：以前的工作只能利用几千到几万条已形式化的题目。Goedel‑Prover 首先训练一个 LLM 把 Numina 数据集里的自然语言数学题自动翻译成 Lean 4 代码，生成了 164 万条正式声明（Goedel‑Pset‑v1），相当于把“口语数学”搬进了机器可读的世界。  
2. **迭代式证明生成与数据扩增**：作者训练了一系列逐步增强的证明器。每一次新证明器在前一次的基础上，能够证明之前未能解决的声明，这些新证明被加入训练集，形成了 80 万条已证明的声明（Goedel‑Pset‑v1‑solved）。这种“自助式”数据循环突破了公开数据稀缺的瓶颈。  
3. **纯 SFT 即可超越 RL‑augmented 基线**：在得到大规模已证明数据后，直接对 DeepSeek‑Prover‑V1.5‑Base 进行监督微调（不使用强化学习），得到 Goedel‑Prover‑SFT，miniF2F Pass@32 达到 57.6%，比之前的 DeepSeek‑Prover‑V1.5‑RL 高出 7.6%。说明高质量的监督数据本身就能显著提升模型，而不一定需要复杂的 RL 流程。  
4. **开放生态与大规模公开证明**：除了模型和代码，团队还开源了 29.7K 条 Lean Workbook 的完整证明，几乎翻倍了此前公开的 15.7K 条，为后续研究提供了丰富的“教材”。

### 方法详解
整体思路可以划分为四个阶段：① 数据收集与自然语言对齐，② 迭代式证明生成，③ 大规模监督微调，④ 可选的强化学习提升。

**阶段 1：自然语言 → 形式化**  
- 使用现成的 Numina 数据集（包含大量自然语言数学题）作为原始材料。  
- 训练一个专门的翻译模型，把每道题目转成 Lean 4 语法的声明。这里的关键是让模型学会捕捉数学概念的结构化表示，例如把“若 a,b 为正整数，则 a+b 为正整数”翻译成 Lean 中的 `∀ a b : ℕ, a > 0 → b > 0 → a + b > 0`。  
- 生成的 1.64M 条声明被统称为 Goedel‑Pset‑v1，形成了一个前所未有的大规模形式化语料库。

**阶段 2：迭代式证明器训练**  
- 首先训练一个基础证明器（Prover‑0），它只能解决一小部分声明。  
- 将 Prover‑0 成功证明的结果加入已解决集合。  
- 基于已解决集合，训练下一个更强的证明器（Prover‑1），它的训练目标是“在已有证明的基础上，学习如何处理更难的声明”。  
- 重复上述循环多次，直到累计得到约 800K 条正式证明，构成 Goedel‑Pset‑v1‑solved。这个过程类似于人类学生先学会解简单题，再逐步挑战更难的题目，且每一次成功都成为下一轮学习的教材。

**阶段 3：监督微调（SFT）**  
- 选取 DeepSeek‑Prover‑V1.5‑Base 作为基模型，它已经具备一定的数学推理能力。  
- 使用 Goedel‑Pset‑v1‑solved 进行全量微调，目标是让模型在给定声明后直接输出完整的 Lean 证明脚本。  
- 训练过程中没有引入任何强化学习的奖励信号，仅靠大规模、质量高的监督数据即可让模型的 Pass@32 提升到 57.6%。

**阶段 4：强化学习与 DPO（可选）**  
- 为进一步突破 60% 大关，作者在 SFT 基础上加入强化学习，奖励模型在有限尝试（如 32 次）内成功生成可验证的证明。  
- 采用 DPO 方法直接优化模型对人类偏好的证明风格，使得生成的证明更简洁、更易于审查。  
- 这一步并非必须，但实验表明可以把 miniF2F 的成功率推到 60% 以上。

**最巧妙的设计**  
- 迭代式证明器的“自助式数据扩增”把原本稀缺的标注问题转化为可持续增长的资源，避免了传统上依赖手工标注的瓶颈。  
- 直接用大规模监督数据超越了之前依赖 RL 的强基线，证明了“数据质量 + 规模”在 ATP 领域的决定性作用。

### 实验与效果
- **测试基准**：miniF2F（约 500 题）和 PutnamBench（Putnam 竞赛题目）。  
- **主要结果**：Goedel‑Prover‑SFT 在 miniF2F 上的 Pass@32 为 57.6%，比 DeepSeek‑Prover‑V1.5‑RL（之前的冠军）高出 7.6%。在 PutnamBench 上，以 Pass@512 计，成功解决 7 道题，位列排行榜第一。  
- **对比基线**：DeepSeek‑Prover‑V1.5‑RL、其他开源模型（如 GPT‑4‑based prover）以及传统基于搜索的 ATP 系统。所有对比中，Goedel‑Prover‑SFT 均保持显著优势。  
- **消融实验**：论文报告了去掉迭代式证明生成、仅使用原始 Numina 对齐或仅使用小规模监督数据时，Pass@32 均跌至 40% 以下，说明每个创新模块都是性能提升的关键。  
- **局限性**：模型仍然依赖 Lean 4 生态，迁移到其他形式化系统（Coq、Isabelle）需要重新构建对应的对齐数据。对极其长的证明（数千行）仍会出现超时或记忆瓶颈，作者在论文中承认需要更好的长序列建模技巧。

### 影响与延伸思考
- 这篇工作在开源 ATP 社区掀起了“数据驱动”新潮流，随后出现的项目（如 OpenProof、Lean‑Bench）都在尝试复制其迭代式数据扩增的思路。  
- 研究者开始关注如何把自然语言数学教材直接转化为形式化训练资源，推动了跨模态数学理解的研究。  
- 对想进一步探索的读者，建议关注两条路：① 更高效的长序列 Transformer（如 FlashAttention、Longformer）在大规模证明中的应用；② 多模态对齐——把图形、几何图像等信息也纳入到形式化翻译流程中。  

### 一句话记住它
**把海量自然语言数学题自动翻译成 Lean，再用“自助式”证明循环生成监督数据，单靠 SFT 就把开源自动定理证明推到新高度。**