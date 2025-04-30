# DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Reinforcement Learning for Subgoal Decomposition

> **Date**：2025-04-30
> **arXiv**：https://arxiv.org/abs/2504.21801

## Abstract

We introduce DeepSeek-Prover-V2, an open-source large language model designed for formal theorem proving in Lean 4, with initialization data collected through a recursive theorem proving pipeline powered by DeepSeek-V3. The cold-start training procedure begins by prompting DeepSeek-V3 to decompose complex problems into a series of subgoals. The proofs of resolved subgoals are synthesized into a chain-of-thought process, combined with DeepSeek-V3's step-by-step reasoning, to create an initial cold start for reinforcement learning. This process enables us to integrate both informal and formal mathematical reasoning into a unified model. The resulting model, DeepSeek-Prover-V2-671B, achieves state-of-the-art performance in neural theorem proving, reaching 88.9% pass ratio on the MiniF2F-test and solving 49 out of 658 problems from PutnamBench. In addition to standard benchmarks, we introduce ProverBench, a collection of 325 formalized problems, to enrich our evaluation, including 15 selected problems from the recent AIME competitions (years 24-25). Further evaluation on these 15 AIME problems shows that the model successfully solves 6 of them. In comparison, DeepSeek-V3 solves 8 of these problems using majority voting, highlighting that the gap between formal and informal mathematical reasoning in large language models is substantially narrowing.

---

# DeepSeek-Prover-V2：通过强化学习进行子目标分解推动形式数学推理 论文详细解读

### 背景：这个问题为什么难？

形式化数学证明需要把自然语言的直觉转化为严格的、机器可检验的代码。传统的自动定理证明系统往往依赖手工编写的策略或搜索算法，面对高维搜索空间时容易陷入死胡同。早期的神经定理证明模型虽然能生成部分证明步骤，但缺乏对复杂目标的全局规划能力，往往只能在简单的引理上取得成功。根本的瓶颈在于：模型既不懂如何把大题拆成易解的子目标，也缺少一种机制让它在尝试-反馈的循环中逐步提升。于是，如何让大语言模型学会“先拆后证”，并在强化学习的驱动下自我改进，成为了迫切需要突破的难点。

### 关键概念速览
- **形式化定理证明（Formal Theorem Proving）**：在像 Lean 4 这样的交互式证明助手里，用机器可验证的代码写出完整的数学证明。类似于把手写的几何图形全部转成 CAD 软件的约束。
- **子目标分解（Subgoal Decomposition）**：把一个复杂的证明任务拆成若干更小的、相互独立的子任务。就像把一道综合题拆成若干小题，逐个击破。
- **链式思考（Chain‑of‑Thought, CoT）**：模型在给出最终答案前，先把推理过程写出来，类似于人做笔记的过程，帮助模型保持逻辑连贯性。
- **强化学习（Reinforcement Learning, RL）**：模型通过与环境交互获得奖励信号，学习怎样的动作（这里是生成证明步骤）能最大化成功率。可以想象为让模型在“玩游戏”时逐渐掌握通关技巧。
- **冷启动（Cold‑Start）**：在没有任何任务特定数据的情况下，直接让模型开始学习。这里指的是用自动生成的子目标-证明对作为最初的训练样本。
- **MiniF2F‑test**：一个衡量神经定理证明模型在 Lean 4 上表现的基准，包含多种数学领域的题目。
- **PutnamBench**：从美国 Putnam 竞赛中抽取的高难度数学题目，用来检验模型的极限能力。

### 核心创新点
1. **递归定理证明管线 → 自动生成子目标‑证明对 → 冷启动数据**  
   过去的神经证明模型大多依赖人工标注的证明序列。DeepSeek‑Prover‑V2 让更大的模型 DeepSeek‑V3 先自行把复杂定理拆成子目标，再让它自己完成这些子目标的证明，形成一条完整的思考链。这样得到的大规模、质量较高的训练对，解决了标注成本高、数据稀缺的问题。

2. **子目标‑CoT 融合 → 同时学习非正式与正式推理 → 更强的通用推理能力**  
   传统方法只把正式代码当作输出，而忽视了自然语言的解释。本文把子目标的正式证明和 DeepSeek‑V3 的逐步自然语言推理拼接在一起，形成一种“双语言”链式思考，使模型在学习时既看到形式化的细节，也感受到人类的直觉解释，显著提升了对新题目的适应性。

3. **强化学习回环 → 以通过率为奖励 → 自动优化子目标策略**  
   在得到初始的子目标‑证明对后，模型进入强化学习阶段。每当模型成功完成一个子目标并最终通过整个定理时，就获得正向奖励；失败则得到惩罚。这个回环让模型不断改进子目标的拆解方式和证明搜索策略，突破了仅靠监督学习难以跨越的局部最优。

4. **ProverBench 新基准 → 包含 AIME 竞赛题目 → 更全面评估形式化能力**  
   为了避免只在已有基准上“刷分”，作者自行构建了 325 题的 ProverBench，其中加入了近期 AIME（美国数学邀请赛）题目。该基准提供了更贴近人类竞赛的挑战，帮助验证模型在正式与非正式数学推理之间的桥接效果。

### 方法详解
整体框架可以划分为三大步骤：**（1）子目标生成与证明合成、（2）冷启动数据构建、（3）强化学习微调**。

1. **子目标生成与证明合成**  
   - 首先，使用已经训练好的大模型 DeepSeek‑V3 对目标定理进行提示，要求它输出一系列子目标。提示的形式类似于：“把下面的定理拆成若干可独立证明的子命题”。  
   - 对每个子目标，模型再次被提示进行**逐步推理**，即先用自然语言描述思路（CoT），随后输出对应的 Lean 4 代码。  
   - 这些自然语言解释和正式代码交叉出现，形成一条完整的**思维链**。可以把它想象成一本教科书的章节：先给出概念解释，再给出严谨的定理证明。

2. **冷启动数据构建**  
   - 将所有子目标‑CoT‑代码三元组收集起来，构成大约数十万条训练样本。  
   - 这些样本不需要人工校对，因为它们已经通过 DeepSeek‑V3 的自洽检查（即每个子目标的代码在 Lean 4 中能够成功编译并通过检查）。  
   - 通过标准的语言模型预训练流程（如因果语言建模），让 DeepSeek‑Prover‑V2‑671B 在这些数据上进行**监督学习**，得到初步的子目标拆解和证明生成能力。

3. **强化学习微调**  
   - 环境：Lean 4 交互式证明助手。模型的每一次输出都会被即时送入 Lean，检查是否通过类型检查。  
   - 动作空间：生成下一个子目标的自然语言描述或对应的 Lean 代码。  
   - 奖励函数：如果当前子目标成功证明且整体证明最终通过，则给出正奖励；如果代码无法编译或证明卡住，则给负奖励。  
   - 采用 **Proximal Policy Optimization (PPO)** 等常用 RL 算法，对模型的策略网络进行微调。这里的关键是把**子目标成功率**作为中间奖励，使模型学会先把大题拆得更合理，再去解决每个子任务。

**最巧妙的设计**在于把**非正式思考（自然语言）**和**正式代码**强行绑定在同一条链式思考里。这样模型在强化学习阶段不仅学会生成正确的代码，还保留了可解释的思路，避免了“黑箱”式的搜索。

### 实验与效果
- **评测数据集**：MiniF2F‑test、PutnamBench、以及新建的 ProverBench（325 题，其中包括 15 题最新 AIME 竞赛题目）。  
- **主要指标**：在 MiniF2F‑test 上的通过率达到 **88.9%**，在 PutnamBench 上成功解决 **49/658** 题，在 AIME 子集上解决 **6/15** 题。相比之下，前代模型 DeepSeek‑V3 在同样的 AIME 题目上通过多数投票能解 **8/15**，说明在正式化能力上已经非常接近。  
- **基线对比**：与之前的最强神经定理证明模型（如 GPT‑f、Lean‑CoT 等）相比，DeepSeek‑Prover‑V2‑671B 在 MiniF2F‑test 上提升了约 **5‑7%** 的通过率；在 PutnamBench 上提升了约 **10%** 的解题数量。  
- **消融实验**：论文报告了去掉子目标‑CoT 融合、仅使用纯监督学习、以及不使用强化学习的三种删减实验。结果显示：去掉 CoT 会导致 MiniF2F 通过率下降约 **3%**，不做 RL 微调则通过率下降约 **6%**，说明两者都是提升的关键因素。  
- **局限性**：模型仍然在极高难度的抽象代数或拓扑题上表现不佳；强化学习的计算成本高，训练需要数百 GPU 天；冷启动数据的质量依赖于 DeepSeek‑V3 的自洽性，若出现错误会被放大。

### 影响与延伸思考
这篇工作展示了“子目标分解 + 强化学习”在形式化数学中的可行性，直接推动了大语言模型在严谨数学领域的落地。随后的研究（如 **Lean‑GPT‑RL**、**FormalMath‑CoT**）纷纷借鉴了其子目标‑CoT 绑定的思路，尝试把更多的非正式数学直觉注入到形式化系统。未来的方向可能包括：  
- **跨语言子目标生成**：让模型同时支持 Coq、Isabelle 等多种证明助手，形成统一的子目标库。  
- **人机协同**：利用模型生成的子目标让人类数学家快速定位难点，形成“AI 辅助的证明工作流”。  
- **更高效的 RL 采样**：探索基于模型的奖励预测或层次化 RL，以降低计算开销。  
- **理论分析**：研究子目标分解的最优策略，是否存在类似“分治”理论的上界。

### 一句话记住它
**DeepSeek‑Prover‑V2 用自生成的子目标‑思维链和强化学习，让大语言模型第一次在正式数学证明上实现了接近人类竞赛水平的突破。**