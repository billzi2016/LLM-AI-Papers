# DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for   Reinforcement Learning and Monte-Carlo Tree Search

> **Date**：2024-08-15
> **arXiv**：https://arxiv.org/abs/2408.08152

## Abstract

We introduce DeepSeek-Prover-V1.5, an open-source language model designed for theorem proving in Lean 4, which enhances DeepSeek-Prover-V1 by optimizing both training and inference processes. Pre-trained on DeepSeekMath-Base with specialization in formal mathematical languages, the model undergoes supervised fine-tuning using an enhanced formal theorem proving dataset derived from DeepSeek-Prover-V1. Further refinement is achieved through reinforcement learning from proof assistant feedback (RLPAF). Beyond the single-pass whole-proof generation approach of DeepSeek-Prover-V1, we propose RMaxTS, a variant of Monte-Carlo tree search that employs an intrinsic-reward-driven exploration strategy to generate diverse proof paths. DeepSeek-Prover-V1.5 demonstrates significant improvements over DeepSeek-Prover-V1, achieving new state-of-the-art results on the test set of the high school level miniF2F benchmark ($63.5\%$) and the undergraduate level ProofNet benchmark ($25.3\%$).

---

# DeepSeek-Prover-V1.5：利用证明助理反馈进行强化学习与蒙特卡洛树搜索 论文详细解读

### 背景：这个问题为什么难？

在形式化数学里，让语言模型自动生成 Lean 4 代码完成定理证明一直是高难度任务。早期模型大多只能一次性输出完整证明，若出现细微错误就会导致整个证明崩溃。与此同时，证明助理（如 Lean）本身提供的交互式反馈并没有被充分利用，模型难以在生成过程中纠错或探索多条思路。单纯靠大规模预训练或监督微调，往往在高难度的高校或本科基准上停留在 10% 左右的成功率，远低于人类数学家的水平。

### 关键概念速览
- **Lean 4**：一种交互式证明助理，用户通过写代码来表达数学定义和证明步骤，类似于在电脑上写“可验证的数学笔记”。  
- **RLPAF（Reinforcement Learning from Proof Assistant Feedback）**：把证明助理的成功/失败信号当作奖励，让模型在生成过程中通过强化学习自我改进，类似于让机器人在玩游戏时根据得分来学习策略。  
- **Monte‑Carlo Tree Search（MCTS）**：一种在搜索空间中随机采样并逐步扩展树结构的算法，常用于围棋等复杂决策问题，这里用来探索不同的证明分支。  
- **RMaxTS**：本文提出的 MCTS 变体，利用模型内部的“内在奖励”（如生成的代码是否符合语法、是否接近目标定理）来引导搜索，像在黑暗中用手电筒寻找出口。  
- **miniF2F**：面向高中数学的定理集合，用来评估模型在相对简单但多样化的证明任务上的表现。  
- **ProofNet**：包含本科层次定理的基准，难度更高，检验模型的深层数学推理能力。  
- **SFT（Supervised Fine‑Tuning）**：在已有标注数据上继续训练模型，使其更贴合特定任务的输入输出格式。  
- **Intrinsic Reward**：模型内部生成的奖励信号，例如预测下一个证明步骤的置信度，用来衡量当前分支的潜力。

### 核心创新点
1. **从单次全证明生成到交互式搜索**  
   - 之前的 DeepSeek‑Prover‑V1 只能一次性输出完整证明，错误率高。  
   - 本文引入 RMaxTS，让模型在每一步都可以根据 Lean 的即时反馈决定是否继续、回溯或尝试其他路径。  
   - 结果是搜索过程更稳健，成功率在 miniF2F 上提升到 63.5%，在 ProofNet 上提升到 25.3%。  

2. **强化学习利用证明助理的二元反馈**  
   - 传统强化学习在数学证明中缺少明确的奖励信号。  
   - 通过 RLPAF，把 Lean 给出的“证明成功/失败”作为稀疏奖励，同时结合内在奖励（语法合法性、目标接近度）形成更丰富的学习信号。  
   - 训练后模型在没有搜索时的单步生成质量也有明显提升。  

3. **数据管线的两阶段微调**  
   - 首先在 DeepSeekMath‑Base 上进行大规模预训练，专注于形式化数学语言的语法和常用结构。  
   - 接着使用经过 DeepSeek‑Prover‑V1 生成并人工过滤的高质量定理证明数据进行 SFT，确保模型学习到真实的证明策略。  
   - 这种“先宽后深”的微调方式比一次性直接在 ProofNet 上微调更有效。  

4. **内在奖励驱动的 MCTS 变体**  
   - 标准 MCTS 只依赖外部评估函数，搜索效率受限。  
   - RMaxTS 将模型的自评概率（如下一个 tactic 的置信度）作为内在奖励，帮助搜索在早期就识别出有潜力的分支。  
   - 实验显示，加入内在奖励后搜索深度相同的情况下成功率提升约 8%。  

### 方法详解
整体思路可以分为三大阶段：**预训练 → 监督微调 → 强化学习+搜索**。  
1. **预训练阶段**  
   - 使用 DeepSeekMath‑Base，这是一套包含数千条形式化数学定义、定理和证明脚本的语料库。模型学习 Lean 语法、常见 tactic（如 `simp`, `rw`, `apply`）的使用方式，类似于让学生先熟悉数学符号和常用技巧。  

2. **监督微调（SFT）阶段**  
   - 从 DeepSeek‑Prover‑V1 生成的证明中挑选出通过 Lean 检验的高质量样本，构建正式的定理‑证明对。  
   - 采用标准的指令式微调，让模型在给定定理陈述后输出对应的 Lean 代码序列。此阶段的目标是让模型掌握“从目标到 tactic 序列”的映射。  

3. **强化学习与搜索阶段**  
   - **RLPAF**：在每一次生成过程中，模型提交当前的 tactic 给 Lean，Lean 返回两类信息：① 是否成功完成当前子目标，② 剩余未解决的子目标列表。成功的子目标被视为正奖励，失败或卡死则给负奖励。与此同时，模型内部计算的置信度被当作**内在奖励**加入总奖励。  
   - **RMaxTS**：在搜索树的每个节点，模型先用自身的策略网络预测若干候选 tactic，并用价值网络估计每个 tactic 的潜在成功概率。搜索过程遵循 UCT（上置信界）公式，但把内在奖励直接加到价值估计中，使得高置信度的 tactic 更容易被扩展。搜索结束后，选取累计奖励最高的完整 tactic 序列提交给 Lean。  

4. **关键实现细节**  
   - **策略网络**和**价值网络**共享底层 Transformer 编码器，只在输出层分叉，保持参数经济。  
   - 为了防止搜索过程中过度依赖单一步骤的高置信度，RMaxTS 在每层加入**噪声扰动**，保证探索多样性。  
   - 强化学习使用 **PPO（Proximal Policy Optimization）** 进行梯度更新，保持策略变化的平稳性。  

最巧妙的地方在于把 Lean 的二元反馈（成功/失败）和模型自评的连续奖励融合成一个统一的学习信号，让模型既能从外部纠错，又能利用内部直觉进行前瞻性搜索。

### 实验与效果
- **测试数据**：miniF2F（高中层）和 ProofNet（本科层）两套公开基准。  
- **对比基线**：DeepSeek‑Prover‑V1、GPT‑4‑based 定理证明模型、Lean‑CoT 等。  
- **主要结果**：  
  - 在 miniF2F 上，DeepSeek‑Prover‑V1.5 达到 63.5% 的成功率，较 V1 的 48% 提升约 15%。  
  - 在 ProofNet 上，成功率从 V1 的 17% 提升到 25.3%，提升约 8%。  
- **消融实验**：  
  - 去掉 RLPAF，仅使用 SFT + RMaxTS，miniF2F 成功率下降到 57%。  
  - 去掉内在奖励的 RMaxTS，搜索成功率下降约 6%。  
  - 只用单步生成（无搜索），成功率回落到 V1 水平，说明搜索和强化学习是提升的关键驱动。  
- **局限性**：  
  - 对于极其长的证明（超过 50 步）搜索仍会出现爆炸式分支，计算成本显著上升。  
  - RLPAF 依赖 Lean 的即时反馈，迁移到不提供细粒度反馈的证明系统时需要重新设计奖励函数。  

### 影响与延伸思考
这篇工作把“证明助理反馈”正式纳入强化学习框架，为后续的 **交互式数学 AI** 指明了方向。随后出现的几篇论文（如 *Lean‑GPT‑RL*、*FormalMath‑MCTS*）都在不同程度上借鉴了 RLPAF 与内在奖励驱动的搜索思路。未来的研究可能会：
- 探索 **多模态反馈**（如 proof trace 可视化）来进一步丰富奖励信号。  
- 将 **层次化搜索** 与 **自适应树剪枝** 结合，降低长证明的计算开销。  
- 把这种框架迁移到其他交互式系统（Coq、Isabelle），验证其通用性。  

如果想深入，可以关注 **强化学习在形式化系统中的稀疏奖励设计** 以及 **基于价值网络的证明搜索策略** 两大方向。

### 一句话记住它
把 Lean 的对错反馈当成奖励，让模型在 Monte‑Carlo 树搜索中“自带灯塔”，从而显著提升自动定理证明的成功率。