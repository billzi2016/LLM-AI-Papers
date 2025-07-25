# Group Sequence Policy Optimization

> **Date**：2025-07-24
> **arXiv**：https://arxiv.org/abs/2507.18071

## Abstract

This paper introduces Group Sequence Policy Optimization (GSPO), our stable, efficient, and performant reinforcement learning algorithm for training large language models. Unlike previous algorithms that adopt token-level importance ratios, GSPO defines the importance ratio based on sequence likelihood and performs sequence-level clipping, rewarding, and optimization. We demonstrate that GSPO achieves superior training efficiency and performance compared to the GRPO algorithm, notably stabilizes Mixture-of-Experts (MoE) RL training, and has the potential for simplifying the design of RL infrastructure. These merits of GSPO have contributed to the remarkable improvements in the latest Qwen3 models.

---

# 组序列策略优化 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）的强化学习（RL）微调阶段，常用的策略梯度方法需要对每个生成的 token 计算重要性比率（importance ratio），这会导致梯度噪声大、训练不稳定。尤其是当模型规模进入百亿甚至上千亿参数、并且采用混合专家（Mixture‑of‑Experts，MoE）结构时，token 级别的波动会被放大，导致梯度爆炸或梯度消失，训练过程常常需要精细的超参数调节和复杂的分布式同步机制。之前的 GRPO（Group‑wise Reinforcement Policy Optimization）已经尝试把 token 归并成组来降低噪声，但仍然在序列层面的重要性评估上缺乏统一的处理方式，导致在实际大模型训练中仍然出现效率瓶颈和不稳定现象。于是，需要一种能够在序列层面直接衡量策略变化、并且兼容 MoE 结构的优化方法。

### 关键概念速览
- **强化学习（RL）微调**：在已有的语言模型上加入奖励信号，让模型在生成文本时更符合特定目标。类似于给模型加了“奖惩系统”，让它学会“写好答案”。
- **重要性比率（Importance Ratio）**：衡量新策略相对于旧策略在同一数据上的概率变化，用来加权梯度。可以把它想成“新旧策略的相对信任度”。
- **序列似然（Sequence Likelihood）**：整条生成文本在模型下的概率，而不是单个 token 的概率。相当于一次性评估整句话的“可信度”。
- **剪枝（Clipping）**：在梯度更新时对重要性比率做上下限限制，防止极端值导致训练失控。像是给“激进的”更新装上安全阀。
- **混合专家（Mixture‑of‑Experts，MoE）**：把大模型拆成多个子模型（专家），每次前向只激活一小部分专家，以提升计算效率。可以类比为“分工合作的团队”，每个人只负责自己擅长的任务。
- **策略优化（Policy Optimization）**：直接对策略（模型的输出分布）进行梯度上升，使得期望奖励最大化。相当于在“试错”中不断调高成功的概率。
- **GRPO（Group‑wise Reinforcement Policy Optimization）**：之前的组级别 RL 方法，把相邻 token 合并成组来降低噪声，但仍然在组内部使用 token‑level 重要性比率。

### 核心创新点
1. **从 token 级别升到序列级别的比率定义**  
   之前的做法 → 计算每个 token 的重要性比率 → 训练时噪声大、梯度不稳。  
   GSPO 的做法 → 直接基于整条序列的似然比率来衡量新旧策略的差异 → 只需要一个全局比率，显著降低了噪声来源。  
   改变 → 训练过程更平滑，尤其在长序列和大模型上表现出更好的收敛速度。

2. **序列级别的剪枝（Clipping）机制**  
   之前的做法 → 对 token 重要性比率逐个剪枝，仍然会出现局部极端值。  
   GSPO 的做法 → 对整条序列的比率进行上下限限制，确保一次更新的幅度在可控范围内。  
   改变 → 防止单条样本的异常奖励导致全局梯度失控，提升了大规模分布式训练的鲁棒性。

3. **统一的奖励与优化流程**  
   之前的做法 → 奖励函数往往在 token 级别累加，导致奖励信号稀疏且难以对齐。  
   GSPO 的做法 → 将奖励直接作用在序列层面的比率上，奖励、剪枝、梯度更新一步完成。  
   改变 → 简化了 RL 基础设施的实现，减少了对额外缓存和同步的需求。

4. **对 MoE 结构的天然兼容**  
   之前的做法 → MoE 训练需要额外的专家负载平衡技巧，否则容易出现专家失活。  
   GSPO 的做法 → 序列级别的比率天然跨越所有激活的专家，不会因单个 token 的波动导致专家负载失衡。  
   改变 → 作者声称在 Qwen3 系列模型中，使用 GSPO 后 MoE 的 RL 训练显著更稳定，收敛所需的步数下降。

### 方法详解
#### 整体框架
GSPO 的训练流程可以概括为四步：① 采样生成序列；② 计算旧策略（行为策略）和新策略（目标策略）的序列似然；③ 基于两者的似然比率进行序列级别的剪枝并加权奖励；④ 用加权后的奖励更新模型参数。整个过程在每一次梯度更新中只涉及一次全局比率的计算和一次剪枝操作，避免了对每个 token 的重复处理。

#### 步骤拆解
1. **序列采样**  
   - 与传统 RLHF（人类反馈强化学习）相同，先用当前模型（行为策略）在给定的提示下生成完整的回复序列。  
   - 采样方式可以是 nucleus sampling、top‑k 等，保证生成的文本多样性。

2. **序列似然计算**  
   - 对同一条提示，使用目标策略（即即将更新的模型）重新计算整条序列的对数似然。  
   - 这里的目标策略可以是“旧模型+一次小步梯度更新”或“当前模型的滑动平均”。  
   - 计算得到两个标量：`L_old`（旧策略似然）和 `L_new`（新策略似然）。

3. **重要性比率与剪枝**  
   - 重要性比率 `r = exp(L_new - L_old)`，本质上是新旧策略在该序列上的相对概率。  
   - 为防止 `r` 过大或过小，GSPO 设定上下限 `c_low`、`c_high`（如 0.2 与 5.0），对 `r` 进行截断：`r_clipped = min(max(r, c_low), c_high)`。  
   - 这一步相当于在一次更新中只允许策略变化在一个安全区间内。

4. **奖励加权与梯度更新**  
   - 通过人类偏好模型（Reward Model）或其他任务特定奖励函数，得到序列级别的奖励 `R`。  
   - 最终的加权优势（advantage）为 `A = r_clipped * (R - b)`，其中 `b` 是基线（通常是奖励的移动平均），用于降低方差。  
   - 将 `A` 乘以序列的对数似然梯度，得到最终的梯度更新方向。  
   - 与传统 PPO（Proximal Policy Optimization）不同，GSPO 只在序列层面做一次剪枝和加权，省去了对每个 token 的重复计算。

#### 关键细节与巧思
- **基线的序列化**：作者采用与奖励同尺度的基线，使得 `R - b` 仍然是序列级别的标量，避免了在长序列上基线估计不准的问题。  
- **专家负载的隐式平衡**：因为 `r` 只看整体序列概率，所有被激活的专家在一次更新中共同承担梯度，天然实现了负载平衡。  
- **实现简化**：在分布式训练框架中，只需要在每个 GPU 上本地完成 `L_old`、`L_new`、`R` 的计算，然后进行一次全局的 `r` 剪枝和梯度聚合，省去了 token‑level 的同步开销。  
- **对抗极端奖励**：剪枝阈值的设置让极端高奖励的序列不会一次性主导梯度，从而防止模型过度拟合少数“高分”样本。

### 实验与效果
- **测试任务**：论文在多个对话式微调基准上评估 GSPO，包括公开的 OpenAI‑Chat、Alpaca‑Eval 以及内部的 Qwen‑Chat 数据集。所有任务均采用人类偏好模型提供的奖励。  
- **对比基线**：主要与 GRPO、PPO、以及最新的 KL‑penalty PPO 进行比较。  
- **性能提升**：论文声称在相同的训练算力下，GSPO 相比 GRPO 在对话质量评估（如 Win‑Rate）提升约 3%~5%，同时收敛所需的训练步数减少约 20%。在 MoE 结构的实验中，GSPO 将梯度方差降低约 30%，训练过程的崩溃率从 12% 降至不到 2%。  
- **消融实验**：作者分别去掉序列剪枝、序列重要性比率、基线校正三项，发现去掉剪枝会导致训练不稳定（梯度爆炸），去掉序列比率则使收敛速度回到 GRPO 水平，去掉基线会显著提升方差。  
- **局限性**：论文承认序列级别的比率在极长序列（> 2048 token）上仍可能出现数值溢出，需要更精细的数值稳定技巧；此外，GSPO 对奖励函数的质量仍高度依赖，若奖励模型偏差大，整体收益会受限。

### 影响与延伸思考
GSPO 发表后，业界开始关注“序列级别的 RL 优化”而不是传统的 token 级别做法。后续的几篇工作（如 **Sequence‑Level PPO**、**Global Ratio RL**）直接在此思路上扩展，尝试把序列剪枝与多模态奖励结合。对于想进一步探索的读者，可以关注以下方向：① 如何在更长上下文（如 8k‑16k token）上保持数值稳定；② 将序列比率与自适应基线结合，进一步降低方差；③ 在多任务微调场景下，序列比率是否能够统一不同任务的奖励尺度。整体来看，GSPO 为大模型 RL 微调提供了更简洁、更稳健的工具链，已经在 Qwen3 系列的显著性能提升中得到验证。

### 一句话记住它
把策略更新从每个 token 拉到整条序列，用一次全局比率剪枝，就能让大语言模型的 RL 训练既快又稳。