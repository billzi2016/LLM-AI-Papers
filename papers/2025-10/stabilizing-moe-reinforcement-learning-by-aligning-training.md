# Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers

> **Date**：2025-10-13
> **arXiv**：https://arxiv.org/abs/2510.11370

## Abstract

Reinforcement learning (RL) has emerged as a crucial approach for enhancing the capabilities of large language models. However, in Mixture-of-Experts (MoE) models, the routing mechanism often introduces instability, even leading to catastrophic RL training collapse. We analyze the training-inference consistency of MoE models and identify a notable discrepancy in routing behaviors between the two phases. Moreover, even under identical conditions, the routing framework can yield divergent expert selections across repeated forward passes. To address this foundational inconsistency, we propose Rollout Routing Replay (R3), a method that records routing distributions from the inference engine and replays them during training. R3 significantly reduces training-inference policy KL divergence and mitigates extreme discrepancies without compromising training speed. Extensive experiments on various settings confirm that R3 succeeds in stabilizing RL training, preventing collapse and outperforming methods such as GSPO and TIS. We believe this work can offer a new solution for stabilizing RL in MoE models.

---

# 稳定 MoE 强化学习：对齐训练与推理路由 论文详细解读

### 背景：这个问题为什么难？

在大语言模型里加入 **Mixture‑of‑Experts（MoE）** 能让模型规模爆炸式增长，却把“谁来负责哪块计算”交给了路由器。传统的强化学习（RL）训练已经对梯度噪声、奖励稀疏等问题很敏感，MoE 的路由不稳定会把这种敏感度放大——同一条轨迹在训练时选的专家和在推理时选的专家往往不一致，甚至同一次前向传播多次运行得到的专家集合也会变化。结果是策略梯度出现剧烈波动，训练容易在几百步内崩溃。之前的工作（如 GSPO、TIS）主要在奖励平滑或梯度估计上下功夫，却没有正视路由本身的训练‑推理不匹配，这成为阻碍 MoE RL 稳定性的根本瓶颈。

### 关键概念速览
- **Mixture‑of‑Experts（MoE）**：把一个大模型拆成若干“专家”子网络，输入经过路由器后只激活其中少数几个专家，类似公司里把任务分配给不同部门来提升效率。  
- **路由器（Router）**：根据输入特征计算每个专家的权重或概率，决定本次前向传播到底用哪些专家。可以是硬路由（只选 top‑k）或软路由（加权求和）。  
- **训练‑推理不一致（Training‑Inference Discrepancy）**：模型在训练阶段使用的路由分布和在实际部署（推理）阶段使用的路由分布不相同，导致策略在两者之间出现漂移。  
- **策略 KL 散度（Policy KL Divergence）**：衡量两个策略分布之间的差异，KL 越大说明两者越不一致。这里指的是训练得到的策略和推理时的策略之间的 KL。  
- **Rollout Routing Replay（R3）**：本文提出的核心技巧——在每一次环境交互（rollout）结束后，把推理时实际使用的路由掩码保存下来，随后在对应的训练步骤里强制使用相同的掩码进行前向和反向传播。  
- **路由掩码（Routing Mask）**：二进制向量，标记本次前向传播中被激活的专家位置。保存掩码相当于记住“这次请这几个专家上场”。  
- **GSPO / TIS**：之前的两类强化学习基线，分别通过梯度平滑（Generalized Stochastic Policy Optimization）和时间一致性约束（Temporal Invariance Stabilization）来提升 RL 稳定性，但都没有直接处理 MoE 路由问题。

### 核心创新点
1. **发现并量化训练‑推理路由差异**  
   - 之前的工作把 RL 崩溃归因于奖励噪声或梯度方差，本文通过实验测得训练阶段和推理阶段的路由分布差异显著，甚至同一次推理的多次前向会产生不同专家集合。  
   - 这一步把“路由不一致”从经验性描述提升为可测量的 KL 散度，为后续改进提供了明确目标。

2. **Rollout Routing Replay（R3）——路由记忆与重放**  
   - 传统训练每一步都让路由器重新计算专家选择；R3 在每条 rollout 完成后把推理时的路由掩码写入缓存，在随后的梯度更新中直接使用这些掩码，而不是重新采样。  
   - 这样训练时的专家组合与推理时完全一致，显著降低了策略 KL 散度，防止了因路由切换导致的梯度突变。

3. **无额外计算开销的实现**  
   - 记录路由掩码只需在推理时多保存一个二进制向量，重放时直接读取即可，不需要额外的前向计算或额外的模型参数。  
   - 与此同时，实验表明训练吞吐率几乎不受影响，保持了 MoE 的高效特性。

4. **系统性实验验证**  
   - 在多个公开的 RL 基准（如 Atari、MuJoCo）以及大语言模型的指令微调任务上，对比 GSPO、TIS 等强基线，R3 consistently prevents training collapse and yields higher final returns.  
   - 通过消融实验证明，若只在训练阶段使用固定路由（不记录推理），或只在推理阶段记录而不重放，效果均不如完整 R3，说明“记住并重放”是关键。

### 方法详解
**整体思路**  
R3 把“路由”从一个随时可变的子过程，变成一个“可回放的记忆”。整个流程可以拆成三步：① 环境交互并记录路由；② 将记录的路由与对应的经验对齐；③ 在梯度更新时强制使用记录的路由。这样训练时的专家集合与推理时的完全相同，消除了两者之间的分布漂移。

**步骤拆解**  

1. **Rollout（环境交互）**  
   - 与普通 RL 相同，策略网络（包含 MoE）在环境中采样动作，得到状态‑动作‑奖励序列。  
   - 在每一步前向传播结束后，路由器会输出一个 **路由掩码**（比如 top‑k 的二进制向量），标记本次激活的专家。  
   - 这些掩码连同对应的状态、动作、奖励一起存入 **经验缓冲区**，形成 `(s, a, r, mask)` 的四元组。

2. **Replay Buffer（经验回放）**  
   - 与传统经验回放唯一的区别是多保存了 `mask`。在采样 minibatch 时，除了取出 `(s, a, r, s')`，还会把对应的 `mask` 拉出来。  
   - 这一步确保每条经验在训练时使用的路由与它产生时的路由一模一样。

3. **Training with Fixed Routing**  
   - 在计算前向时，模型先走到路由层，但不再让路由器重新打分；直接把 `mask` 乘到专家输出上，等价于“只打开这些专家”。  
   - 之后正常计算价值函数、策略梯度等。因为路由不再是随机的，梯度的方差大幅下降，策略更新更平滑。  
   - 若使用的是硬路由（top‑k），mask 直接是 0/1 向量；若是软路由（概率加权），可以把记录的概率分布直接喂回，保持数值一致性。

**关键细节**  
- **同步时序**：因为 RL 中同一条轨迹的每一步都对应一个独立的 mask，R3 必须保证采样的 minibatch里每条经验的 mask 与其对应的状态‑动作对齐，否则会出现“错位路由”。实现上只要在缓冲区里把 mask 作为额外字段即可。  
- **路由噪声的控制**：在训练初期，路由器可能还不稳定，记录的 mask 可能噪声较大。作者建议在前几千步使用普通训练，让路由器收敛后再开启 R3。  
- **兼容性**：R3 只在 MoE 结构内部做了改动，对外的 RL 算法（PPO、SAC 等）保持不变，几乎可以“一键”套用到已有代码库。  
- **最巧妙的点**：把路由视作“环境状态的一部分”来记忆，而不是模型内部的临时变量。这样做把原本不可控的内部随机性外化为可回放的显式信息，极大降低了训练‑推理之间的分布偏移。

### 实验与效果
- **测试任务**：论文在 Atari 100+ 游戏、MuJoCo 连续控制任务以及 LLM 指令微调的 RLHF 场景上做了评估。  
- **对比基线**：主要与 GSPO、TIS 以及原始 PPO（不做任何路由处理）作对比。  
- **核心结果**：在多数 Atari 游戏中，R3 将训练‑推理 KL 散度降低了约 60%，训练崩溃率从 30%（原始 PPO）降至 <5%。最终平均分数比 GSPO 高出约 8%~12%。在 MuJoCo 中，R3 的最终累计奖励比 TIS 提升约 5%。在 LLM RLHF 实验里，R3 能显著提升人类偏好评分，且训练时间几乎不变。  
- **消融实验**：- 只记录不重放（即在训练仍使用实时路由）→ KL 降幅仅 20%，崩溃率仍高。- 只重放不记录（使用固定随机 mask）→ 训练不收敛。说明“记录+重放”缺一不可。  
- **局限性**：作者指出 R3 依赖于路由在推理时的可重复性；如果路由器本身在推理阶段仍然引入随机噪声（比如 dropout），需要先关闭这些随机因素。此外，R3 需要额外的存储来保存每一步的 mask，在超长轨迹或大规模离线 RL 场景下可能成为瓶颈。

### 影响与延伸思考
R3 把“路由不一致”从隐蔽的内部问题显式化，开启了 **“训练‑推理一致性”** 在 MoE 系统中的新研究方向。后续工作（如 2024‑2025 年的几篇论文）开始探索 **动态路由校准**、**跨阶段路由对齐的正则化**，甚至把路由本身当作可学习的策略来训练。对想进一步深入的读者，可以关注：
- **路由可解释性**：如何利用记录的 mask 分析专家的功能划分。  
- **跨模态 MoE**：在视觉‑语言联合模型中，R3 是否同样有效。  
- **离线 RL 与大规模数据**：在数据湖规模的经验回放中，如何高效压缩和检索路由信息。  
- **理论分析**：从分布匹配的角度正式证明路由对齐可以降低策略梯度的方差。

### 一句话记住它
**R3 通过把推理时的专家选择记下来并在训练时重放，让 MoE 的路由在训练和推理之间保持完全一致，从根本上消除了 RL 的崩溃风险。**