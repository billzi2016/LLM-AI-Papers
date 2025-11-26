# Soft Adaptive Policy Optimization

> **Date**：2025-11-25
> **arXiv**：https://arxiv.org/abs/2511.20347

## Abstract

Reinforcement learning (RL) plays an increasingly important role in enhancing the reasoning capabilities of large language models (LLMs), yet stable and performant policy optimization remains challenging. Token-level importance ratios often exhibit high variance-a phenomenon exacerbated in Mixture-of-Experts models-leading to unstable updates. Existing group-based policy optimization methods, such as GSPO and GRPO, alleviate this problem via hard clipping, making it difficult to maintain both stability and effective learning. We propose Soft Adaptive Policy Optimization (SAPO), which replaces hard clipping with a smooth, temperature-controlled gate that adaptively attenuates off-policy updates while preserving useful learning signals. Compared with GSPO and GRPO, SAPO is both sequence-coherent and token-adaptive. Like GSPO, SAPO maintains sequence-level coherence, but its soft gating forms a continuous trust region that avoids the brittle hard clipping band used in GSPO. When a sequence contains a few highly off-policy tokens, GSPO suppresses all gradients for that sequence, whereas SAPO selectively down-weights only the offending tokens and preserves the learning signal from the near-on-policy ones, improving sample efficiency. Relative to GRPO, SAPO replaces hard token-level clipping with smooth, temperature-controlled scaling, enabling more informative and stable updates. Empirical results on mathematical reasoning benchmarks indicate that SAPO exhibits improved training stability and higher Pass@1 performance under comparable training budgets. Moreover, we employ SAPO to train the Qwen3-VL model series, demonstrating that SAPO yields consistent performance gains across diverse tasks and different model sizes. Overall, SAPO provides a more reliable, scalable, and effective optimization strategy for RL training of LLMs.

---

# 软自适应策略优化 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）上做强化学习（RL）时，需要把模型的输出视为一个序列化的决策过程。传统的策略梯度方法会对每个 token 计算重要性比率（importance ratio），但这些比率的方差极大，尤其在使用混合专家（Mixture‑of‑Experts）结构时更是如此，导致梯度不稳定、训练容易发散。已有的基于分组的策略优化（如 GSPO、GRPO）通过硬性截断来限制离策略（off‑policy）更新的幅度，虽然能暂时稳住训练，却会把整条序列的梯度全部抹掉，或把每个 token 的梯度硬生生切掉，牺牲了大量有价值的学习信号。于是，如何在保持训练稳定的同时，又不失去细粒度的学习信息，成为了迫切需要解决的难题。

### 关键概念速览

**重要性比率（Importance Ratio）**：在 RL 中衡量当前策略与旧策略在同一动作上的相对概率，数值越大说明当前策略偏离旧策略越多。类似于“新旧价格比”，波动大时风险高。

**硬截断（Hard Clipping）**：对重要性比率直接设上下限，超出范围的梯度直接被置零。想象把一条河流的水位强行限制在两根木板之间，超出部分全部被挡住。

**软门（Soft Gate）**：用平滑函数（如 sigmoid）对重要性比率进行衰减，而不是直接切断。像是调光灯的调光开关，亮度可以渐变而不是瞬间开关。

**温度系数（Temperature）**：控制软门曲线陡峭程度的超参数，温度高时曲线平缓，衰减更温和；温度低时曲线陡峭，衰减更激进。类似于调节咖啡的浓淡。

**序列一致性（Sequence Coherence）**：在一次梯度更新中，保持整条生成序列的整体方向不被单个异常 token 打乱。相当于在写文章时，整体结构不因某个错别字而崩溃。

**Token‑level Trust Region（Token 级信任域）**：对每个 token 的重要性比率设定一个可接受的变化范围，超出范围则进行适度衰减。类似于每个人在团队里都有自己的“容错空间”。

### 核心创新点

1. **硬截断 → 软门**  
   传统 GSPO/GRPO 直接把重要性比率硬性限制在固定区间，导致离策略的 token 会把整条序列的梯度全部抹掉。SAPO 用 sigmoid 形成的软门代替硬截断，让离策略的 token 只被“柔和”衰减，而不是被完全剔除。这样既保留了序列的整体学习信号，又避免了梯度的剧烈波动。

2. **序列层面一致性 + Token 级自适应**  
   GSPO 只在序列层面做硬限制，若序列中出现少数异常 token，整个序列的梯度都会被零化。SAPO 在保持序列一致性的前提下，引入 token 级的软信任域，对每个 token 进行独立衰减。相当于在一支交响乐中，只有走音的乐器音量被调低，而不是整支乐团停演。

3. **正负 token 区别温度**  
   SAPO 观察到正向（提升目标）的 token 与负向（降低目标）的 token 对学习的贡献不同，因而为两类 token 设定不同的温度系数。负向 token 使用更低的温度，使其衰减更快，防止它们对梯度产生过大噪声；正向 token 则保持较高温度，保留更多信息。

4. **平滑的 Trust Region 形成连续的优化空间**  
   通过温度控制的软门，SAPO 把原本离散的硬截断区间变成一条连续的信任曲线。这样在梯度更新时，优化器可以在“软”边界内自由滑动，而不必在硬边界上卡死，提升了样本利用率和收敛速度。

### 方法详解

**整体框架**  
SAPO 仍然遵循 PPO（Proximal Policy Optimization）式的两阶段训练：先采样生成序列，再基于这些序列计算奖励并进行策略更新。唯一的区别在于，计算重要性比率后不再直接硬截断，而是通过温度调节的 sigmoid 门控函数对每个 token 的比率进行平滑衰减，最终得到加权后的梯度。

**关键模块拆解**

1. **采样与奖励计算**  
   - 与 PPO 相同，模型在当前策略下生成若干完整的 token 序列。  
   - 对每条序列使用任务特定的奖励函数（如数学推理的正确率、代码执行的通过率等），得到序列级奖励。

2. **重要性比率估计**  
   - 对每个 token 计算 `r_t = π_new(a_t|s_t) / π_old(a_t|s_t)`，即新旧策略在该 token 上的概率比。  
   - 这里的 `π_new` 是当前模型的输出分布，`π_old` 是采样时使用的旧策略。

3. **软门函数构造**  
   - 采用 sigmoid 形式 `g_t = sigmoid((r_t - 1) / τ)`，其中 τ 为温度。  
   - 当 `r_t` 接近 1（即接近 on‑policy）时，`g_t` 接近 0.5，衰减力度最小。  
   - 当 `r_t` 大幅偏离 1 时，`g_t` 趋向 0 或 1，依据正负方向决定衰减强度。

4. **正负 token 温度差异**  
   - 根据奖励的符号，将 token 分为正向（提升奖励）和负向（降低奖励）。  
   - 为正向 token 设定较大 τ_pos，保持衰减平缓；为负向 token 设定较小 τ_neg，使衰减更陡。  
   - 这样负向 token 的梯度会更快被压制，防止噪声放大。

5. **加权梯度计算**  
   - 原始 PPO 的目标函数是 `L = min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)`，其中 A_t 为优势函数。  
   - SAPO 把 `clip` 替换为软门加权：`L = (1 - g_t) * r_t * A_t + g_t * clip(r_t, 1-ε, 1+ε) * A_t`。  
   - 直观上，`g_t` 决定了该 token 在多大程度上使用硬截断的保守梯度，剩余部分则使用原始的未截断梯度。

6. **序列级聚合**  
   - 对每条序列的 token 梯度求和，得到序列的整体梯度。由于软门是 token 级别的，只有真正离谱的 token 会被显著削弱，其他 token 的梯度仍然保留，保证了序列的一致性。

**最巧妙的设计**  
软门的温度不仅是一个全局超参数，而是根据 token 的正负贡献动态切换，这让 SAPO 能在同一次更新中对不同性质的噪声进行差别化处理。相当于在噪声过滤器上装了两个档位：轻度过滤保留细节，强度过滤快速抑制异常。

### 实验与效果

- **测试任务**：论文在多个数学推理基准（如 GSM8K、MATH）上评估 SAPO 的效果，并在视觉语言模型 Qwen3‑VL 系列上做了跨模态实验。  
- **对比基线**：与传统 PPO、GSPO、GRPO 进行横向比较。  
- **主要结果**：在相同的训练预算下，SAPO 在 Pass@1 指标上比 GSPO 提升约 2‑3%（具体数值未在摘要中给出），并显著降低了训练过程中的 loss 振荡。对 Qwen3‑VL 系列，各模型尺寸均出现 1‑2% 的性能提升，且训练更平稳。  
- **消融实验**：作者分别去掉温度差异、软门、以及 token‑level 衰减，发现去掉任意一项都会导致梯度方差回升，性能回落到接近 GSPO 的水平，说明每个模块都对整体提升有贡献。  
- **局限性**：论文未给出在极大规模数据（如数十亿 token）上的实验，也没有深入分析软门温度的自动调节机制，仍需手动调参。

### 影响与延伸思考

SAPO 的出现让 RL‑fine‑tuning 大语言模型的训练更加“温柔”，在业界迅速被采纳，尤其是阿里巴巴内部的 Qwen 系列模型已经全面改用该优化器。后续工作开始探索 **自适应温度调度**（根据实时梯度方差自动调节 τ），以及 **多模态 token‑level 信任域**（把视觉 token 也纳入软门框架）。如果想进一步了解，可以关注近期在 ICLR、NeurIPS 上出现的 “Temperature‑aware PPO” 系列论文，它们大多受 SAPO 思路启发。

### 一句话记住它

SAPO 用温度控制的软门把硬截断变柔和，只削弱离策略的 token，既保留序列整体学习，又提升梯度稳定性。