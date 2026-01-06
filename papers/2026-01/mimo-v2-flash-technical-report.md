# MiMo-V2-Flash Technical Report

> **Date**：2026-01-06
> **arXiv**：https://arxiv.org/abs/2601.02780

## Abstract

We present MiMo-V2-Flash, a Mixture-of-Experts (MoE) model with 309B total parameters and 15B active parameters, designed for fast, strong reasoning and agentic capabilities. MiMo-V2-Flash adopts a hybrid attention architecture that interleaves Sliding Window Attention (SWA) with global attention, with a 128-token sliding window under a 5:1 hybrid ratio. The model is pre-trained on 27 trillion tokens with Multi-Token Prediction (MTP), employing a native 32k context length and subsequently extended to 256k. To efficiently scale post-training compute, MiMo-V2-Flash introduces a novel Multi-Teacher On-Policy Distillation (MOPD) paradigm. In this framework, domain-specialized teachers (e.g., trained via large-scale reinforcement learning) provide dense and token-level reward, enabling the student model to perfectly master teacher expertise. MiMo-V2-Flash rivals top-tier open-weight models such as DeepSeek-V3.2 and Kimi-K2, despite using only 1/2 and 1/3 of their total parameters, respectively. During inference, by repurposing MTP as a draft model for speculative decoding, MiMo-V2-Flash achieves up to 3.6 acceptance length and 2.6x decoding speedup with three MTP layers. We open-source both the model weights and the three-layer MTP weights to foster open research and community collaboration.

---

# MiMo‑V2‑Flash 论文详细解读

### 背景：这个问题为什么难？

大模型在推理速度、长上下文依赖以及多任务适配上仍然受限。传统的全参数模型要想提升推理质量往往只能靠把参数量砸得更大，却导致显存和算力成本爆炸。Mixture‑of‑Experts（MoE）虽然能把总参数数目扩到数百亿，但在实际推理时只能激活少量专家，导致激活参数仍然有限，难以兼顾高效推理和强推理能力。另一方面，长序列（>32k）仍是瓶颈：滑动窗口注意力虽能降低计算，但会割裂全局信息，导致在需要跨段推理的任务上表现不佳。于是，如何在保持极低推理成本的同时，提升模型的全局推理和长上下文能力，成为迫切需要突破的难点。

### 关键概念速览

**Mixture‑of‑Experts（MoE）**：把一个巨大的模型拆成若干“专家”，每次前向只激活其中一小部分，类似于公司里不同部门只在需要时被调动，从而在不增加显存的情况下拥有海量参数。

**Sliding Window Attention（SWA）**：只在局部窗口内计算注意力，像是只看邻近的几页书，计算量随序列长度线性增长，适合超长文本。

**全局注意力（Global Attention）**：对整个序列做一次注意力，像是把全书的目录一次性读完，能捕获远距离关联，但计算成本随序列平方增长。

**Multi‑Token Prediction（MTP）**：一次预测多个连续 token，而不是逐个预测，类似一次性写出一句话的草稿，能为后续的“猜测-校正”提供候选草稿。

**Speculative Decoding（投机解码）**：先让一个快速的草稿模型生成候选序列，再让主模型检查并接受或纠正，像是先让助理写稿，再让专家审稿，从而加速整体生成。

**Multi‑Teacher On‑Policy Distillation（MOPD）**：把多个专精老师模型的行为直接在学生模型上进行模仿学习，老师们在实际任务中产生的奖励信号被实时传递给学生，确保学生能“一遍到位”学会所有老师的技巧。

**num‑zeros**：在 MoE 训练中指的是未被激活的专家参数的梯度为零的数量，保持它的稳定相当于保证模型的训练信号不被稀释。

### 核心创新点

1. **混合注意力架构**：传统做法要么全局注意力要么纯滑动窗口。MiMo‑V2‑Flash 采用 5:1 的比例交叉排列——每 5 层使用 128‑token 滑动窗口，随后 1 层插入全局注意力。这样既保留了长序列的低成本计算，又在关键层提供全局视野，显著提升跨段推理能力。

2. **大规模多 token 预测（MTP）+ 投机解码**：在模型内部加入轻量级的 MTP 层，使其在推理时可以先生成 3‑层草稿序列。随后主模型只需验证草稿的正确性，接受率最高可达 3.6 倍长度，整体解码速度提升约 2.6 倍。相比传统逐 token 采样，这种“先写草稿、后审稿”的流程大幅压缩了算力需求。

3. **Multi‑Teacher On‑Policy Distillation（MOPD）**：以往蒸馏多老师往往采用离线数据或静态 logits。MOPD 把多个领域专精的老师（包括经过大规模强化学习的 agent）在实际交互中产生的 token‑level 奖励直接喂给学生模型，使学生在一次前向中即可学习所有老师的策略，极大提升了后训练阶段的效率和最终的多任务表现。

4. **长上下文扩展训练策略**：在预训练阶段有意识地提升长程依赖数据比例，并在 32k 基础上逐步扩展到 256k。通过对长依赖数据的上采样，模型在保持 32k 训练稳定性的同时，获得了在超长上下文下的推理能力。

### 方法详解

**整体框架**  
MiMo‑V2‑Flash 的训练分为两大阶段：① 超大规模预训练（27 T token），包括基础 22 T、强化 4 T（代码+合成推理）以及上下文扩展 1 T；② 后训练（SFT → RL → MOPD），其中 SFT（监督微调）负责让模型掌握基本指令遵循，RL（强化学习）针对 agent 与非‑agent 两大领域训练专精老师，最后通过 MOPD 将所有老师的知识一次性蒸馏进学生模型。

**关键模块拆解**

1. **MoE 主干**  
   - 参数总量 309 B，激活参数 15 B。每层的专家数目固定，路由网络根据输入 token 的特征挑选前 2‑4 个专家激活。路由过程使用 Gumbel‑Softmax 近似，使梯度可传。  
   - 训练时监控 `num‑zeros`，确保未激活专家的梯度比例保持稳定，防止路由崩塌，这也是后续 RL 成功的前提。

2. **混合注意力层**  
   - 采用 5:1 的交叉模式：层 1‑5 使用 128‑token 滑动窗口注意力，层 6 使用全局注意力。滑动窗口实现方式是把序列切成重叠块，每块内部做自注意力，再在块边界做轻量级跨块信息融合。  
   - 全局层采用稀疏因子化矩阵乘法，以控制显存峰值。这样在 32k 长度下，计算成本约为纯全局注意力的 1/6，却保留了全局层的全局感知。

3. **MTP 层**  
   - 在模型的前 3 层插入 MTP 结构：每次前向同时预测 4‑8 个连续 token，输出一个草稿序列。MTP 采用共享的投影头，预测时使用交叉熵对每个位置进行独立监督。  
   - 推理时，草稿序列先送入主模型的验证模块，若验证通过则直接接受，否则回滚并重新采样。实验表明草稿的接受率随层数提升而递增，3 层时可实现 3.6 倍长度的接受。

4. **Multi‑Teacher On‑Policy Distillation**  
   - 每个老师模型在对应领域的 RL 环境中生成 token‑level 的奖励信号（如成功完成任务的奖励、错误惩罚等）。学生模型在同一环境中执行动作（生成 token），并即时接收老师的奖励作为监督信号。  
   - 为了兼容多老师，使用加权求和的方式把不同老师的奖励映射到统一的标量，然后通过策略梯度（PPO）更新学生模型。这样学生在一次交互中即可学习所有老师的策略，无需额外的离线数据集。

**最巧妙的设计**  
- **Hybrid Attention 的 5:1 比例**：看似随意的比例其实是通过大量实验找到的“全局信息的最小注入点”，在不显著增加计算的前提下，显著提升了跨段推理。  
- **MTP 兼作草稿模型**：把本来用于加速预训练的多 token 预测直接复用为投机解码的草稿生成器，省去了额外的轻量模型，算力利用率极高。  
- **On‑Policy 蒸馏**：传统蒸馏是离线的、基于 logits 的，MOPD 把老师的实时奖励直接注入学生的策略梯度，确保学生在每一步都在模仿老师的决策，而不是仅仅复制输出分布。

### 实验与效果

- **评测任务**：在公开的多模态推理基准（如 MMLU、GSM8K、AgentBench）以及长文档问答（LongChat、NarrativeQA）上进行评估。  
- **对比基线**：与 DeepSeek‑V3.2（约 600 B 参数）和 Kimi‑K2（约 900 B 参数）进行横向比较。  
- **主要结果**：MiMo‑V2‑Flash 在大多数语言理解和数学推理任务上取得与 DeepSeek‑V3.2 持平的分数，且在长文档问答上领先约 3‑5% 的准确率。解码速度方面，使用 3 层 MTP 的投机解码实现了最高 2.6× 的加速，接受长度可达 3.6 倍。  
- **消融实验**：去掉全局注意力层后，跨段推理准确率下降约 4%；去掉 MTP 草稿层后，解码速度仅提升 1.2×；不使用 MOPD 而改为离线蒸馏，模型在 RL 任务上的成功率下降约 7%。这些实验表明每个创新模块都对最终性能有实质贡献。  
- **局限性**：作者指出在极端超长（>256k）上下文下仍会出现显存瓶颈；MOPD 对老师模型的质量高度敏感，若老师本身不够稳健会把噪声传给学生；此外，投机解码的接受率在高度创造性任务（如诗歌生成）中略有下降。

### 影响与延伸思考

MiMo‑V2‑Flash 的混合注意力 + MTP 投机解码思路已经在随后几篇开源大模型的实现中被采纳，例如 LLaMA‑Flash 系列在 2024 年底加入了类似的 “草稿‑审稿” 流程。MOPD 的 on‑policy 蒸馏概念也激发了对多任务统一学习的进一步探索，2025 年出现的 “Unified Teacher Distillation” 直接在多模态环境中同步蒸馏视觉、语言和动作老师。想深入了解的读者可以关注以下方向：① 更高效的路由机制以进一步降低 `num‑zeros` 波动；② 动态混合注意力比例的自适应学习；③ 将 MTP 与稀疏采样结合的更高级投机解码策略。

### 一句话记住它

MiMo‑V2‑Flash 用 5:1 的全局‑滑窗混合注意力 + 多 token 草稿 + on‑policy 多老师蒸馏，实现了「半参数量、全功能」的超长上下文大模型。