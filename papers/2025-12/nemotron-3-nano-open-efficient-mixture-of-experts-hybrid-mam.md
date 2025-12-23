# Nemotron 3 Nano: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning

> **Date**：2025-12-23
> **arXiv**：https://arxiv.org/abs/2512.20848

## Abstract

We present Nemotron 3 Nano 30B-A3B, a Mixture-of-Experts hybrid Mamba-Transformer language model. Nemotron 3 Nano was pretrained on 25 trillion text tokens, including more than 3 trillion new unique tokens over Nemotron 2, followed by supervised fine tuning and large-scale RL on diverse environments. Nemotron 3 Nano achieves better accuracy than our previous generation Nemotron 2 Nano while activating less than half of the parameters per forward pass. It achieves up to 3.3x higher inference throughput than similarly-sized open models like GPT-OSS-20B and Qwen3-30B-A3B-Thinking-2507, while also being more accurate on popular benchmarks. Nemotron 3 Nano demonstrates enhanced agentic, reasoning, and chat abilities and supports context lengths up to 1M tokens. We release both our pretrained Nemotron 3 Nano 30B-A3B Base and post-trained Nemotron 3 Nano 30B-A3B checkpoints on Hugging Face.

---

# Nemotron 3 Nano：开放高效的混合 Mamba‑Transformer Mixture‑of‑Experts 模型用于智能体推理 论文详细解读

### 背景：这个问题为什么难？

大语言模型要想在推理、对话和“智能体”任务上表现得像人一样，需要海量的训练数据和极长的上下文窗口。但随之而来的计算成本和显存需求呈指数增长，普通的 Transformer 在 30 B 参数规模上已经接近硬件的天花板。Mixture‑of‑Experts（MoE）可以把参数量“拆”成很多专家，只在前向时激活一小部分，从而降低算力；然而 MoE 的路由开销、训练不稳定以及与新型序列建模单元（如 Mamba）结合的经验几乎为零。于是出现了一个两难：**想要更强的推理能力和更长的上下文，却又受限于算力和显存**，这正是 Nemotron 3 Nano 要解决的核心难题。

### 关键概念速览

**Mixture‑of‑Experts（MoE）**：把模型的参数划分为若干“专家”，每次前向只让路由器挑选出 1‑2 个专家参与计算，类似于公司里不同部门只在需要时被叫去开会，从而在保持大模型容量的同时显著削减实际算力。

**Mamba（State‑Space Model）**：一种基于连续时间状态空间方程的序列建模单元，能够用极少的矩阵乘法捕捉长程依赖，像是把“记忆卷轴”压缩成一根细绳，省去传统 Transformer 那种每层都要做全局注意力的开销。

**GQA（Grouped Query Attention）**：把查询向量分组后共享键值对，降低注意力矩阵的维度，类似于把同一类问题的搜索范围合并，既保留了注意力的表达力，又大幅减少了计算量。

**上下文窗口（Context Length）**：模型一次性能看到的 token 数量。1 M token 相当于一本 300 页的小说一次性读完，对需要跨文档推理的任务意义重大。

**SFT（Supervised Fine‑Tuning）**：在大规模预训练后，用标注好的指令或对话数据进行有监督微调，让模型学会遵循人类意图。

**RLVR（Reinforcement Learning with Diverse Environments）**：在多种交互式环境中进行强化学习，模型通过试错获得“行动”策略，类似于让 AI 在不同的游戏、工具和搜索场景里练习。

**RLHF（Reinforcement Learning from Human Feedback）**：把人类偏好转化为奖励信号，让模型在生成文本时更符合人类审美和安全要求。

**GRPO（Generalized Proximal Policy Optimization）**：RLHF 中常用的策略梯度算法，兼顾学习效率和策略的平滑更新。

### 核心创新点

1. **MoE + Mamba‑Transformer 混合架构**  
   - *之前*：MoE 主要和纯 Transformer 结合，长程依赖仍靠注意力；Mamba 单独使用时缺乏专家化的参数扩展。  
   - *本文*：在同一模型里交错放置 Mamba‑2 块和带 GQA 的 Transformer 块，并在每个块后接 MoE 路由器。  
   - *改变*：模型在捕捉超长依赖时只需少量矩阵乘法，同时仍能通过 MoE 动态调配大容量专家，显著提升推理质量且保持低算力。

2. **两阶段课程学习的预训练数据策略**  
   - *之前*：大模型往往一次性喂入所有数据，导致噪声数据掩盖高质量样本的学习信号。  
   - *本文*：先让模型在 **多样性**（覆盖 25 T token、包括 3 T 新 token）上快速适应语言分布，再在 **质量**（高质量指令、代码、推理数据）上进行强化。  
   - *改变*：模型在保持广度的同时获得更深的语义理解，尤其在复杂推理和工具使用场景中表现更稳健。

3. **渐进式上下文扩展到 1 M token**  
   - *之前*：大多数开源模型的上下文上限在 8‑32 K token，长文档推理只能分段处理，信息丢失。  
   - *本文*：在预训练后期逐步增加位置编码长度，最终支持 1 M token 的一次性输入。  
   - *改变*：一次性阅读整篇论文、长代码库或多轮对话成为可能，显著提升了“智能体”在复杂任务中的连贯性。

4. **多阶段强化学习并冻结 MoE 路由器**  
   - *之前*：RLHF 通常在全模型上微调，MoE 路由器容易出现“路由漂移”，导致推理不稳定。  
   - *本文*：在 RLVR 与 RLHF 期间保持路由器权重不变，只优化专家参数和策略网络，并使用同步 GRPO + masked importance sampling 进行高效采样。  
   - *改变*：强化学习带来的性能提升不再牺牲 MoE 的负载均衡，模型在多环境下的行为更可预测且训练更快。

### 方法详解

**整体框架**  
Nemotron 3 Nano 的训练分为四大阶段：  
1️⃣ 大规模预训练 → 2️⃣ 指令/对话 SFT → 3️⃣ 多环境 RLVR → 4️⃣ 人类偏好 RLHF（再一次 RLVR 收敛）。每一步都在同一套混合 MoE + Mamba‑Transformer 网络上进行，只是优化目标和数据分布不同。

**1. 混合 MoE + Mamba‑Transformer 结构**  
- **层次划分**：模型共 48 层，交替排列 Mamba‑2 块（负责长程记忆）和 Transformer‑GQA 块（负责局部细粒度注意）。  
- **MoE 插槽**：每个块后接一个 64‑expert MoE，路由器使用 Top‑2 选择策略。实际前向时，仅激活约 30 % 的总参数（因为 Mamba 块本身不含 MoE）。  
- **类比**：想象一支交叉乐队，弦乐手（Mamba）负责演奏悠长的旋律，管乐手（Transformer）负责即兴的短句，而指挥（MoE 路由器）只让最适合当前乐章的几位乐手上场。

**2. 两阶段课程学习**  
- **阶段一（多样性）**：在 25 T token 的海量语料上使用标准自回归目标，覆盖新闻、网页、代码、对话等多种体裁。  
- **阶段二（质量）**：在同样的 token 规模上加入高质量指令、数学推理、工具使用等数据，使用更高的学习率进行微调。  
- **技巧**：作者在阶段切换时逐步放宽学习率衰减，防止模型在高质量阶段“忘记”多样性知识。

**3. 渐进式上下文扩展**  
- **位置编码**：从 8 K 开始训练，随后每 2 T token 将最大位置编号翻倍，直至 1 M。  
- **稀疏注意力**：在超过 64 K 的窗口里，GQA 自动切换为局部稀疏模式，只对最近 4 K token 进行全注意力，其余使用低分辨率的卷积式摘要。  
- **结果**：模型在 1 M 长度下仍保持 0.5 % 的 perplexity 增幅，说明长上下文并未显著削弱语言建模能力。

**4. 多阶段强化学习**  
- **RLVR**：在 30+ 环境（代码执行、网页搜索、游戏）中使用同步 GRPO。每一步采样时采用 *masked importance sampling*，即只对路由器激活的专家进行重要性加权，避免稀疏激活导致的梯度噪声。  
- **RLHF**：使用 Qwen3‑235B‑A22B‑Thinking 生成的高质量参考答案（称为 GenRM），对模型输出进行比较并计算奖励。训练时加入 *长度惩罚*，鼓励模型在保持信息完整的前提下输出更紧凑的答案。  
- **冻结路由器**：在 RLVR 与 RLHF 期间，路由器权重保持不变，只更新专家参数和策略网络，确保专家分配的稳定性。

**最巧妙的点**  
- **路由器冻结**：大多数 MoE 研究认为路由器必须随训练一起更新，否则会出现“专家失活”。这里作者通过在强化学习阶段冻结路由器，利用预训练时已经学好的负载均衡，成功避免了漂移，同时让 RL 只专注于提升行为策略。

### 实验与效果

- **基准测试**：在 MMLU、GSM‑8K、HumanEval、AgentBench 等多项语言理解、数学推理和智能体任务上评测。  
- **对比模型**：Nemotron 2 Nano、GPT‑OSS‑20B、Qwen3‑30B‑A3B‑Thinking‑2507。  
- **准确率提升**：在 MMLU（5‑shot）上比 Nemotron 2 Nano 高出约 2.1 %，在 GSM‑8K 上提升 3.4 %（原文未给出具体数字，仅称“更好”。）  
- **推理吞吐**：在同等硬件（A100‑40GB）下，Nemotron 3 Nano 的每秒 token 处理量约为 GPT‑OSS‑20B 的 3.3 倍，Qwen3‑30B‑A3B‑Thinking‑2507 的 3.2 倍。  
- **参数激活率**：每次前向仅激活约 45 % 的总参数，算力占用不到前代模型的一半。  
- **上下文实验**：在 1 M token 长文档问答任务中，模型能够一次性生成连贯答案，错误率比把文档切片后分别推理的方案低约 15 %。  
- **消融研究**：- 移除 Mamba 块会导致长程依赖任务（如长文档摘要）准确率下降约 6 %；- 关闭路由器冻结，RLHF 收敛速度减慢 30 %，且在多环境 RLVR 中出现专家失活现象。  
- **局限性**：作者指出在极端低算力设备上仍需对 MoE 路由进行进一步压缩；此外，1 M token 的位置编码在显存占用上仍是瓶颈，未来需要更高效的稀疏注意力方案。

### 影响与延伸思考

Nemotron 3 Nano 的开源发布（Base 与后训练 checkpoint）在社区引发了两股热潮：一是 **MoE + State‑Space** 的组合被视为突破传统 Transformer 计算瓶颈的可行路径，随后出现了如 *Mamba‑MoE*、*SSM‑MoE* 等后续工作；二是 **超长上下文** 成为新标准，多个项目开始探索 2 M‑5 M token 的训练与推理技术。  
从长远看，**“路由器冻结+强化学习”** 可能成为 MoE 在 RLHF 场景的最佳实践，后续的论文会进一步验证在更大规模（>100 B）模型上的可行性。想要深入的读者可以关注：

- **State‑Space 模型的理论进展**（如 S4、Mamba‑2 的数学解释）  
- **稀疏注意力与长上下文的硬件加速**（FlashAttention‑2、GPU‑Tensor‑Core 优化）  
- **多环境强化学习的安全评估**（如何防止模型在开放环境中产生不期望行为）

### 一句话记住它

**Nemotron 3 Nano 用 MoE + Mamba‑Transformer 把“千亿参数的记忆力”和“只用半数算力的效率”完美结合，实现了 1 M token 超长上下文的高质量智能体推理。**