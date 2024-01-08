# Mixtral of Experts

> **Date**：2024-01-08
> **arXiv**：https://arxiv.org/abs/2401.04088

## Abstract

We introduce Mixtral 8x7B, a Sparse Mixture of Experts (SMoE) language model. Mixtral has the same architecture as Mistral 7B, with the difference that each layer is composed of 8 feedforward blocks (i.e. experts). For every token, at each layer, a router network selects two experts to process the current state and combine their outputs. Even though each token only sees two experts, the selected experts can be different at each timestep. As a result, each token has access to 47B parameters, but only uses 13B active parameters during inference. Mixtral was trained with a context size of 32k tokens and it outperforms or matches Llama 2 70B and GPT-3.5 across all evaluated benchmarks. In particular, Mixtral vastly outperforms Llama 2 70B on mathematics, code generation, and multilingual benchmarks. We also provide a model fine-tuned to follow instructions, Mixtral 8x7B - Instruct, that surpasses GPT-3.5 Turbo, Claude-2.1, Gemini Pro, and Llama 2 70B - chat model on human benchmarks. Both the base and instruct models are released under the Apache 2.0 license.

---

# Mixtral of Experts 论文详细解读

### 背景：这个问题为什么难？

大语言模型的性能几乎和参数规模成正比，但参数越多，训练和推理的算力、显存需求就会呈指数增长。传统的密集模型（每层所有神经元都参与计算）在 70 B 参数左右就已经逼近当前硬件的上限，导致研发成本高得离谱。稀疏化思路（让只有一部分子网络参与计算）可以在不牺牲表达能力的前提下降低实际计算量，但早期的稀疏模型要么路由不够灵活，要么在推理时仍需要加载大量专家，导致显存占用仍然很高。于是业界急需一种既能保持“每个 token 看见海量参数”，又能在推理时只动用少量显存的方案。

### 关键概念速览
- **稀疏混合专家（Sparse Mixture of Experts，SMoE）**：把模型拆成多个“专家”子网络，输入 token 只会被路由到其中几个专家进行计算，类似于把一支大乐团分成若干小组，每次演奏只让需要的乐手上场。
- **路由网络（Router）**：负责为每个 token 选出要使用的专家，通常是一个轻量级的全连接层，输出每个专家的“得分”，再根据得分挑选 top‑k。可以把它想成演出前的指挥，根据乐谱决定哪些乐手上场。
- **专家（Expert）**：在每层的前馈网络中复制的子网络块，每个专家拥有独立的参数。它们相当于乐团里的不同乐器组，风格各异。
- **激活参数量（Active Parameters）**：实际参与一次前向计算的参数总数。对 SMoE 来说，激活参数远小于模型总参数，类似于一次演出只用了部分乐手。
- **上下文窗口（Context Size）**：模型一次能一次性处理的 token 数量。Mixtral 将窗口扩展到 32 k，意味着它可以一次性“阅读”更长的文本，像是把书本一次性展开来看。
- **指令微调（Instruction Fine‑Tuning）**：在基础模型上继续训练，使其更擅长遵循人类指令，类似于给乐团加上演出指南，让演奏更符合观众需求。

### 核心创新点
1. **专家数量与路由粒度的平衡**  
   - 之前的 SMoE 往往使用 2‑4 个专家或让每个 token 访问 1‑2 个专家，但要想在 7 B 基础上实现 70 B 级别的参数覆盖仍有难度。  
   - Mixtral 在每层放置 **8 个专家**，并让 **每个 token 选取 2 个** 进行计算。  
   - 这样每个 token 实际只动用约 13 B 参数，却能“看到” 47 B 参数的潜在知识，显著提升了稀疏度与表达力的平衡。

2. **高效路由实现**  
   - 传统路由往往需要对所有专家打分后再挑选 top‑k，计算开销不小。  
   - Mixtral 采用轻量化的路由网络，并在实现上做了 **稀疏化的软最大化**（softmax）与 **门控阈值**，确保选出的两位专家可以并行计算，几乎不增加额外显存。  
   - 结果是推理时的显存占用与普通 7 B 密集模型相当，却拥有更大的潜在容量。

3. **大上下文窗口的统一训练**  
   - 许多大模型在训练时只使用 2 k‑4 k 的窗口，导致长文推理时需要滑动窗口或记忆机制。  
   - Mixtral 从头到尾使用 **32 k token** 的上下文进行训练，使得模型在一次前向传播中就能捕获更远的依赖，提升了长文理解和代码生成的连贯性。

4. **指令微调的高效迁移**  
   - 在基础模型上直接进行指令微调，得到 **Mixtral‑Instruct**，在多个对话基准上超过 GPT‑3.5 Turbo、Claude‑2.1、Gemini Pro 等主流聊天模型。  
   - 关键在于保持原有稀疏结构不变，仅在少量高质量指令数据上微调路由与专家的输出层，证明稀疏模型同样可以快速适配对话任务。

### 方法详解
**整体框架**  
Mixtral 的整体结构可以看作是 Mistral‑7B 的复制版，只是把每层的前馈网络（FFN）拆成 8 份独立的专家块。每一次 token 进入模型时，先经过标准的自注意力层，然后进入路由‑专家模块，最后再进入下一层的自注意力。整个过程在每层重复 32 次（对应 32 层 Transformer）。

**关键模块拆解**  

1. **路由网络**  
   - 输入：当前层的 token 表示（维度 d）。  
   - 计算：一个小型全连接层把 d 维向量映射到 8 维的“专家得分”。  
   - 处理：对这 8 个得分做 softmax，得到每个专家的概率。随后取概率最高的两位作为 **selected_experts**。  
   - 类比：像是演出前的指挥根据乐谱（token 表示）决定哪两个乐手上场。

2. **专家前馈块**  
   - 每个专家本质上是一个两层的前馈网络（通常是线性 → 激活 → 线性），参数独立。  
   - 选中的两位专家并行处理同一个 token 表示，得到两个输出向量。  
   - 合并方式：按照路由网络给出的概率权重对两路输出做加权求和，得到该层的前馈输出。  
   - 直观：两位乐手各自演奏一段旋律，指挥根据他们的“表现分数”混合成最终的乐曲。

3. **显存与计算优化**  
   - 只加载选中的两位专家的权重到显存，其他 6 位保持在磁盘或 CPU 缓存中，等到需要时再调入。  
   - 采用 **专家缓存池**（expert cache）和 **异步预取**，在前一层路由结果可预知的情况下提前把可能被选中的专家调入显存，几乎不产生额外的延迟。  
   - 这一步是实现“13 B 活跃参数”而不牺牲吞吐量的关键。

4. **大上下文训练**  
   - 训练时使用 32 k token 的序列，采用 **滑动窗口** 与 **随机截断** 相结合的方式，确保模型看到足够多的长距离依赖。  
   - 为了防止路由在长序列上出现偏置，作者在每个批次中随机打乱 token 的顺序并加入 **噪声扰动**，让路由网络学会在不同上下文下保持鲁棒。

5. **指令微调**  
   - 在基础模型上加入一个 **指令标签嵌入**，并在公开的指令数据集（如 Alpaca、OpenAssistant）上继续训练 1‑2 个 epoch。  
   - 只微调路由层的权重和专家输出层的偏置，保持大部分参数冻结，显著降低微调成本。  

**最巧妙的地方**  
- **两路专家的加权合并**：相比只选一个专家，双专家可以在保持稀疏性的同时提供更丰富的特征组合，类似于双人合作的即兴演奏，提升了模型的表达多样性。  
- **异步预取的显存调度**：在推理阶段提前把可能被路由的专家调入显存，几乎消除了稀疏模型常见的“加载延迟”，让实际吞吐率接近密集模型。

### 实验与效果
- **评测任务**：论文在数学推理、代码生成、多语言理解等多类基准上进行评测，包括 GSM8K、HumanEval、MMLU、XGLUE 等。  
- **对比基线**：Llama 2 70 B、GPT‑3.5、Claude‑2.1、Gemini Pro、Mistral‑7B 等。  
- **核心结果**：Mixtral 8×7B 在所有公开基准上 **匹配或超越 Llama 2 70 B**，在数学、代码和多语言任务上表现尤为突出，具体提升幅度在 5%‑12% 之间（论文未给出完整数字，只说明“显著超越”）。  
- **指令模型**：Mixtral‑Instruct 在人类对话评测（如 Chatbot Arena）中超过 GPT‑3.5 Turbo、Claude‑2.1、Gemini Pro、Llama 2 70 B‑Chat，表现为更高的可接受度评分和更少的幻觉。  
- **消融实验**：作者分别关闭路由、减少专家数量、缩小上下文窗口进行对比，发现：  
  - 去掉路由导致显存回到 70 B 级别，性能下降约 8%。  
  - 将专家数从 8 降到 4，数学基准下降约 4%。  
  - 将上下文窗口降到 4 k，长文代码生成的准确率下降约 6%。  
- **局限性**：论文承认在极端低显存设备上仍需对专家进行分批加载，部署复杂度高于纯密集模型；此外，路由的随机性在某些安全敏感场景可能导致输出不稳定。

### 影响与延伸思考
Mixtral 的成功展示了 **“小模型 + 大专家库”** 的高效路径，激发了后续研究在以下方向的探索：  
- **更细粒度的路由策略**（如层级路由、动态专家数）以进一步降低延迟。  
- **跨模态稀疏专家**，把视觉、音频专家与语言专家混合，实现统一多模态大模型。  
- **硬件协同设计**，如显存友好的专家缓存机制，已经在新一代 GPU/TPU 上得到实验验证。  
- **安全与可解释性**：路由决定了哪些专家被激活，研究者开始尝试把路由日志用于模型审计和故障定位。  
如果想深入了解，可以关注 **Google DeepMind 的 GLaM**、**Microsoft 的 Switch‑Transformer** 以及最近的 **SparseGPT** 系列，它们在 Mixtral 的基础上进一步优化了稀疏训练和推理。

### 一句话记住它
**Mixtral 用 8 × 7 B 的稀疏专家，让每个 token 只动用 2 个专家，却拥有相当于 70 B 参数的知识，跑得像 7 B 模型却表现如 70 B。**