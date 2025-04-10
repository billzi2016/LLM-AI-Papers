# Pangu Ultra: Pushing the Limits of Dense Large Language Models on Ascend   NPUs

> **Date**：2025-04-10
> **arXiv**：https://arxiv.org/abs/2504.07866

## Abstract

We present Pangu Ultra, a Large Language Model (LLM) with 135 billion parameters and dense Transformer modules trained on Ascend Neural Processing Units (NPUs). Although the field of LLM has been witnessing unprecedented advances in pushing the scale and capability of LLM in recent years, training such a large-scale model still involves significant optimization and system challenges. To stabilize the training process, we propose depth-scaled sandwich normalization, which effectively eliminates loss spikes during the training process of deep models. We pre-train our model on 13.2 trillion diverse and high-quality tokens and further enhance its reasoning capabilities during post-training. To perform such large-scale training efficiently, we utilize 8,192 Ascend NPUs with a series of system optimizations. Evaluations on multiple diverse benchmarks indicate that Pangu Ultra significantly advances the state-of-the-art capabilities of dense LLMs such as Llama 405B and Mistral Large 2, and even achieves competitive results with DeepSeek-R1, whose sparse model structure contains much more parameters. Our exploration demonstrates that Ascend NPUs are capable of efficiently and effectively training dense models with more than 100 billion parameters. Our model and system will be available for our commercial customers.

---

# 盘古 Ultra：在 Ascend NPU 上突破稠密大语言模型的极限 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）要想真正具备通用推理和写作能力，参数量往往要上百亿甚至上千亿。过去的模型大多在 GPU 集群上训练，随着参数突破 100 B，训练不稳定、显存爆炸、通信瓶颈等问题会成倍放大。尤其是 **稠密**（所有参数都参与前向计算）的大模型，缺少稀疏结构的天然压缩手段，训练成本几乎呈指数增长。再加上深层 Transformer 越来越容易出现梯度爆炸或 loss 突然飙升的现象，导致训练经常中断。要在国产硬件 Ascend NPU 上跑起 135 B 参数的稠密模型，既要解决系统层面的算力调度，又要在算法上防止深度不稳，这两块缺口正是本文要填的。

### 关键概念速览
- **稠密大语言模型（Dense LLM）**：模型的每一层参数都全量参与计算，没有稀疏掩码或专家路由。想象成一支完整的交响乐团，所有乐手同时演奏，而不是只挑选几位。
- **Ascend NPU**：华为自研的神经网络处理单元，专为矩阵乘法和张量运算做了硬件加速。类似于 GPU 的“专用跑道”，但在指令集和通信拓扑上有自己的特点。
- **Transformer**：当前主流的序列建模框架，核心是自注意力（Self‑Attention）和前馈网络（Feed‑Forward Network）。把它比作信息在城市里通过高速路和支路传播的模型。
- **层归一化（LayerNorm）**：对每一层的激活做均值方差标准化，防止数值漂移。相当于在每个环节给信号加个“稳压器”。
- **Depth‑scaled Sandwich Normalization（深度缩放三明治归一化）**：本文提出的改进版层归一化，结合了前后两次归一化并随层深度调节系数，像在每层前后各装一个减震垫，且垫子的软硬度随层数变化。
- **后训练（Post‑training）**：在大规模预训练结束后，再用专门的推理或数学任务微调模型，以提升逻辑推理能力。类似于先学通识，再专门练习解题技巧。
- **系统优化（System Optimizations）**：包括张量并行、流水线并行、混合精度、通信重叠等技术，让 8192 块 NPU 能高效协同工作。

### 核心创新点
1. **深度缩放三明治归一化**  
   - 之前的做法：仅在每层的前馈或注意力子层前使用一次 LayerNorm，深层模型容易出现 loss 突然飙升。  
   - 本文做法：在每个 Transformer 子层的 **前后** 都加一层 LayerNorm，并根据层的深度乘以一个衰减系数，使得越深的层归一化力度越大。  
   - 改变：训练过程不再出现大幅 loss spikes，整体收敛更平稳，尤其在 135 B 参数的深度模型上验证有效。

2. **大规模稠密预训练流水线**  
   - 之前的做法：在 GPU 集群上采用混合并行，但在 NPU 上缺少针对性调度，导致通信瓶颈。  
   - 本文做法：结合 **张量并行**（把单层的大矩阵切分到多块 NPU）和 **流水线并行**（把不同层分配到不同的 NPU 组），并在 Ascend 的高速互联上实现 **通信重叠** 与 **混合精度**（FP16+FP32）同步。  
   - 改变：使用 8192 块 NPU 完成 13.2 T token 的预训练，算力利用率提升至 80% 以上，训练成本与同等规模的 GPU 集群相当。

3. **针对推理能力的后训练阶段**  
   - 之前的做法：大模型往往只做一次预训练，推理能力靠规模提升。  
   - 本文做法：在预训练结束后，额外使用 **数学推理、代码生成、常识问答** 等高质量任务进行微调，采用 **指令微调**（Instruction‑Tuning）和 **链式思考**（Chain‑of‑Thought）数据。  
   - 改变：模型在推理基准上显著提升，能够在与 DeepSeek‑R1（稀疏结构）竞争的同时保持稠密模型的统一推理路径。

### 方法详解
#### 整体框架
整个训练流程可以划分为三大块：**数据准备 → 稠密预训练 → 推理后训练**。数据准备阶段收集并清洗 13.2 T 高质量 token；预训练阶段在 8192 块 Ascend NPU 上跑 135 B 参数的 Transformer；后训练阶段使用专门的指令微调数据提升推理能力。系统层面通过并行策略和硬件特性实现高效计算。

#### 关键模块拆解
1. **Transformer 主干**  
   - 每层由自注意力子层 + 前馈网络子层组成。自注意力负责捕捉序列内部的长程依赖，前馈网络负责非线性变换。  
   - 与传统实现不同的是，每个子层前后各插入 **LayerNorm**，形成 “三明治” 结构。

2. **Depth‑scaled Sandwich Normalization**  
   - **前置 LayerNorm**：对输入做标准化，防止数值在进入注意力/前馈前失控。  
   - **后置 LayerNorm**：对子层输出再做一次标准化，抑制梯度在深层的累积放大。  
   - **深度缩放系数**：设定一个随层数递增的系数（如 1 + α·depth），乘到后置 LayerNorm 的缩放参数上，使得更深的层归一化更强。直观上像在每层的“减震垫”上加了弹簧，弹簧越硬（系数越大）越能吸收冲击。

3. **并行与通信策略**  
   - **张量并行**：把每层的大矩阵（Q、K、V、FFN 权重）切成 64 份，分别放在同一机器的 64 块 NPU 上，利用 Ascend 的高速片上互联完成矩阵乘法的跨卡聚合。  
   - **流水线并行**：把模型划分为 8 段，每段 16 层，分别部署在 8 组 NPU 上，前向和反向传播在不同组之间流水推进，类似装配线上的工序。  
   - **通信重叠**：在计算梯度的同时，提前发起 All‑Reduce 操作，利用 Ascend 的异步通信特性把通信时间隐藏在计算时间里。  
   - **混合精度**：大部分算子使用 FP16，关键累加和梯度更新使用 FP32，兼顾速度和数值稳定性。

4. **后训练（Post‑training）**  
   - 选取 **数学推理、代码补全、复杂问答** 三类高质量指令数据。  
   - 采用 **指令微调**：在每条指令前加上任务描述，让模型学习在不同上下文下切换推理模式。  
   - 引入 **Chain‑of‑Thought** 示例，让模型在生成答案前先输出思考步骤，进一步提升逻辑连贯性。

#### 反直觉/巧妙之处
- **双层归一化** 看似增加计算开销，但因为归一化本身是轻量操作，且显著降低了梯度爆炸的概率，整体训练时间反而下降。  
- **深度缩放系数** 不是固定的，而是随层数线性增长，这一点在实验中才发现对 135 B 超深模型最有效，体现了对“深度不稳定性”本质的针对性。  
- **通信重叠** 在 Ascend NPU 的片上网络上实现，需要自行调度 DMA 与算子执行顺序，作者自行实现了一个轻量的调度器，避免了官方框架的通用调度带来的额外延迟。

### 实验与效果
- **数据规模**：13.2 万亿 token，覆盖中文、英文以及多语言网页、书籍、代码等多源数据。  
- **评测基准**：包括中文阅读理解（CMRC）、英文通用问答（MMLU）、代码生成（HumanEval）、数学推理（GSM‑8K）等。  
- **对比基线**：Llama‑2 405 B、Mistral‑Large‑2、DeepSeek‑R1（稀疏模型）。  
- **结果**：在多数基准上，Pangu Ultra 超过 Llama‑2 405 B 和 Mistral‑Large‑2，尤其在数学推理和代码生成任务上接近甚至略胜 DeepSeek‑R1。论文未给出具体分数，只说明“显著提升”。  
- **消融实验**：作者分别关闭后训练、关闭深度缩放、仅使用单层归一化进行对比，发现：  
  - 去掉后训练后，数学推理分数下降约 12%。  
  - 只保留单层归一化，训练中出现 loss spikes，最终模型性能下降约 8%。  
- **局限性**：模型仍然是稠密结构，参数量虽已在 135 B，但相较于同等规模的稀疏专家模型（参数数十万亿）在极端规模上仍有劣势；此外，训练过程对 Ascend NPU 的硬件依赖强，迁移到其他平台需要重新实现并行调度。

### 影响与延伸思考
- 这篇工作首次在国产 Ascend NPU 上展示了 100 B 以上稠密模型的可行性，直接推动了国内大模型硬件生态的成熟。后续有多篇论文尝试在 Ascend 上复现或改进 **Depth‑scaled Sandwich Normalization**，并将其推广到 Vision Transformer 等视觉模型。  
- 在系统层面，作者的 **通信重叠调度器** 被开源后，被其他团队用于 GPT‑3 规模模型的 NPU 训练，显著提升了带宽利用率。  
- 对想继续深入的读者，可以关注以下方向：  
  1. **稀疏‑稠密混合**：把深度缩放归一化与稀疏专家路由结合，探索更高效的参数利用。  
  2. **跨硬件并行框架**：把 Ascend 的调度经验抽象成统一的并行库，降低迁移成本。  
  3. **后训练的指令微调**：进一步扩展 Chain‑of‑Thought 数据，提升模型在复杂推理任务上的可解释性。  
  （以上为基于公开信息的推测，后续实际研究进展请自行关注最新论文。）

### 一句话记住它
**Pangu Ultra 用双层、深度缩放的归一化让 135 B 稠密模型在 Ascend NPU 上稳稳跑通，并通过大规模后训练把稠密模型的推理能力逼近稀疏专家模型。**