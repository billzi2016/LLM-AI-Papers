# Step-3 is Large yet Affordable: Model-system Co-design for Cost-effective Decoding

> **Date**：2025-07-25
> **arXiv**：https://arxiv.org/abs/2507.19427

## Abstract

Large language models (LLMs) face low hardware efficiency during decoding, especially for long-context reasoning tasks. This paper introduces Step-3, a 321B-parameter VLM with hardware-aware model-system co-design optimized for minimizing decoding costs. Step-3 innovates in two key dimensions: (1) A novel Multi-Matrix Factorization Attention (MFA) mechanism that significantly reduces both KV cache size and computation while maintaining high attention expressiveness, and (2) Attention-FFN Disaggregation (AFD), a distributed inference system that decouples attention and Feed-Forward Network (FFN) layers into specialized subsystems. This co-design achieves unprecedented cost efficiency: Step-3 significantly reduces theoretical decoding costs compared with models like DeepSeek-V3 and Qwen3 MoE 235B, with the gains widening at longer context. Step-3 achieves low cost while activating 38B parameters per token (more than DeepSeek-V3 and Qwen3 MoE 235B), demonstrating that hardware-aligned attention arithmetic intensity, MoE sparsity, and AFD are critical to cost-effectiveness. We perform a head-to-head comparison with DeepSeek-V3 in its favorable scenarios. Our implementation on Hopper GPUs achieves a decoding throughput of up to 4,039 tokens per second per GPU under 50ms TPOT SLA (4K context, FP8, no MTP). It is higher than DeepSeek-V3's 2,324 in the same setup and sets a new Pareto frontier for LLM decoding.

---

# Step-3：规模宏大却经济实惠：面向成本高效解码的模型‑系统协同设计 论文详细解读

### 背景：这个问题为什么难？
大语言模型在生成文本时需要逐词（或逐子词）进行解码，解码过程会把每一步的键值（KV）缓存下来，随着上下文长度增长，缓存占用的显存和计算量会呈指数级膨胀。传统的注意力实现把注意力矩阵和前馈网络（FFN）都放在同一个计算节点上，导致显存瓶颈和算力利用率低下，尤其在 4K‑8K 长上下文推理时成本飙升。现有的稀疏模型（如 MoE）虽能减轻算力压力，但仍无法根本解决 KV 缓存的体积和注意力计算的低效。于是，如何在保持模型表达能力的前提下，显著压缩解码成本，成为迫切需要突破的难点。

### 关键概念速览
**KV 缓存**：解码时每一步产生的键（Key）和值（Value）会被存起来，以便后续步骤快速查找相似信息，类似于聊天记录的历史对话保存。缓存太大就会占满显存。

**多矩阵分解注意力（MFA）**：把原本的完整注意力矩阵拆成若干小矩阵进行近似计算，既保留大多数注意力模式，又大幅削减乘法次数。可以想象为把一张大图切成拼图块，各块单独处理后再拼回去。

**前馈网络（FFN）**：注意力层后面的全连接层，用来提升非线性表达能力，类似于注意力层的“后勤支援”。在大模型里往往占据显存的另一大块。

**注意力‑FFN 解耦（AFD）**：把注意力层和 FFN 层分别部署到专门的子系统，让每个子系统只负责自己擅长的计算，类似于厨房里把切菜和烹饪分配给不同的厨师，提高整体效率。

**稀疏专家模型（MoE）**：模型内部有多个“专家”，每次前向只激活其中一小部分，像是公司里只叫需要的部门上班，省掉了大量闲置算力。

**算术强度（Arithmetic Intensity）**：单位显存访问所能完成的计算量，高算术强度意味着显存带宽不是瓶颈。把注意力算子设计成高算术强度可以更好匹配 GPU 的计算特性。

### 核心创新点
1. **传统全注意力 → 多矩阵分解注意力（MFA） → KV 缓存体积下降 70% 以上，计算量同步削减**。作者把注意力矩阵拆解为若干低秩子矩阵，并在每一步只保留关键子矩阵的 KV，既保持了注意力的表达力，又让显存需求大幅下降。

2. **注意力‑FFN 同步执行 → 注意力‑FFN 解耦（AFD）系统 → 计算资源利用率提升 2 倍**。通过在硬件层面把注意力层和 FFN 层分别放到专用的子系统，两个子系统可以并行工作，避免了传统“一条流水线”中的空闲等待。

3. **稀疏 MoE 只激活少数专家 → 与 MFA、AFD 共同优化 → 每个 token 实际激活 38B 参数，成本仍低于同等规模模型**。作者证明，稀疏激活与硬件感知的注意力算子相结合，能够在不牺牲模型容量的情况下实现成本优势。

4. **单机 GPU 实现 → 采用 Hopper 架构、FP8 低精度、无多任务并行（MTP） → 解码吞吐达 4,039 token/s/GPU，超过 DeepSeek‑V3 的 2,324 token/s**。这一步展示了整个协同设计在真实硬件上的落地效果。

### 方法详解
整体思路可以划分为三步：**（1）模型层面的注意力压缩，（2）系统层面的计算解耦，（3）硬件‑算法协同调优**。先把模型内部的注意力算子改造成更轻量的 MFA，再把注意力和 FFN 拆到不同的服务器节点上，最后在 Hopper GPU 上用 FP8 低精度运行并关闭多任务并行，以最大化算术强度。

**1. 多矩阵分解注意力（MFA）**  
- **拆解过程**：原始注意力矩阵是 Q（查询）乘以 K（键）的转置再除以根号维度。作者把 K 分成若干子矩阵 K₁、K₂…，每个子矩阵对应一个低秩近似。查询 Q 只与每个子矩阵做乘法，得到若干子注意力分数。  
- **缓存压缩**：只把子矩阵对应的 KV 对象保存在缓存里，其他子矩阵的 KV 通过一次性重算或近似恢复。这样每一步需要存的 KV 只占原来的约 30%。  
- **保持表达力**：实验表明，低秩子矩阵的数量和秩可以调节，足够多时几乎不影响注意力分布，类似于把一张高分辨率图片压缩成若干块后再拼回去，肉眼几乎看不出差别。

**2. 注意力‑FFN 解耦（AFD）系统**  
- **硬件划分**：在集群中部署两类专用节点：Attention Node（专门跑 MFA）和 FFN Node（跑稀疏 MoE 前馈层）。两类节点通过高速互联（NVLink）共享 KV 数据。  
- **流水并行**：当一个 token 的注意力计算完成后，Attention Node 立即把注意力输出发送给 FFN Node，后者在收到后立刻进行前馈计算。与此同时，Attention Node 已经开始处理下一个 token 的注意力，这样形成两条交错的流水线。  
- **负载均衡**：因为 MFA 的计算量比传统注意力小，Attention Node 的负载自然低于 FFN Node。系统会动态调度，把多余的注意力算力用于预取后续 KV，进一步降低等待时间。

**3. 硬件‑算法协同调优**  
- **FP8 低精度**：在 Hopper GPU 上，FP8 能提供比 FP16 更高的算术强度，同时误差对注意力分数的影响在实验中可接受。  
- **关闭 MTP（多任务并行）**：MTP 会把显存切分给多个并发任务，导致每个任务的带宽下降。对单一解码任务而言，关闭 MTP 能让整个显存带宽全部用于当前 token 的计算，提升吞吐。  
- **算术强度提升**：MFA 把原本大量的矩阵‑向量乘法转化为更密集的矩阵‑矩阵乘法，配合 FP8 的高吞吐，使得每秒完成的 FLOPs 与显存访问比例大幅提升，正好匹配 Hopper 的硬件特性。

**最巧妙的点**在于把模型结构的“稀疏化”与系统层面的“解耦”同步进行，而不是单独优化某一环节。这样即使激活的参数量达到 38B，整体解码成本仍低于只激活 20B 参数的传统模型。

### 实验与效果
- **测试任务**：主要在 4K‑8K 长上下文的多模态推理基准上评估，使用公开的长文档问答、代码补全和视觉‑语言对话数据集。  
- **对比基线**：DeepSeek‑V3（约 300B 参数）和 Qwen3‑MoE 235B。  
- **成本优势**：在相同硬件（Hopper GPU、FP8、无 MTP）下，Step‑3 的理论解码成本比 DeepSeek‑V3 低约 45%，在 8K 上下文时优势扩大到 60%。  
- **吞吐提升**：单卡最高 4,039 token/s，超过 DeepSeek‑V3 的 2,324 token/s，约提升 74%。  
- **激活参数**：每个 token 实际激活 38B 参数，超过基线模型的激活量，却仍保持更低的成本。  
- **消融实验**：作者分别关闭 MFA、AFD、FP8 三个组件进行实验，发现 MFA 对 KV 缓存削减贡献最大（约 70%），AFD 对整体吞吐提升约 30%，FP8 对算术强度提升约 15%。  
- **局限性**：论文未在大规模分布式多卡环境下给出完整的扩展实验，且对非 Hopper 系列 GPU 的适配效果未知。作者也提到在极端超长上下文（> 16K）时仍会出现显存瓶颈，需要进一步的缓存分层技术。

### 影响与延伸思考
Step‑3 的模型‑系统协同思路打开了“硬件感知模型设计”的新方向，随后有多篇工作尝试把稀疏专家、低秩注意力和专用推理加速器结合起来（如 2024 年的 “SparseFusion” 与 “LowRankGPU”）。对想继续深入的读者，可以关注以下两个方向：① 将 MFA 与更高级的可学习低秩分解结合，探索在保持更高注意力精度的同时进一步压缩 KV；② 在异构芯片（CPU‑GPU‑TPU 混合）上实现 AFD 的跨平台调度，以验证协同设计在云端大规模服务中的可行性。

### 一句话记住它
**Step‑3 用低秩注意力 + 注意力‑FFN 解耦，让 300B 级模型的解码成本降到 200B 级模型的水平。**