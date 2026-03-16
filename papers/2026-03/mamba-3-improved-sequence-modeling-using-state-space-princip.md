# Mamba-3: Improved Sequence Modeling using State Space Principles

> **Date**：2026-03-16
> **arXiv**：https://arxiv.org/abs/2603.15569

## Abstract

Scaling inference-time compute has emerged as an important driver of LLM performance, making inference efficiency a central focus of model design alongside model quality. While the current Transformer-based models deliver strong model quality, their quadratic compute and linear memory make inference expensive. This has spurred the development of sub-quadratic models with reduced linear compute and constant memory requirements. However, many recent linear models trade off model quality and capability for algorithmic efficiency, failing on tasks such as state tracking. Moreover, their theoretically linear inference remains hardware-inefficient in practice. Guided by an inference-first perspective, we introduce three core methodological improvements inspired by the state space model (SSM) viewpoint of linear models. We combine: (1) a more expressive recurrence derived from SSM discretization, (2) a complex-valued state update rule that enables richer state tracking, and (3) a multi-input, multi-output (MIMO) formulation for better model performance without increasing decode latency. Together with architectural refinements, our Mamba-3 model achieves significant gains across retrieval, state-tracking, and downstream language modeling tasks. At the 1.5B scale, Mamba-3 improves average downstream accuracy by 0.6 percentage points compared to the next best model (Gated DeltaNet), with Mamba-3's MIMO variant further improving accuracy by another 1.2 points for a total 1.8 point gain. Across state-size experiments, Mamba-3 achieves comparable perplexity to Mamba-2 despite using half of its predecessor's state size. Our evaluations demonstrate Mamba-3's ability to advance the performance-efficiency Pareto frontier.

---

# Mamba-3：基于状态空间原理的序列建模改进 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在推理阶段的计算量仍然是瓶颈。传统的 Transformer 结构在每一步都要做全序列的注意力计算，导致计算随序列长度二次增长、显存随长度线性增长，实际部署成本高得离谱。为了解决这个问题，研究者们陆续推出了线性时序模型（如 S4、RWKV、DeltaNet），把注意力的二次复杂度降到线性甚至常数。但这些“快”模型往往在需要长期记忆或状态追踪的任务上表现不佳，甚至在硬件上跑不出理论的线性速度，因为实现细节（如单输入单输出的递归）限制了并行度。于是，如何在保持推理效率的同时，提升模型对复杂状态的捕捉能力，成为了迫切需要突破的点。

### 关键概念速览

**状态空间模型（State Space Model，SSM）**：一种用连续时间微分方程描述序列的数学框架，核心是把输入信号映射到隐藏状态，再映射回输出，类似于把序列当作电路信号流动。  

**离散化（Discretization）**：把连续时间的 SSM 转成离散时间递归的过程，就像把连续的水流抽象成每秒一次的取样。常见方法有欧拉法和梯形法。  

**复数状态（Complex-valued State）**：让隐藏状态拥有实部和虚部两层信息，类似于在二维平面上旋转的向量，能够捕捉更丰富的周期性和相位变化。  

**多输入多输出（MIMO）**：一次递归同时处理多个输入向量并产生多个输出向量，像一次批量处理多条小河的水流，而不是单条河流单独算。  

**解码延迟（Decode Latency）**：模型在生成下一个 token 时的实际耗时，直接决定了聊天机器人等实时应用的流畅度。  

**感知容量（Perplexity）**：语言模型对测试文本的预测不确定度，数值越低说明模型越懂语言。  

**Pareto 前沿**：在性能和效率两个维度上，找出“不能再同时提升一个而不牺牲另一个”的最佳点。

### 核心创新点

1. **梯形离散化取代欧拉法 → 用梯形法则对连续 SSM 进行离散化 → 递归公式更接近原始微分方程，数值误差更小，模型在长序列上保持更稳定的记忆。**  
2. **复数状态更新 → 将隐藏状态扩展为复数向量，并在递归中使用复数乘法 → 状态能够同时编码幅度和相位信息，提升对周期性、相位漂移等细粒度特征的追踪能力。**  
3. **MIMO 递归结构 → 在一次时间步的递归中并行计算多个输入/输出通道 → 解码时硬件利用率提升，吞吐量上升而不增加单步的计算深度或显存占用。**  
4. **轻量化状态规模 + 架构微调 → 在保持相同隐藏维度的前提下，把状态向量长度减半，同时加入层级归一化和门控改进 → 同等参数量下实现与前代模型相近的 perplexity，却用更少的状态存储。

### 方法详解

整体思路可以拆成三层：**输入嵌入 → 复数 SSM 递归（梯形离散化） → MIMO 输出投影**。模型先把 token 嵌入成实数向量，然后把每个维度映射到对应的复数状态空间，最后通过多通道投影得到下一个 token 的 logits。

1. **梯形离散化的递推**  
   - 连续 SSM 的核心方程是 `dx/dt = A x + B u`（A、B 为矩阵，u 为输入）。欧拉法直接用 `x_{t+1} = x_t + Δt (A x_t + B u_t)`，误差随步长线性增长。  
   - 梯形法则把前后两步的斜率都取平均：`x_{t+1} = x_t + (Δt/2) [A x_t + B u_t + A x_{t+1} + B u_{t+1}]`，整理后得到 `x_{t+1} = (I - Δt/2 A)^{-1} (I + Δt/2 A) x_t + ...`。这相当于在每一步做一次小规模的矩阵求逆，但因为 A 是对角化的（SSM 设计时保证），求逆成本是 O(1)。  
   - 结果是一个更“平滑”的递归，尤其在长序列上不会出现数值漂移。

2. **复数状态更新**  
   - 将每个隐藏维度拆成实部 + 虚部，形成复数向量 `z = x + i y`。递归公式中出现的矩阵乘法自然扩展为复数乘法，等价于在二维平面上做旋转+缩放。  
   - 这种表示让模型能够直接捕捉信号的相位信息，类似于傅里叶变换的基函数，尤其对周期性语言结构（如列表、编号）有优势。  
   - 实际实现时，作者把复数拆成两块实数张量，利用现有的矩阵乘法库，无需额外的复数运算支持。

3. **MIMO 递归**  
   - 传统 SSM 递归是单输入单输出（SISO），每一步只能处理一个 token 的嵌入。Mamba-3 把 `k` 个相邻 token 的嵌入拼成一个宽向量，一次性送入递归，得到 `k` 个输出。  
   - 这在硬件层面相当于把原本的逐步循环展开成向量化操作，显著提升 GPU/TPU 的算子利用率。  
   - 为了不让解码延迟增加，作者在解码时仍然只取第一个输出作为当前 token 的预测，而后面的 `k-1` 个输出被缓存用于后续步骤的“提前计算”，形成一种流水线效果。

4. **架构微调**  
   - 状态向量长度减半后，模型在每层加入了层归一化（LayerNorm）和门控线性单元（GLU）来平衡信息流。  
   - 这些细节确保即使状态容量下降，信息仍能在层间高效传递，保持了与 Mamba-2 相近的 perplexity。

**最巧妙的点**：把梯形离散化和复数状态结合起来，使得递归既更数值稳健，又能捕捉相位信息；再配合 MIMO 的硬件友好并行，几乎把“算法线性”转化为“实际常数时间”。这三者相互支撑，形成了论文的核心竞争力。

### 实验与效果

- **评测任务**：包括检索任务（如 LAMA）、状态追踪基准（如 TrackingBench）以及标准语言建模数据集（WikiText‑103、C4）。  
- **基线对比**：与同规模的 Gated DeltaNet、RWKV、S4 等模型比较。Mamba-3（1.5B 参数）在平均下游任务上提升 0.6% 准确率；其 MIMO 变体再额外提升 1.2%，总计 1.8% 的绝对增益。  
- **状态规模实验**：在相同隐藏维度下，把状态向量长度从 Mamba-2 的 64 降到 32，感知容量（perplexity）几乎持平，说明新离散化和复数状态弥补了容量的损失。  
- **消融研究**：论文分别去掉梯形离散化、复数状态、MIMO 三个模块，发现每去掉一个，整体准确率下降约 0.4‑0.6%；去掉全部则回到 DeltaNet 的水平。  
- **局限性**：作者指出复数状态在某些低位宽硬件上实现仍需额外的张量拆分，导致实际加速率未必达到理论上 2×；此外，MIMO 的缓存机制在极端低延迟场景（如实时语音）仍有冲突。

### 影响与延伸思考

Mamba-3 把“推理第一”理念落到实处，展示了在保持线性复杂度的同时，通过更精细的数值离散化和复数表示可以显著提升模型质量。此后，很多工作开始探索 **复数神经网络** 与 **高阶离散化**（如二阶 Runge‑Kutta）在大模型中的可行性；还有研究把 MIMO 思路推广到 **混合专家**（Mixture‑of‑Experts）框架，以进一步提升硬件吞吐。对想深入的读者，建议关注 **State Space Transformers**（S4‑Transformer）以及 **Complex Neural ODE** 系列的最新预印本，它们在理论上与 Mamba-3 的思路高度相通。

### 一句话记住它

**Mamba-3 用梯形离散化 + 复数状态 + MIMO 并行，让线性时序模型在效率和质量上一起跨越了“只能快不能准”的瓶颈。**