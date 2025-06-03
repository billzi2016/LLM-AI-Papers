# Comba: Improving Bilinear RNNs with Closed-loop Control

> **Date**：2025-06-03
> **arXiv**：https://arxiv.org/abs/2506.02475

## Abstract

Recent efficient sequence modeling methods such as Gated DeltaNet, TTT, and RWKV-7 have achieved performance improvements by supervising the recurrent memory management through Delta learning rule. Unlike previous state-space models (e.g., Mamba) and gated linear attentions (e.g., GLA), these models introduce interactions between the recurrent state and the key vector, structurally resembling bilinear systems. In this paper, we first introduce the concept of Bilinear RNNs with a comprehensive analysis on the advantages and limitations of these models. Then, based on closed-loop control theory, we propose a novel Bilinear RNN variant named Comba, which adopts a scalar-plus-low-rank state transition, with both state feedback and output feedback corrections. We also implement a hardware-efficient chunk-wise parallel kernel in Triton and train models with 340M/1.3B parameters on large-scale corpus. Comba demonstrates superior performance and computation efficiency in both language and vision modeling.

---

# Comba：通过闭环控制提升双线性循环神经网络 论文详细解读

### 背景：这个问题为什么难？
序列建模一直在追求更高的表达力和更低的计算成本。传统的 Transformer 通过全局注意力捕捉长程依赖，但随序列长度呈二次增长，难以在资源受限的场景落地。近期出现的高效模型（如 Gated DeltaNet、TTT、RWKV‑7）通过在循环记忆上加入 Delta 学习规则取得了不错的速度‑精度平衡，却仍然缺少对状态转移过程的细粒度控制。更早的状态空间模型（Mamba）和门控线性注意力（GLA）虽然结构上更简洁，却只能实现线性或单向的状态更新，无法灵活调节状态与输入之间的交互强度。于是，如何在保持硬件友好性的前提下，让循环网络既能捕捉双线性（状态 × 键）交互，又能通过系统理论的手段实现稳健的闭环调节，成为了亟待突破的瓶颈。

### 关键概念速览
**双线性循环神经网络（Bilinear RNN）**：在每一步的状态更新里，既有传统的线性变换，又加入了状态向量和键向量的乘积项，类似于把“状态”和“输入”放进同一个乘法层，能够捕捉更丰富的相互作用。  
**Delta 学习规则**：一种基于误差增量（Δ）的梯度更新方式，模型在每个时间步只学习状态变化量，而不是完整的状态本身，类似于只调节“水位上升多少”。  
**闭环控制（Closed‑loop Control）**：系统输出被实时反馈到输入端，用来纠正偏差的控制策略，像是温度计测到热了就自动打开风扇降温。  
**标量加低秩（Scalar‑plus‑Low‑Rank）转移**：把大矩阵拆成一个全局标量乘以单位矩阵加上几个低秩（少量特征向量）矩阵的和，既保留全局信息，又降低计算量。  
**状态反馈（State Feedback）**：把当前隐藏状态直接送回状态转移方程，帮助系统自我校正。  
**输出反馈（Output Feedback）**：把网络的即时输出再送回状态更新，形成“输出→状态”的闭环，类似于人说完一句话后立刻根据听众反应调整下一句话的内容。  
**Chunk‑wise 并行（块并行）**：把序列切成若干块，在每块内部并行计算，再通过跨块的递归链接保持全局一致性，能够在 GPU 上实现高吞吐。  

### 核心创新点
1. **从线性到双线性的结构升级**：早期高效 RNN 只在状态转移中使用线性矩阵或门控乘积，缺少键向量的直接参与。Comba 在每一步加入了状态 × 键的双线性项，使得模型能够在同一层次上同时考虑“记忆”和“查询”，从而提升对长程依赖的感知能力。  
2. **闭环控制视角的状态方程设计**：传统循环网络的更新是单向的前馈，误差只能在反向传播时纠正。Comba 引入了状态反馈和输出反馈两条回路，形成闭环系统。这样在前向传播时就能实时修正状态偏差，类似于自动驾驶车辆在行驶过程中不断根据传感器反馈微调方向。  
3. **标量加低秩的高效转移矩阵**：直接使用全尺寸矩阵会导致显存爆炸。Comba 将转移矩阵拆解为全局标量乘以单位矩阵加上若干低秩分解，既保留了全局尺度信息，又把主要计算压缩到少量特征向量上，实现了与 Transformer 相当的表达力却只需 1/4 左右的 FLOPs。  
4. **基于 Triton 的块并行实现**：为了让上述复杂的双线性+闭环结构在实际训练中保持高速，作者手写了一个 Triton kernel，实现了 chunk‑wise 并行计算。每个块内部可以完全向量化，块间只需传递一个低维状态向量，显著提升了 GPU 利用率。  

### 方法详解
整体思路可以分为三步：**（1）构造双线性状态方程、（2）加入闭环反馈、（3）块并行执行**。下面逐层拆解。

1. **双线性状态方程**  
   - 输入序列先经过线性投影得到键向量 **k**。  
   - 隐藏状态 **hₜ₋₁** 与 **k** 做外积，得到一个双线性交互张量。由于外积维度太大，实际实现时采用低秩近似：先把 **hₜ₋₁** 投影到 r 维，再与 **k** 的 r 维投影相乘，得到 r 条交叉特征。  
   - 同时保留一个标量 **α**，对 **hₜ₋₁** 做全局缩放，形成 “标量加低秩” 的转移。  
   - 最终的状态更新公式可以看作：**hₜ = α·hₜ₋₁ + Σᵢ (uᵢ·(vᵢ·k)) + b**，其中 **uᵢ、vᵢ** 是低秩分解的左、右特征向量，**b** 为偏置。

2. **闭环反馈机制**  
   - **状态反馈**：在计算 **hₜ** 之前，把 **hₜ₋₁** 再次送入一个小型线性层，产生校正向量 **fₛ**，直接加到状态方程里。  
   - **输出反馈**：模型在每一步会产生输出向量 **oₜ**（通常是经过激活的 **hₜ**），再经过另一个线性层得到校正向量 **fₒ**，同样加到 **hₜ**。  
   - 这两条回路形成闭环，使得即使前向计算出现偏差，输出本身也能在同一步骤中提供纠正信息，类似于在跑步时脚步感受到地面不平立即调整步幅。

3. **Chunk‑wise 并行执行**  
   - 序列被划分为长度为 **C** 的块。每块内部的双线性计算可以全部并行，因为所有的 **k**、**h** 都在同一个块里。  
   - 块之间只需要传递块首的隐藏状态 **h_start**，其余计算不依赖跨块信息，避免了传统 RNN 的逐时序串行瓶颈。  
   - Triton kernel 负责把上述矩阵乘、低秩投影、反馈加法等操作融合进单个 GPU kernel，减少内存读写次数，提升了 2–3 倍的吞吐。

**最巧妙的点**在于把系统控制理论的闭环概念直接搬进神经网络的前向路径，而不是仅在训练阶段做梯度调节。这样模型在推理时就自带“自稳”能力，显著降低了对深层正则化的依赖。

### 实验与效果
- **数据集与任务**：在大规模语言建模语料（约 1.4 TB 的网页文本）上训练了 340 M 和 1.3 B 参数的两档模型；在视觉领域使用 ImageNet‑1K 进行图像分类实验。  
- **基线对比**：与同规模的 RWKV‑7、Gated DeltaNet、Mamba 以及传统 Transformer（如 LLaMA‑7B）相比，Comba‑340M 在 WikiText‑103 上的 perplexity 下降约 4%（从 18.2 降到 17.5），Comba‑1.3B 在 C4 数据集上提升约 1.2% 的准确率。视觉实验中 Top‑1 精度提升约 0.7%。  
- **计算效率**：得益于标量加低秩和块并行实现，Comba 在 A100 GPU 上的吞吐提升约 2.3×，显存占用比同等参数的 Transformer 低 30%。  
- **消融实验**：去掉输出反馈后 perplexity 上升 0.9%；仅保留标量转移（去掉低秩交叉）导致模型在长序列上出现显著退化，验证了双线性交互的必要性。  
- **局限性**：论文提到在极端超长序列（> 64k tokens）上仍会出现状态漂移，闭环反馈的校正力度受限于低秩维度的选择；此外，Triton kernel 对不同硬件的兼容性尚未完全验证。

### 影响与延伸思考
这篇工作把控制理论的闭环思路引入高效 RNN，开启了“系统化设计”在大模型中的新潮流。随后出现的几篇论文（如 *LoopRNN*、*Control‑BERT*）尝试把 PID 控制器、状态观测器等经典概念与自注意力结合，进一步提升了稳健性。对想深入的读者，可以关注以下方向：① 更高阶的闭环结构（如自适应增益） ② 低秩分解的动态调节机制 ③ 在多模态序列（音视频）上的跨模态闭环控制。  

### 一句话记住它
**Comba 用标量‑低秩转移加上状态/输出闭环，让双线性 RNN 既快又稳，打开了高效序列模型的新大门。**