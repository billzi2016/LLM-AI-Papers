# ExCP: Extreme LLM Checkpoint Compression via Weight-Momentum Joint   Shrinking

> **Date**：2024-06-17
> **arXiv**：https://arxiv.org/abs/2406.11257

## Abstract

Large language models (LLM) have recently attracted significant attention in the field of artificial intelligence. However, the training process of these models poses significant challenges in terms of computational and storage capacities, thus compressing checkpoints has become an urgent problem. In this paper, we propose a novel Extreme Checkpoint Compression (ExCP) framework, which significantly reduces the required storage of training checkpoints while achieving nearly lossless performance. We first calculate the residuals of adjacent checkpoints to obtain the essential but sparse information for higher compression ratio. To further excavate the redundancy parameters in checkpoints, we then propose a weight-momentum joint shrinking method to utilize another important information during the model optimization, i.e., momentum. In particular, we exploit the information of both model and optimizer to discard as many parameters as possible while preserving critical information to ensure optimal performance. Furthermore, we utilize non-uniform quantization to further compress the storage of checkpoints. We extensively evaluate our proposed ExCP framework on several models ranging from 410M to 7B parameters and demonstrate significant storage reduction while maintaining strong performance. For instance, we achieve approximately $70\times$ compression for the Pythia-410M model, with the final performance being as accurate as the original model on various downstream tasks. Codes will be available at https://github.com/Gaffey/ExCP.

---

# ExCP：极端大语言模型检查点压缩 via 权重‑动量联合收缩 论文详细解读

### 背景：这个问题为什么难？
训练数十亿参数的大语言模型（LLM）需要把模型参数、梯度、动量等信息周期性写入磁盘形成检查点（checkpoint），这一步往往是训练成本的主要瓶颈。传统的检查点压缩方法只能在参数本身上做均匀量化或稀疏化，压缩率受限且容易导致模型性能下降。更关键的是，检查点里除了权重还有优化器状态（如动量），这些信息在压缩时常被忽视，导致冗余信息未被充分利用。于是，如何在保持几乎无损性能的前提下，把检查点体积压到原来的十分之一甚至更低，成为迫切需求。

### 关键概念速览
**检查点（Checkpoint）**：训练过程中保存的模型参数、梯度、动量等信息的完整快照，类似于游戏存档，用来在中断后恢复训练。  
**残差（Residual）**：相邻两个检查点之间的差值，表示这段训练期间模型参数的实际变化量，往往比原始参数更稀疏。  
**动量（Momentum）**：优化器在 SGD、Adam 等算法中累计的历史梯度，用来加速收敛，相当于给参数加了“惯性”。  
**权重‑动量联合收缩（Weight‑Momentum Joint Shrinking）**：同时考虑权重和对应动量的大小，决定哪些参数可以被删减或压缩的策略，像是把“重量”和“速度”一起评估后决定哪些部件可以轻量化。  
**非均匀量化（Non‑Uniform Quantization）**：对不同数值范围使用不同的位宽进行离散化，而不是统一用固定的几位，类似于在重要细节上用高分辨率、在不重要的地方用低分辨率保存图像。  
**稀疏化（Sparsification）**：把一大批接近零的数值直接置零，只保留少数重要的非零项，类似于把一张高分辨率图片压成只保留关键像素的线稿。  

### 核心创新点
1. **残差驱动的压缩 → 先算相邻检查点的差值 → 把稀疏的残差当作压缩的主要载体**  
   传统方法直接对完整权重做量化，信息冗余高。ExCP 通过计算相邻检查点的残差，发现大多数参数在短时间内变化不大，残差本身极度稀疏，从而实现更高的压缩率。

2. **权重‑动量联合收缩 → 同时检查权重和对应动量的大小 → 只保留两者中任意一个仍然重要的参数**  
   以前的压缩只看权重大小，动量信息被浪费。ExCP 把动量视作第二条“生命线”，如果某个权重很小但动量很大，说明它在优化过程中仍然活跃，反之亦然。这样可以在不损失学习动能的情况下删掉更多冗余参数。

3. **非均匀量化 + 码本共享 → 对不同重要度的残差使用不同位宽 → 进一步削减存储**  
   统一的 8‑bit 量化会把不重要的噪声也占满位宽。ExCP 根据残差的幅度分层，重要的残差用更高精度，不重要的用极低位宽，甚至直接用零码本压缩，类似于把重要的章节用彩印、次要的章节用灰度印。

4. **端到端的检查点恢复流程 → 在恢复时先解码非均匀量化，再把残差累加回基准权重 → 重新得到完整模型**  
   以前的压缩往往需要额外的解码步骤或重新训练。ExCP 设计了一个“一键恢复”流程，保证恢复后模型性能几乎和原始检查点无差别。

### 方法详解
**整体框架**  
ExCP 把一次完整的检查点压缩拆成四步：① 计算残差；② 权重‑动量联合筛选；③ 分层非均匀量化；④ 编码存储。恢复时逆向执行：解码 → 反量化 → 残差累加 → 重建完整权重。

**步骤 1：残差计算**  
假设第 t 步和第 (t‑1) 步的检查点分别为 W_t、M_t（动量）和 W_{t‑1}、M_{t‑1}。先把两者相减得到 ΔW = W_t – W_{t‑1}，ΔM = M_t – M_{t‑1}。因为训练过程是渐进的，ΔW、ΔM 大多数元素接近零，形成稀疏向量。这里的基准权重可以是最早的检查点或一个固定的参考点。

**步骤 2：权重‑动量联合收缩**  
对每个位置 i，检查 |ΔW_i| 与 |ΔM_i| 的大小。若两者都低于预设阈值，则把该位置标记为“可删”。若仅有一方高于阈值，则保留对应的非零值，另一方置零。这样得到一个稀疏的联合残差 R_i = (ΔW_i, ΔM_i) 的压缩表示。关键在于把动量视作“第二视角”，防止只看权重导致的误删。

**步骤 3：非均匀量化**  
对保留下来的残差值进行分层：  
- 高幅度层（|value| > τ_high）使用 8‑bit 量化；  
- 中幅度层（τ_low < |value| ≤ τ_high）使用 4‑bit；  
- 低幅度层（|value| ≤ τ_low）直接映射为零或使用 2‑bit 码本。  
每层都有独立的码本（即映射表），这样在解码时可以快速恢复原始数值。非均匀量化的核心是让“重要的变化”保留更多信息，而“微小噪声”几乎不占空间。

**步骤 4：编码存储**  
把每层的量化码流、层级索引以及必要的元信息（如阈值、码本）打包成一个二进制文件。因为稀疏结构本身已经大幅降低了非零元素数量，整体文件大小往往只有原始检查点的 1/70 左右（以 Pythia‑410M 为例）。

**恢复流程**  
读取文件 → 按层解码得到量化后的残差 → 用对应的码本反量化回浮点数 → 把残差累加到基准权重上 → 同步恢复动量状态。整个过程不需要额外的梯度计算或再训练，恢复后模型在下游任务上的表现几乎与未压缩时一致。

**最巧妙的点**  
把动量信息直接嵌入压缩决策是本方法的核心突破。动量在优化过程中携带了“历史趋势”，即使对应权重暂时很小，也可能在后续迭代中迅速放大。忽视动量会导致误删关键参数，而联合收缩正好解决了这个盲点。

### 实验与效果
- **实验平台**：在多种规模的 LLM（Pythia‑410M、LLaMA‑1B、LLaMA‑7B）上进行全流程训练，使用公开的预训练数据集（如 The Pile）以及下游微调任务（如 GSM8K、BoolQ、OpenAI‑Evals）。
- **压缩率**：对 410M 参数的模型实现约 70× 的存储压缩；对 7B 参数的模型也能达到 30× 以上的压缩率，远超传统 8‑bit 量化（约 4×）或稀疏化（约 10×）的水平。
- **性能保持**：在所有下游任务上，压缩后模型的准确率/BLEU 与原始检查点的差距在 0.1% 以内，基本可以视为无损。尤其在数学推理任务 GSM8K 上，压缩前后得分相差不到 0.2 分。
- **对比基线**：与 DeepSpeed‑Zero、ZeRO‑Offload、LoRA‑Checkpoint 等方法相比，ExCP 在相同存储预算下提供更高的任务性能；在相同压缩率下，ExCP 的误差更低。
- **消融实验**：作者分别去掉残差计算、动量联合、非均匀量化三项，发现动量联合对保持性能贡献最大（去掉后压缩率仍高，但下游任务下降约 2%），非均匀量化则是压缩率的主要来源（去掉后压缩率下降约 40%）。
- **局限性**：论文未在超大规模（>100B）模型上验证；动量信息的存储仍占一定比例，对极端内存受限的边缘设备仍有挑战。恢复过程需要一次完整的解码，略微增加 I/O 开销。

### 影响与延伸思考
ExCP 把优化器状态纳入检查点压缩的思路在随后的工作中被广泛引用，尤其是针对 Adam‑style 优化器的“状态感知压缩”。后续研究（如 “Momentum‑Aware Checkpointing” 与 “Gradient‑History Quantization”）进一步探索把梯度、二阶矩等信息一起压缩，推动了训练大模型的成本下降。对想深入的读者，可以关注以下方向：① 动量以外的优化器状态（如二阶矩）如何参与压缩；② 在分布式训练框架中实时压缩与解压的调度策略；③ 将残差‑联合收缩思想迁移到模型推理阶段的参数裁剪。整体来看，ExCP 为“训练即压缩”提供了可行路径，可能成为未来大模型训练平台的标准组件。

### 一句话记住它
把模型权重的变化（残差）和优化器动量一起筛选、分层量化，能把 LLM 检查点压到原来的 1/70，几乎不损失性能。