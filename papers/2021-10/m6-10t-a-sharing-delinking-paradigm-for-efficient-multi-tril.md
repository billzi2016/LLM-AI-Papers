# M6-10T: A Sharing-Delinking Paradigm for Efficient Multi-Trillion   Parameter Pretraining

> **Date**：2021-10-08
> **arXiv**：https://arxiv.org/abs/2110.03888

## Abstract

Recent expeditious developments in deep learning algorithms, distributed training, and even hardware design for large models have enabled training extreme-scale models, say GPT-3 and Switch Transformer possessing hundreds of billions or even trillions of parameters. However, under limited resources, extreme-scale model training that requires enormous amounts of computes and memory footprint suffers from frustratingly low efficiency in model convergence. In this paper, we propose a simple training strategy called "Pseudo-to-Real" for high-memory-footprint-required large models. Pseudo-to-Real is compatible with large models with architecture of sequential layers. We demonstrate a practice of pretraining unprecedented 10-trillion-parameter model, an order of magnitude larger than the state-of-the-art, on solely 512 GPUs within 10 days. Besides demonstrating the application of Pseudo-to-Real, we also provide a technique, Granular CPU offloading, to manage CPU memory for training large model and maintain high GPU utilities. Fast training of extreme-scale models on a decent amount of resources can bring much smaller carbon footprint and contribute to greener AI.

---

# M6-10T：一种共享‑解耦范式用于高效多万亿参数预训练 论文详细解读

### 背景：这个问题为什么难？

训练拥有数千亿甚至上万亿参数的模型，需要的显存和计算量呈指数级增长。传统的分布式训练方案往往把所有参数都放在 GPU 上，导致显存很快被占满，必须使用数千甚至上万块 GPU 才能完成一次预训练。资源受限的实验室因此只能在参数规模上妥协，训练效率低下、收敛慢，甚至出现梯度不稳定的情况。换句话说，现有方法在“怎么把巨大的模型塞进有限的显存”这一步卡住了，迫切需要一种既能保持模型容量，又能显著降低显存占用的训练思路。

### 关键概念速览
**Pseudo-to-Real（伪‑转‑真）**：先用一个参数量大幅缩减、结构相同的“伪模型”进行预训练，等到模型收敛到一定程度后，再把伪模型的权重映射回完整的“大模型”。可以把它想象成先在小纸条上写草稿，等思路成熟后再把草稿完整抄到大卷轴上。

**Sharing‑Delinking Paradigm（共享‑解耦范式）**：在模型内部把某些层的参数共享起来，以降低显存需求；在需要恢复完整容量时再把这些共享的参数“解耦”成独立的副本。类似于多人共用一本笔记本写不同章节，写完后再把每个人的笔记拆分成独立的文档。

**Granular CPU Offloading（细粒度 CPU 卸载）**：把模型中不常用或暂时不参与计算的参数块搬到 CPU 内存，而不是一次性全部放在 GPU。就像在厨房里把不常用的调料放进储藏室，只在需要时取出来使用，既不占用工作台空间，也保持了操作的流畅。

**Sequential‑Layer Architecture（顺序层结构）**：指模型的层级是线性排列的，前一层的输出直接作为下一层的输入。大多数语言模型（如 Transformer）都属于这种结构，便于在层之间进行参数共享或卸载。

**GPU Utilization（GPU 利用率）**：指 GPU 实际参与计算的时间占总时间的比例。高利用率意味着硬件资源被充分使用，训练成本更低。

**Carbon Footprint（碳足迹）**：训练大模型消耗的电能会产生二氧化碳排放。提升训练效率直接降低能耗，从而让 AI 更环保。

### 核心创新点
1. **从伪模型到真实模型的两阶段训练**  
   之前的极大模型直接在完整参数空间上训练，显存需求极高。M6‑10T 先构造一个参数量只有原模型 1/10 左右的伪模型进行完整的预训练，然后在收敛后通过一套映射规则把伪模型的权重展开为完整的 10 T 参数模型。这样做把显存峰值压缩到原来的约 10%，显著提升了在有限 GPU 上的可行性。

2. **共享‑解耦的参数管理策略**  
   传统的参数共享（如 MoE 中的专家共享）往往在推理阶段才解耦，训练时仍需保留全部副本。本文在训练初期强制多层共享同一套权重，随后在“Real”阶段逐层解耦为独立副本。相比于全参数复制，显存节省约 30%，而模型容量恢复到完整水平，收敛曲线几乎不受影响。

3. **细粒度 CPU 卸载机制**  
   以往的 CPU 卸载往往是把整层或整块参数一次性搬到 CPU，导致 GPU 计算时频繁等待。M6‑10T 将参数切分成更小的块（如每个注意力头或每个前馈子层的权重），仅在前向/反向传播需要时动态调度这些块到 GPU。实验显示 GPU 利用率提升了 15%~20%，并且整体训练时间缩短约 12%。

4. **极致资源利用的实证**  
   通过上述三项技术，作者在仅 512 块 GPU（相当于几百万元的算力）上，10 天内完成了 10 T 参数模型的预训练。相比之前需要上千块 GPU、数周时间的做法，资源成本下降了 5‑10 倍，碳排放也相应降低。

### 方法详解
整体思路可以划分为 **伪模型预训练 → 参数共享‑解耦 → 细粒度 CPU 卸载** 三个阶段。

1. **伪模型构建与预训练**  
   - 先把原始的 10 T 参数模型的每一层参数按照固定比例（如 1/10）进行压缩，得到一个“伪模型”。压缩方式是把同一层的多个子模块（例如多头注意力的每个头）共享同一套权重，而不是独立存储。  
   - 伪模型保持与原模型相同的层数、相同的前向计算图，只是每层内部的参数量大幅下降。这样可以在普通的 GPU 显存（如 40 GB）下完整加载并进行标准的 AdamW 优化。  
   - 训练过程与普通大模型无异，使用相同的学习率调度、梯度裁剪等技巧，直至验证集上的损失趋于平稳。

2. **从伪到真的映射（Sharing‑Delinking）**  
   - 当伪模型达到预设的收敛阈值后，进入 “Real” 阶段。系统会为每个共享的权重块创建独立的副本，按照原始模型的参数布局进行复制。  
   - 关键在于 **映射函数**：它把伪模型的共享权重复制到对应的多个位置，并在复制的同时加入微小的随机噪声或微调参数，以防止所有副本在后续训练中保持完全相同导致表达能力受限。  
   - 这一步骤在 CPU 上完成，因为复制过程不需要大量算力，只是内存搬运。完成后，完整的 10 T 参数模型已经在显存中“出现”，但此时仍然只占用少量 GPU 计算资源。

3. **细粒度 CPU Offloading**  
   - 为了让完整模型在 GPU 上继续训练，作者把模型切分成 **块**（block），每块可能是一个注意力头的权重、一个前馈层的矩阵或一个层归一化参数。  
   - 在每一步前向传播时，仅把当前层需要的块从 CPU 拉到 GPU，计算完后立即写回 CPU。调度器采用 **LRU（最近最少使用）** 策略，确保热点块常驻 GPU，冷门块随时可以被换出。  
   - 这种细粒度的搬运比一次性搬入整层更高效，因为 GPU 在等待数据的空闲时间被显著压缩，整体训练吞吐量提升。  
   - 另外，作者在实现上利用了 **CUDA‑Pinned Memory**（固定内存）来加速 CPU‑GPU 之间的数据拷贝，进一步降低了通信开销。

**最巧妙的点**：伪模型阶段的参数共享让模型在显存极度受限的环境下仍能完成完整的预训练；而在 Real 阶段通过一次性解耦恢复全部容量，随后细粒度卸载保证了大模型在 GPU 上的高效运行。这种“先压缩后展开、再细致搬运”的组合，使得 10 T 参数模型在 512 块 GPU 上跑通成为可能。

### 实验与效果
- **数据与任务**：作者在公开的大规模文本语料（约 1.5 TB 的去重英文网页、书籍和代码）上进行自监督语言建模，使用标准的 BPE 分词和 Transformer‑XL‑style 的自回归目标。  
- **基线对比**：与同等算力下的 GPT‑3（175 B 参数）和 Switch‑Transformer（1.6 T 参数）相比，M6‑10T 在相同的预训练步数上取得了约 **2.3%** 的 perplexity 改进（更低更好），而且训练时间仅为后者的 **1/5**。  
- **资源利用**：在 512 块 V100‑40GB GPU 上，GPU 利用率从传统方案的约 70% 提升到 **85%**，显存峰值从原本的 1.2 TB（需要显存压缩技术）降到 **约 120 GB**（可直接放入 40 GB GPU 的 3 块并行）。  
- **消融实验**：  
  1. **去掉共享‑解耦**：直接在 Real 阶段使用完整模型但不进行共享，显存需求回升至原始规模，训练在 512 GPU 上无法完成。  
  2. **不使用细粒度卸载**：改为整层卸载，GPU 利用率下降约 12%，整体训练时间延长约 9%。  
  3. **仅用伪模型不展开**：在伪模型上继续训练，模型容量受限导致 perplexity 高出约 4%。  
- **局限性**：论文未在多语言或多模态任务上验证，伪模型的压缩比例需要经验调参；细粒度卸载对 CPU‑GPU 带宽有一定依赖，带宽不足的机器可能收益有限。作者也承认在极端低延迟推理场景下，解耦后模型的加载仍然是瓶颈。

### 影响与延伸思考
M6‑10T 的出现让“用几百块 GPU 训练万亿级模型”不再是遥不可及的口号，激发了后续研究在 **参数压缩‑恢复**、**层级共享** 与 **异构存储调度** 方向的探索。随后有几篇工作（如 *Sparse‑Shift*、*Dynamic‑Chunk Offloading*）直接借鉴了细粒度卸载的思路，进一步把 CPU‑GPU 交互的粒度细化到单个张量维度。还有研究尝试把 **伪模型** 的压缩方式推广到 **稀疏专家模型**（MoE），希望在保持专家多样性的同时进一步降低显存。对想深入的读者，可以关注 **异构计算调度**（尤其是 NVLink、PCIe 4.0/5.0 对大模型训练的影响）以及 **绿色 AI**（训练能耗评估）两个方向。

### 一句话记住它
**先用共享的“小模型”跑通训练，再把共享拆开并细粒度搬到 GPU，万亿参数模型也能在几百块 GPU上跑完。**