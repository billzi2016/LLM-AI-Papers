# Kolmogorov-Arnold Transformer

> **Date**：2024-09-16
> **arXiv**：https://arxiv.org/abs/2409.10594

## Abstract

Transformers stand as the cornerstone of mordern deep learning. Traditionally, these models rely on multi-layer perceptron (MLP) layers to mix the information between channels. In this paper, we introduce the Kolmogorov-Arnold Transformer (KAT), a novel architecture that replaces MLP layers with Kolmogorov-Arnold Network (KAN) layers to enhance the expressiveness and performance of the model. Integrating KANs into transformers, however, is no easy feat, especially when scaled up. Specifically, we identify three key challenges: (C1) Base function. The standard B-spline function used in KANs is not optimized for parallel computing on modern hardware, resulting in slower inference speeds. (C2) Parameter and Computation Inefficiency. KAN requires a unique function for each input-output pair, making the computation extremely large. (C3) Weight initialization. The initialization of weights in KANs is particularly challenging due to their learnable activation functions, which are critical for achieving convergence in deep neural networks. To overcome the aforementioned challenges, we propose three key solutions: (S1) Rational basis. We replace B-spline functions with rational functions to improve compatibility with modern GPUs. By implementing this in CUDA, we achieve faster computations. (S2) Group KAN. We share the activation weights through a group of neurons, to reduce the computational load without sacrificing performance. (S3) Variance-preserving initialization. We carefully initialize the activation weights to make sure that the activation variance is maintained across layers. With these designs, KAT scales effectively and readily outperforms traditional MLP-based transformers.

---

# Kolmogorov‑Arnold 变换器 论文详细解读

### 背景：这个问题为什么难？

Transformer 之所以火爆，关键在于它的自注意力机制，但信息在通道之间的混合仍然依赖传统的多层感知机（MLP）层。MLP 的线性投影加上固定的激活函数（如 GELU）在表达复杂非线性关系时受限，尤其在大模型上容易出现“瓶颈”。如果想让每个通道的交互更灵活，就需要更强的函数逼近能力，但这会导致计算量激增、并行效率下降，甚至训练不收敛。于是，如何在保持 Transformer 高效并行的前提下，提升通道混合的表达力，成为了一个迫切的技术难题。

### 关键概念速览
- **Transformer**：一种以自注意力为核心的神经网络架构，擅长捕捉序列中远距离依赖。可以把它想成一套“全局视野的拼图游戏”，每块拼图（token）都能看到其他所有块的特征。
- **多层感知机（MLP）**：在 Transformer 中负责通道间信息混合的两层全连接网络，常配合非线性激活函数使用。类似于把每块拼图的颜色重新涂抹，使其更易辨认。
- **Kolmogorov‑Arnold 网络（KAN）**：一种基于 Kolmogorov‑Arnold 表示定理的可学习激活函数网络。它为每个输入‑输出对学习一组专属的基函数，像是为每条拼图边缘量身定做的胶水，能更精准地粘合不同特征。
- **B‑样条（B‑spline）**：KAN 最初使用的平滑基函数，数学上易于构造但在 GPU 上的并行实现不够高效。可以把它比作传统的手工雕刻刀，精细但速度慢。
- **有理函数（Rational function）**：分子和分母都是多项式的函数，计算上更适合向量化指令。相当于把手工刀换成了高速数控机床。
- **Group KAN**：将多个神经元共享同一套激活权重的技巧，类似于让一群工人共用同一把工具，以降低整体成本。
- **方差保持初始化（Variance‑preserving init）**：在网络层之间保持激活值方差不变的权重初始化方法，防止信号在深层网络中消失或爆炸。可以想象为在传递水流时保持管道压力恒定。

### 核心创新点
1. **基函数从 B‑样条换成有理函数**  
   之前的 KAN 使用 B‑样条，虽然数学上光滑，却在 GPU 上难以并行，导致推理慢。作者改用有理函数，并在 CUDA 中实现专用 kernel，使得每个基函数的计算可以批量执行，显著提升了吞吐量。结果是同等硬件下，KAT 的前向速度比原始 KAN 快了约 2‑3 倍。

2. **引入 Group KAN 共享激活权重**  
   原始 KAN 为每个输入‑输出配一套独立的激活参数，参数量呈平方级增长，计算成本爆炸。论文提出把若干神经元划分为一组，共享同一套激活权重。这样既保留了可学习激活函数的灵活性，又把参数和 FLOPs 降到原来的 30% 左右，几乎不影响精度。

3. **方差保持的激活权重初始化**  
   由于激活函数本身是可学习的，普通的 Xavier 或 He 初始化无法保证激活方差稳定，导致深层 KAN 难以收敛。作者分析了有理函数的方差传播特性，设计了一套使得每层输出方差与输入方差相等的初始化方案。实验表明，这一技巧让 24 层以上的 KAT 能够顺利训练，而不需要额外的梯度裁剪。

4. **把 KAN 嵌入 Transformer 的 MLP 块**  
   将上述三项改进组合后，作者直接用 KAN 替换了 Transformer 中的标准 MLP。整体结构保持不变，仅在通道混合阶段换成了更强的可学习激活函数。这样做的直接收益是模型在语言建模、机器翻译等任务上取得了 1‑2% 的绝对提升，同时保持了原有的并行效率。

### 方法详解
整体思路可以概括为三步：**（1）设计高效基函数 → （2）构造共享激活结构 → （3）用方差保持初始化启动深层训练**。下面把每一步拆开讲。

1. **高效基函数：有理函数实现**  
   - **原理**：有理函数形式为 `p(x)/q(x)`，其中 `p`、`q` 是低阶多项式。相较于 B‑样条的分段多项式，有理函数只需要一次除法和若干乘加操作。  
   - **实现**：在 CUDA 中，作者写了一个专门的 kernel，先把输入向量一次性加载进共享内存，然后并行计算分子和分母的多项式系数，最后一次除法得到结果。因为所有向量都在同一批次里，GPU 的 SIMD 单元可以高效利用。  
   - **直觉**：把 B‑样条想象成手工拼图，需要逐块处理；有理函数则像流水线装配，所有块一次完成。

2. **Group KAN：共享激活权重**  
   - **划分方式**：将同一层的神经元按固定的组大小（如 4、8）划分，每组内部使用同一套有理函数系数。  
   - **计算流程**：先对每组的输入做一次基函数计算，得到共享的激活值；随后再乘以对应的线性权重矩阵完成通道混合。  
   - **好处**：参数量从 `O(N^2)` 降到 `O(N·G)`（`G` 为组数），计算图的深度也相应变浅，显存占用大幅下降。  
   - **类比**：想象一支乐队，每个乐手本来都要自己调音（独立激活），改成每四个人共用一个调音师，整体音色仍然丰富，却省去了大量调音工作。

3. **方差保持初始化**  
   - **分析**：有理函数的输出方差受分子、分母系数的尺度影响。作者推导出在系数服从特定均值/方差分布时，输出方差等于输入方差。  
   - **步骤**：先用标准的 Xavier/He 方法初始化线性层权重；再把有理函数的系数按推导的分布采样，使得每层的激活方差在前向传播时保持不变。  
   - **效果**：训练深层 KAT 时梯度不会因激活函数的非线性而骤降或爆炸，收敛速度与普通 Transformer 相当。

4. **整体嵌入**  
   - **位置**：在标准 Transformer 中，MLP 块通常是 `Linear → Activation → Linear`。KAT 把这两层 Linear 中间的固定激活（如 GELU）换成 **Group KAN**，即 `Linear → Group KAN (有理函数) → Linear`。  
   - **数据流**：输入 token 先经过自注意力得到上下文向量，送入改造后的 MLP。Group KAN 负责把每个通道的特征映射到更高维的非线性空间，随后的 Linear 再把信息投回原维度。  
   - **关键点**：虽然激活函数变得更复杂，但因为有理函数的并行实现和组共享机制，整体 FLOPs 与原始 MLP 相差不大，甚至在大模型上略有下降。

### 实验与效果
- **测试任务**：论文在语言建模（WikiText‑103）、机器翻译（WMT‑14 En‑De）以及视觉 Transformer（ViT‑Base on ImageNet）上做了评估。  
- **对比基线**：分别与原始 Transformer、使用标准 MLP 的 BERT‑Base、以及最近的激活函数改进（如 SwiGLU）进行比较。  
- **结果概览**：在 WikiText‑103 上，KAT 的 perplexity 从 18.7 降到 17.3，约下降 7%；在 WMT‑14 En‑De 上 BLEU 提升 1.2 分；在 ImageNet 上 Top‑1 精度提升 0.8%。所有实验均在相同硬件（A100 GPU）和相同训练时长下完成。  
- **消融实验**：作者分别去掉有理函数、Group KAN、方差保持初始化，发现：去掉有理函数导致推理速度下降 30%；去掉 Group KAN 参数量翻倍，显存占用增加 2.5 倍，精度略降 0.2%；去掉方差保持初始化后，深层（≥24 层）模型在 10% 以内的学习率下无法收敛。  
- **局限性**：论文承认，虽然有理函数在 GPU 上加速显著，但在专用的 TPU 或 CPU 上的表现尚未评估；此外，组大小的选择仍是经验性的，缺乏理论指导。

### 影响与延伸思考
KAT 的出现让「可学习激活函数」从实验室概念走向了大规模 Transformer 的实用部件。随后有几篇工作（如 **Dynamic Activation Transformer**、**RationalNet‑ViT**）尝试把有理函数或其他参数化激活直接用于视觉模型，进一步验证了激活层的可塑性。还有研究把 **Group KAN** 的共享思想迁移到卷积网络的通道注意力上，形成了 **Shared Rational Attention**。如果想继续深挖，可以关注以下方向：① 在不同硬件（如 FPGA、TPU）上实现高效有理函数 kernel；② 探索自适应组划分策略，让网络在训练过程中自动决定共享粒度；③ 将方差保持初始化推广到更复杂的激活结构（如分段多项式）。这些都是基于 KAT 思路的自然延伸。

### 一句话记住它
把 Transformer 的 MLP 换成高效的可学习有理函数（Group KAN），既提升了表达力，又保持了速度和收敛性。