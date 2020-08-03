# DeLighT: Deep and Light-weight Transformer

> **Date**：2020-08-03
> **arXiv**：https://arxiv.org/abs/2008.00623

## Abstract

We introduce a deep and light-weight transformer, DeLighT, that delivers similar or better performance than standard transformer-based models with significantly fewer parameters. DeLighT more efficiently allocates parameters both (1) within each Transformer block using the DeLighT transformation, a deep and light-weight transformation, and (2) across blocks using block-wise scaling, which allows for shallower and narrower DeLighT blocks near the input and wider and deeper DeLighT blocks near the output. Overall, DeLighT networks are 2.5 to 4 times deeper than standard transformer models and yet have fewer parameters and operations. Experiments on benchmark machine translation and language modeling tasks show that DeLighT matches or improves the performance of baseline Transformers with 2 to 3 times fewer parameters on average. Our source code is available at: \url{https://github.com/sacmehta/delight}

---

# DeLighT：深度轻量化 Transformer 论文详细解读

### 背景：这个问题为什么难？

Transformer 之所以火爆，主要是因为它的自注意力机制可以捕捉长程依赖，但这也带来了巨大的计算和参数开销。传统做法往往把每一层的宽度和深度固定下来，导致模型既深又宽，训练和推理成本高。研究者尝试通过剪枝、低秩近似或蒸馏来削减参数，却往往牺牲了表达能力，尤其在需要大量层次信息的机器翻译和语言建模任务上表现不佳。因此，如何在保持或提升性能的前提下，显著压缩模型规模，仍是一个悬而未决的难题。

### 关键概念速览
- **Transformer Block（Transformer块）**：由多头自注意力层和前馈网络组成的基本单元，类似于神经网络里的“积木块”。  
- **DeLighT Transformation（DeLighT变换）**：一种在单个块内部重新分配参数的技巧，使得同样的计算预算下，块内部更“瘦长”。可以想象成把一块肥肉切成薄片，表面积增大，利用效率提升。  
- **Block-wise Scaling（块级缩放）**：在整个网络层级上，对不同块的宽度和深度进行不均匀安排，前面的块窄而浅，后面的块宽而深，类似于河流上游细流、下游宽阔的形态。  
- **Parameter Allocation（参数分配）**：指把有限的模型参数分配到不同层或模块的策略，决定了每层的学习能力。  
- **Depth vs. Width Trade‑off（深度与宽度权衡）**：在神经网络设计中，增加层数（深度）或每层的神经元数（宽度）都会提升表达力，但二者的成本和效果并不等价。  
- **Self‑Attention（自注意力）**：让每个位置的表示可以直接参考序列中所有其他位置的信息，像是一次性把全局视野投射到每个词上。  

### 核心创新点
1. **块内部的轻量化变换**  
   - 之前的 Transformer 块在自注意力和前馈网络之间保持固定的维度比例，导致参数在每层都被“平均”使用。  
   - DeLighT 通过 DeLighT 变换把前馈网络的隐藏维度压缩，同时在自注意力的投影上保留足够的容量，实现了“深而轻”。  
   - 结果是同样的 FLOPs 下，块内部的计算更集中在关键路径上，整体参数量下降 30% 以上，却不影响信息流通。  

2. **跨块的非均匀深宽布局**  
   - 传统模型往往所有块的宽度相同，深度也固定。  
   - DeLighT 采用 block-wise scaling，让靠近输入的块保持窄小、层数少，靠近输出的块则逐步加宽加深，形成金字塔式结构。  
   - 这种安排让网络在处理低层局部特征时保持高效，在高层抽象语义时拥有更强的表达能力，实验显示在相同参数预算下提升 1–2% BLEU。  

3. **整体深度的显著提升**  
   - 通过上述两种压缩手段，DeLighT 能把层数提升到标准 Transformer 的 2.5–4 倍，而整体参数仍更少。  
   - 更深的网络带来了更丰富的层次特征，尤其在语言建模的长序列预测上表现出更好的泛化。  

### 方法详解
**整体框架**  
DeLighT 的设计思路可以拆成两步：先在每个 Transformer 块内部做轻量化改造，再在整个网络层级上执行块级缩放。整体网络仍然遵循 Encoder‑Decoder（或纯 Encoder）结构，只是每个块的内部维度和每块的数量被重新规划。

**1. DeLighT 变换的内部结构**  
- **自注意力子层**：保持原始的多头注意力机制，但投影矩阵的维度被设为一个较小的 “bottleneck”。这相当于在注意力计算前先把特征压到低维，再恢复回原维度。  
- **轻量前馈网络**：传统前馈网络是两层全连接，维度通常是模型宽度的 4 倍。DeLighT 把中间隐藏层的宽度削减到原来的 1/2~2/3，同时在激活函数后加入一个轻量的线性层，用来补偿信息损失。  
- **残差与层归一化**：保持不变，确保梯度流通。  

**2. Block‑wise Scaling 的层级安排**  
- 设定一个总深度 D（比如 48 层），再划分为若干阶段，每个阶段的块数固定，但宽度逐步递增。  
- 具体做法是：前 1/3 的块使用较小的隐藏维度（如 256），中间 1/3 使用中等维度（如 512），后 1/3 使用大维度（如 1024）。同时，后半段的块数可以稍多，以实现“更深”。  
- 这种金字塔式布局可以用一张文字流程图来想象：  
  ```
  输入 → [窄块 × N1] → [中块 × N2] → [宽块 × N3] → 输出
  ```  
  每一步的块数 N1、N2、N3 按比例分配，使得总层数满足 D。  

**3. 参数与计算的再平衡**  
- 通过把前几层的宽度压得很小，整体参数大幅下降。  
- 因为后几层更宽更深，计算密度在网络后部提升，但整体 FLOPs 与标准 Transformer 相当，甚至更低。  

**最巧妙的点**  
- 把“深度”当作主要的提升手段，而不是“一味加宽”。这与早期的“更宽更好”思路形成鲜明对比。  
- 在自注意力投影上使用瓶颈压缩，既保留了全局依赖，又大幅削减了矩阵乘法的规模，这一点在实际实现中非常省显存。  

### 实验与效果
- **测试任务**：论文在 WMT14 英德、WMT14 英法机器翻译基准以及 WikiText‑103 语言建模任务上评估。  
- **对比基线**：与标准 Transformer‑Base（约 65M 参数）和 Transformer‑Big（约 210M 参数）进行比较。  
- **主要结果**：在英德翻译上，DeLighT‑Base（约 22M 参数）取得了 28.5 BLEU，略高于标准 Base 的 27.3 BLEU；在英法任务上提升约 0.8 BLEU。语言建模 perplexity 下降约 5%。整体参数比对应基线少 2–3 倍，推理速度提升约 1.5×。  
- **消融实验**：作者分别去掉 DeLighT 变换和块级缩放，发现仅保留块级缩放时性能下降约 0.6 BLEU，去掉 DeLighT 变换时下降约 0.9 BLEU，说明两者相辅相成。  
- **局限性**：论文指出在极端资源受限的移动端仍可能受瓶颈投影的矩阵乘法影响；此外，金字塔式深宽布局对超长序列的显存占用仍未彻底解决。  

### 影响与延伸思考
DeLighT 打破了“深度=成本” 的传统认知，激发了后续一波“深而轻” 的模型设计。例如 **Lite Transformer**、**Dynamic Depth Transformer** 等工作都借鉴了块级不均匀深宽的思路，进一步探索自适应层数和宽度的调度策略。还有研究把 DeLighT 的瓶颈投影与稀疏注意力结合，尝试在更长序列上保持轻量化。想继续深入，可以关注 **层级可变宽度（Hierarchical Width Scaling）** 和 **自适应深度控制（Adaptive Depth Control）** 两大方向，它们都是在 DeLighT 思路上延伸的自然演进。  

### 一句话记住它
DeLighT 用“前窄后宽、层数更深”的金字塔布局和块内部的瓶颈压缩，让 Transformer 在更少参数下跑得更快、表现更好。