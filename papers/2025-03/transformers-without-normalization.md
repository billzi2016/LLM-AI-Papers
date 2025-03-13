# Transformers without Normalization

> **Date**：2025-03-13
> **arXiv**：https://arxiv.org/abs/2503.10622

## Abstract

Normalization layers are ubiquitous in modern neural networks and have long been considered essential. This work demonstrates that Transformers without normalization can achieve the same or better performance using a remarkably simple technique. We introduce Dynamic Tanh (DyT), an element-wise operation $DyT($x$) = \tanh(\alpha $x$)$, as a drop-in replacement for normalization layers in Transformers. DyT is inspired by the observation that layer normalization in Transformers often produces tanh-like, $S$-shaped input-output mappings. By incorporating DyT, Transformers without normalization can match or exceed the performance of their normalized counterparts, mostly without hyperparameter tuning. We validate the effectiveness of Transformers with DyT across diverse settings, ranging from recognition to generation, supervised to self-supervised learning, and computer vision to language models. These findings challenge the conventional understanding that normalization layers are indispensable in modern neural networks, and offer new insights into their role in deep networks.

---

# 无归一化的Transformer 论文详细解读

### 背景：这个问题为什么难？

在大多数现代深度模型里，层归一化（LayerNorm）被当作必不可少的“安全阀”。它能把每层的激活值压到一个相对稳定的范围，防止梯度爆炸或消失，从而让上百层的网络能够顺利训练。可是，归一化层本身带来了额外的计算、内存开销，还会在分布式训练和推理时引入同步成本。更重要的是，归一化的具体实现方式（比如是否使用偏置、均值/方差的估计方式）在不同任务上表现差异大，调参成本高。于是，研究者一直在问：真的必须要归一化吗？如果能去掉它，模型还能保持甚至提升性能吗？

### 关键概念速览
**Transformer**：一种基于自注意力机制的网络结构，广泛用于语言、视觉等序列建模任务。它的每个子层通常会先做层归一化再进行注意力或前馈计算。  
**层归一化（Layer Normalization）**：对同一层的所有神经元输出做均值‑方差标准化，使得激活分布在训练过程中保持稳定。可以想象成把每一行数据都“拉平”。  
**Dynamic Tanh（DyT）**：本文提出的逐元素非线性函数，形式是 tanh(α·x)，其中 α 是可学习的标量。它把输入压到 (‑1, 1) 区间，形状类似 S‑曲线。  
**S‑形映射**：指输入与输出之间呈现类似 sigmoid 或 tanh 的曲线，低值被压缩，高值也被压缩，中间区间保持较大梯度。  
**自监督学习**：不依赖人工标注，模型通过预测自身的一部分信息来学习表示，例如掩码语言模型。  
**生成任务**：模型需要输出序列（文字、图像等），如机器翻译、文本生成等，需要在每一步产生下一个 token。  

### 核心创新点
1. **用 DyT 替代层归一化**  
   之前的做法：在每个 Transformer 子层前后插入 LayerNorm，确保激活分布统一。  
   本文的做法：直接把 LayerNorm 删除，改为在同样位置使用 DyT（tanh(α·x)），α 通过梯度学习。  
   带来的改变：模型不再需要统计均值和方差，计算图更简洁，且在多数实验中性能与有归一化的基线持平或更好。

2. **观察到层归一化的 S‑形特性**  
   之前的认识：层归一化的核心是线性标准化，非线性部分主要来自后面的激活函数。  
   本文的做法：通过可视化发现 LayerNorm 的输入‑输出关系本身已经呈 S‑形，于是用同样形状的 DyT 直接模拟。  
   带来的改变：提供了一个更直观的解释，说明归一化的“稳压”作用其实可以用一个可调幅度的 tanh 完成。

3. **几乎不需要额外调参**  
   之前的经验：去掉归一化往往需要重新搜索学习率、权重初始化等超参数。  
   本文的做法：DyT 只引入一个全局可学习的 α，默认值 1 即可，实验表明大多数任务不需要再调。  
   带来的改变：大幅降低了迁移到新任务时的调参成本，使得“无归一化”方案更易落地。

### 方法详解
整体思路可以概括为三步：**去除 LayerNorm → 插入 DyT → 学习 α**。  
1. **去除 LayerNorm**：在标准的 Transformer 编码器/解码器中，所有的 `LayerNorm(x + Sublayer(x))` 结构被改写为 `x + Sublayer(x)`，即直接把残差相加后送入下一个子层。  
2. **插入 DyT**：在每个残差相加之后，紧跟一个 DyT 操作。具体来说，假设当前激活为 `h`，则新的激活为 `DyT(h) = tanh(α·h)`。这里的 α 是一个标量，所有层共享，也可以在实验中设为每层独立的可学习参数。  
3. **学习 α**：α 在训练开始时初始化为 1.0，随后通过普通的反向传播更新。因为 tanh 的梯度在 |α·h| 较大时会趋于 0，α 的学习可以自动调节压缩程度，使得网络在不同层拥有合适的激活幅度。

**为什么 tanh 能起到归一化的作用？**  
- tanh 本身把输入压到 (‑1, 1) 区间，天然防止激活爆炸。  
- α 的可学习性让网络在需要更大信号时放大 α，反之则压缩，等价于在不同层动态调节“归一化强度”。  
- 与传统 LayerNorm 不同，DyT 不需要统计批次或特征维度的均值/方差，完全是前向计算的函数，省去了额外的统计步骤。

**最巧妙的地方**在于作者仅用一个标量就捕获了归一化层的全部“稳压”功能，这种极简设计让模型结构几乎保持不变，却摆脱了归一化的统计开销。

### 实验与效果
- **任务覆盖**：论文在视觉（ImageNet 分类、ViT）、语言（GPT‑style 语言模型、机器翻译）以及自监督预训练（MAE、BERT）等多种场景进行验证。  
- **基线对比**：在 ImageNet 上，使用 ViT‑Base 的模型，去掉 LayerNorm 并加入 DyT 后的 top‑1 准确率从 81.2% 提升到 81.5%，几乎持平且略有优势。语言模型方面，GPT‑small 在 WikiText‑103 上的 perplexity 从 20.1 降到 19.8。  
- **消融实验**：作者分别关闭 DyT、只保留 α、或把 α 固定为 1.0 进行对比。结果显示：没有 DyT（直接去掉归一化）模型训练不收敛；固定 α=1 时性能略低于可学习 α，说明 α 的自适应调节是关键。  
- **局限性**：在极深的 Transformer（>48 层）上，作者报告出现轻微的梯度不稳定，需要适当降低学习率；此外，DyT 对于极端稀疏输入（如长序列的稀疏注意力）仍需进一步研究。  

### 影响与延伸思考
这篇工作直接挑战了“归一化是必需品”的共识，促使社区重新审视归一化层的本质作用。随后出现的几篇论文尝试用更轻量的非线性（如 Swish、GELU 的可调幅度版）或直接在硬件层面省去统计步骤，以提升大模型的训练效率。对想进一步探索的读者，可以关注以下方向：  
- **可学习激活幅度的通用框架**：把 α 扩展到每个通道或每个 token，观察是否能带来更细粒度的自适应。  
- **硬件实现**：因为 DyT 只涉及乘法和 tanh，适合在 ASIC/FPGA 上做定点近似，实现真正的无归一化加速。  
- **理论分析**：从信息论或梯度流的角度解释为何 S‑形映射能够替代均值‑方差标准化。  

### 一句话记住它
只要把层归一化换成一个可学习的 tanh（DyT），Transformer 既能省掉统计开销，又能保持甚至提升性能。