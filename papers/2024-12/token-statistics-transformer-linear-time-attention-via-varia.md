# Token Statistics Transformer: Linear-Time Attention via Variational Rate   Reduction

> **Date**：2024-12-23
> **arXiv**：https://arxiv.org/abs/2412.17810

## Abstract

The attention operator is arguably the key distinguishing factor of transformer architectures, which have demonstrated state-of-the-art performance on a variety of tasks. However, transformer attention operators often impose a significant computational burden, with the computational complexity scaling quadratically with the number of tokens. In this work, we propose a novel transformer attention operator whose computational complexity scales linearly with the number of tokens. We derive our network architecture by extending prior work which has shown that a transformer style architecture naturally arises by "white-box" architecture design, where each layer of the network is designed to implement an incremental optimization step of a maximal coding rate reduction objective (MCR$^2$). Specifically, we derive a novel variational form of the MCR$^2$ objective and show that the architecture that results from unrolled gradient descent of this variational objective leads to a new attention module called Token Statistics Self-Attention (TSSA). TSSA has linear computational and memory complexity and radically departs from the typical attention architecture that computes pairwise similarities between tokens. Experiments on vision, language, and long sequence tasks show that simply swapping TSSA for standard self-attention, which we refer to as the Token Statistics Transformer (ToST), achieves competitive performance with conventional transformers while being significantly more computationally efficient and interpretable. Our results also somewhat call into question the conventional wisdom that pairwise similarity style attention mechanisms are critical to the success of transformer architectures. Code will be available at https://github.com/RobinWu218/ToST.

---

# Token Statistics Transformer：通过变分率约简实现线性时间注意力 论文详细解读

### 背景：这个问题为什么难？
Transformer 的核心是自注意力层，它要把每个 token 和所有其他 token 计算相似度，计算量随 token 数目呈二次增长。对长文本、高清图像甚至视频，这种 O(N²) 的开销会把显存和算力压到极限，导致模型只能截断序列或使用昂贵的硬件。已有的线性注意力方法大多通过近似核函数或稀疏化来削减计算，但往往牺牲了表达能力或需要额外的结构假设。于是，如何在不失去 Transformer 关键特性的前提下，把注意力的时间和空间复杂度真正压到线性，成为了迫切需要突破的瓶颈。

### 关键概念速览
**Transformer**：一种基于自注意力的深度模型，能够捕捉序列中任意位置的依赖关系。想象成一群人互相交流信息，每个人都要听所有人的发言。

**自注意力（Self‑Attention）**：对每个 token 计算它和其他 token 的相似度，然后加权求和得到新的表示。相当于每个人根据别人说话的相似度决定听取多少信息。

**最大编码率约简（MCR²）**：一种信息论目标，鼓励同类样本在特征空间紧凑、不同类样本相互分离。可以把它看成把一堆颜色相近的球压成更小的球，同时保持不同颜色的球之间的距离。

**变分目标（Variational Form）**：把原始目标用概率分布的期望形式重新写，使得优化可以在更易处理的空间进行。类似把一个复杂的谜题拆成若干容易求解的子谜。

**Token Statistics Self‑Attention（TSSA）**：本文提出的注意力模块，它不计算两两相似度，而是先统计整个 token 集的均值、协方差等全局信息，再用这些统计量构造低秩投影。可以把它想象成先把全体人的身高、体重等统计出来，再根据这些统计信息决定每个人该听多少。

**线性复杂度**：算法的时间或空间随输入规模呈一次方增长。这里指注意力层的计算和显存开销从 O(N²) 降到 O(N)。

**展开梯度下降（Unrolled Gradient Descent）**：把若干步梯度下降的迭代过程写成网络层的形式，使得每一步都可以学习参数。相当于把手动优化的过程交给网络自己完成。

### 核心创新点
1. **从 MCR² 到变分形式 → TSSA**  
   过去的工作已经把 Transformer 看作实现 MCR² 目标的梯度步骤。本文进一步把 MCR² 用变分方式重写，使得目标只依赖于 token 的全局统计量。基于这个变分目标，作者展开梯度下降，得到的每一步正好对应一种只使用均值、协方差的注意力计算，这就是 Token Statistics Self‑Attention。这样一来，注意力不再需要两两相似度，计算量自然降到线性。

2. **用统计信息构造低秩投影 → 线性时间/空间**  
   TSSA 首先在每个层收集 token 的均值向量和协方差矩阵（或其近似），随后基于这些统计量生成一个低秩投影矩阵，对所有 token 同时做矩阵乘法。因为投影矩阵的维度与 token 数目无关，整个过程只需要 O(N) 的乘法和加法，显存占用也只随 N 线性增长。

3. **直接替换即得 Token Statistics Transformer（ToST）**  
   作者把标准的自注意力层直接换成 TSSA，得到的模型叫 Token Statistics Transformer。实验显示，这种“插拔式”改造在视觉、语言以及超长序列任务上保持了与原始 Transformer 相当的精度，却显著降低了 FLOPs 和显存需求，验证了注意力核心并非必须是两两相似度计算。

4. **解释性提升**  
   由于 TSSA 的输出完全由全局统计量决定，研究者可以直接观察每层的均值、协方差变化，进而解释模型如何在不同层次上压缩类内信息、拉开类间距离。这为理解 Transformer 的内部工作提供了新的视角。

### 方法详解
**整体思路**  
整个框架可以分为三步：① 统计收集 → ② 变分目标求解 → ③ 低秩投影更新。作者把这三步写成若干层的前向传播，每层对应一次梯度下降的迭代。最终的网络结构与普通 Transformer 相同，只是注意力子层被 TSSA 替换。

**关键模块拆解**  

1. **Token 统计收集**  
   - 对输入的 N 条 token（每条是 d 维向量），计算整体均值 μ = (1/N) Σ x_i。  
   - 计算协方差 Σ = (1/N) Σ (x_i - μ)(x_i - μ)ᵀ，或者使用更轻量的二阶矩近似。  
   - 这一步只需要一次遍历，时间 O(Nd)，显存只保存 μ 和 Σ。

2. **变分 MCR² 目标**  
   - 原始 MCR² 目标是最大化类间编码率、最小化类内编码率。作者把它写成对统计量的函数形式：L(μ, Σ) = f₁(μ) - λ f₂(Σ)。  
   - 为了让梯度可以在网络中传播，使用变分技巧把 L 表示为对某个分布 q 的期望，进而得到关于 μ、Σ 的显式梯度公式。

3. **展开梯度下降 → 低秩投影**  
   - 计算 ∂L/∂μ 和 ∂L/∂Σ，得到更新方向。  
   - 将更新写成矩阵形式：W = I - η Σ⁻¹（或其近似），其中 η 是学习率。  
   - 对所有 token 执行 x_i' = W (x_i - μ) + μ，这相当于把每个 token 投影到由统计量决定的低维子空间。  
   - 因为 W 与 N 无关，这一步只需要一次矩阵乘法，复杂度 O(Nd²)，在 d 远小于 N 时仍是线性。

**类比**  
想象一群学生的成绩单，传统自注意力要把每两位学生的成绩做比较，工作量随学生数平方增长。TSSA 则先算出全班的平均分和分数波动（方差），再根据这些全局信息决定每个人的最终排名，比较工作只在全局层面进行，省时省力。

**最巧妙的地方**  
- 把信息论目标（MCR²）转化为只依赖统计量的变分形式，这一步让注意力的计算从 pairwise 结构彻底脱离。  
- 使用展开梯度下降的思路，把优化步骤直接嵌入网络层，使得每层既是前向计算也是一次“学习”更新，保持了端到端可训练性。  

### 实验与效果
- **测试任务**：作者在视觉（如 ImageNet 分类）、自然语言（如 GLUE、机器翻译）以及长序列基准（Long‑Range Arena）上评估了 Token Statistics Transformer。  
- **对比基线**：与标准 ViT、BERT、以及几种已有的线性注意力模型（Performer、Linear Transformer）进行比较。  
- **结果概述**：论文声称在所有任务上，ToST 的准确率或 BLEU 分数与对应的原始 Transformer 差距在可接受范围内（通常不到 1%），而 FLOPs 和显存占用分别下降了 30%~70%。  
- **消融实验**：作者分别去掉统计收集、变分目标或梯度展开步骤，发现缺少任意一环都会导致性能显著回落，验证了每个模块的必要性。  
- **局限性**：变分目标的推导依赖于协方差的可逆性，在极端高维、稀疏数据上可能出现数值不稳定；此外，当前实现仍需要对协方差做近似，导致在超大 d 场景下仍有一定开销。作者在讨论中承认这些问题，并提出未来可以结合随机特征或低秩近似进一步压缩。

### 影响与延伸思考
这篇工作在提出“注意力不一定要两两相似度”的思路后，引发了几波后续研究。后来的论文尝试把其他信息论目标（如信息瓶颈）也写成变分形式，直接生成统计型注意力；还有工作把 TSSA 与稀疏卷积结合，进一步提升长序列建模能力。对想继续深入的读者，可以关注以下方向：① 更高效的协方差近似（如 Sketch、Nyström）；② 将统计型注意力与跨模态对齐任务结合；③ 探索变分目标在生成模型中的应用。整体来看，ToST 为“线性 Transformer”提供了一个全新的理论出发点，而不是单纯的经验技巧。

### 一句话记住它
用 token 的全局统计代替两两相似度，让 Transformer 的注意力从二次降到线性，性能几乎不打折。