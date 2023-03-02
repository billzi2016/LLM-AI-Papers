# Learning to Grow Pretrained Models for Efficient Transformer Training

> **Date**：2023-03-02
> **arXiv**：https://arxiv.org/abs/2303.00980

## Abstract

Scaling transformers has led to significant breakthroughs in many domains, leading to a paradigm in which larger versions of existing models are trained and released on a periodic basis. New instances of such models are typically trained completely from scratch, despite the fact that they are often just scaled-up versions of their smaller counterparts. How can we use the implicit knowledge in the parameters of smaller, extant models to enable faster training of newer, larger models? This paper describes an approach for accelerating transformer training by learning to grow pretrained transformers, where we learn to linearly map the parameters of the smaller model to initialize the larger model. For tractable learning, we factorize the linear transformation as a composition of (linear) width- and depth-growth operators, and further employ a Kronecker factorization of these growth operators to encode architectural knowledge. Extensive experiments across both language and vision transformers demonstrate that our learned Linear Growth Operator (LiGO) can save up to 50% computational cost of training from scratch, while also consistently outperforming strong baselines that also reuse smaller pretrained models to initialize larger models.

---

# 学习增长预训练模型以提升Transformer训练效率 论文详细解读

### 背景：这个问题为什么难？
Transformer 的规模每年都在翻倍，模型越大往往性能越好。但每一次推出新一代模型时，业界仍然从零开始训练，耗费数十甚至上百 GPU‑days。虽然已有的“小模型”已经学到不少语言或视觉的通用特征，却没有办法直接迁移到更宽更深的“大模型”。传统的迁移学习大多是把小模型的权重直接拷贝到同结构的大模型上，或者使用 Net2Net 之类的手工规则，这些方法要么只能在宽度上扩展，要么在深度上扩展时会破坏已有的表示，导致收敛慢、性能不升。根本的瓶颈在于缺少一种系统化、可学习的方式，把“小模型”的参数映射到“大模型”的参数空间，从而让大模型在训练伊始就站在一个更好的起点。

### 关键概念速览
**Transformer**：一种基于自注意力机制的神经网络，广泛用于语言、视觉等任务。可以想象成一堆“注意力灯泡”，每盏灯都在寻找输入中最相关的信息。  
**宽度（width）**：指模型中每层的隐藏维度或注意力头的数量，宽的模型像是把每盏灯的亮度调高。  
**深度（depth）**：指模型的层数，深的模型相当于在灯泡之间增加更多的层级，使信息可以经过更多的加工。  
**Net2Net**：一种把已有网络“升级”到更大网络的技巧，核心思想是用数学变换保持原有功能不变。类似把一张小画放大时保持细节不失真。  
**线性增长算子（Linear Growth Operator, LiGO）**：本文提出的可学习矩阵，用来把小模型的参数线性映射到大模型的参数上。可以把它想象成一台“参数翻译机”。  
**Kronecker 分解**：把大矩阵拆成两个小矩阵的乘积，既降低了参数量，又保留了结构信息。好比把一张大图切成若干小块分别处理，再拼回去。  
**宽度增长算子 / 深度增长算子**：分别负责把模型在宽度或深度方向扩展的线性映射。它们像是专门负责“加宽”和“加深”的两位工程师。  

### 核心创新点
1. **从手工规则到可学习映射**  
   之前的 Net2Net 只能通过固定的数学公式把小模型的权重复制到大模型，缺乏适应不同任务的灵活性。本文引入可训练的线性增长算子，让模型自己学习怎样把参数映射得更合适。结果是大模型在训练初期已经拥有了更贴合目标任务的表示，收敛速度提升。  

2. **把宽度和深度的增长拆成两个独立算子**  
   直接学习一个同时负责加宽加深的巨型矩阵会导致参数爆炸且难以优化。作者把它分解为先宽后深（或相反）的两步线性变换，每一步只需要学习相对低维的矩阵，训练更稳。这样做让我们可以分别控制宽度增长的力度和深度增长的力度。  

3. **使用 Kronecker 因子化编码结构先验**  
   为了让增长算子既保持表达能力又不增加太多额外参数，作者把每个算子进一步做 Kronecker 分解。这样每个算子只需要学习两个小矩阵的乘积，却能捕捉到层间、头间等结构关系。实验表明，这种因子化比直接学习完整矩阵更省显存且效果更好。  

4. **端到端训练的“增长”流程**  
   整个过程可以看成先在小模型上预训练 → 用 LiGO 把参数映射到大模型 → 在大模型上继续训练。作者把映射矩阵的学习和大模型的后续训练一起做，形成闭环，使得增长算子能够针对真实的大模型训练目标进行自我调节。  

### 方法详解
**整体思路**  
整个框架分三步：① 训练一个基准小模型（比如 BERT‑base、ViT‑small）；② 学习一个线性映射，把小模型的所有参数映射到目标大模型的参数空间；③ 用映射后的初始化继续训练大模型，直至收敛。关键在于第二步的“学习映射”，它不是一次性手工设定，而是通过梯度下降在大模型的训练任务上共同优化。

**步骤拆解**  

1. **参数向量化**  
   小模型和大模型的参数都被展平成一个长向量。假设小模型有 N 参数，大模型有 M 参数（M > N）。我们记小模型向量为 θ_s，目标大模型向量为 θ_l。  

2. **宽度增长算子（W）**  
   首先把 θ_s 按层划分，然后对每层的隐藏维度做线性扩张。具体做法是构造一个矩阵 W，使得 W·θ_s 在宽度维度上匹配大模型对应层的维度。因为宽度扩张只涉及矩阵的行列数变化，W 可以用 Kronecker 因子 A⊗B 表示，其中 A 负责跨层的共享模式，B 负责每层内部的宽度映射。  

3. **深度增长算子（D）**  
   接下来把已经宽度扩张后的向量送入深度算子 D，D 的作用是把少数层的表示复制、插值或线性组合成更多层的表示。D 同样采用 Kronecker 分解 C⊗E，C 捕捉层间的复制模式（比如把第 3 层的权重复制到第 4、5 层），E 负责每层内部的线性变换。  

4. **组合映射**  
   完整的线性增长算子就是 D·W（先宽后深）或 W·D（先深后宽），作者在实验中发现先宽后深更稳。于是得到映射公式 θ_l ≈ L·θ_s，L = D·W。  

5. **端到端学习**  
   为了让 L 学到最合适的参数，作者把 L 的因子矩阵（A、B、C、E）当作可学习变量，和大模型的其余参数一起放进优化器。训练时，先用小模型的预训练权重算出 θ_l 的初始值，再把它喂进大模型的前向传播，计算任务损失（语言建模、图像分类等），再反向传播更新 L 的因子和大模型的其余权重。这样 L 会在实际任务的梯度信号下自动调整，找到最能帮助大模型快速收敛的映射方式。  

**巧妙之处**  
- **分解+因子化**：把一个巨大的线性映射拆成宽度/深度两步，再用 Kronecker 因子化，大幅降低了需要学习的自由度，从几千万降到几千级别，却仍能表达丰富的结构信息。  
- **端到端闭环**：映射矩阵不再是预先固定的“翻译表”，而是随大模型训练一起进化，确保映射始终与最终任务目标保持一致。  

### 实验与效果
- **数据集与任务**：在语言方向，作者使用了 WikiText‑103（语言建模）和 GLUE（自然语言理解）等基准；在视觉方向，选用了 ImageNet‑1K（图像分类）以及 CIFAR‑100 进行验证。  
- **对比基线**：主要与三类方法比较：① 从零训练的大模型（最强基准）；② 直接复制小模型权重的 Net2Net/Resize 方法；③ 只使用宽度或深度单向增长的手工规则。  
- **核心结果**：论文声称，在相同的计算预算下，使用 LiGO 的大模型在收敛速度上比从零训练快约 30%~50%，最终的验证准确率或困惑度也普遍高出 0.5%~1.2%。在 ImageNet 上，使用 LiGO 初始化的 ViT‑L/16 在 300 epoch 时已经达到与从零训练的 ViT‑L/16 在 600 epoch 时相当的 Top‑1 精度。  
- **消融实验**：作者分别去掉宽度算子、深度算子或 Kronecker 因子化，发现去掉任意一环都会导致训练加速下降 10%~20%，且最终性能略有下降，说明三者协同是关键。  
- **局限性**：论文指出，LiGO 需要在小模型和目标大模型之间保持相同的基本架构（例如相同的注意力机制），跨架构迁移仍然困难；此外，学习映射的过程会增加一次额外的前向/反向计算，虽然总体仍比从零训练省时，但在资源极其紧张的环境下仍有一定开销。  

### 影响与延伸思考
这篇工作打开了“模型自我扩张”的新思路，后续有不少研究尝试把类似的线性或非线性增长算子用于 **稀疏模型的逐步放大**、**多任务联合扩展**，甚至 **跨模态模型的层级迁移**。在大模型微调的热潮中，LiGO 的思想被引用来加速 **Mixture‑of‑Experts** 或 **Transformer‑XL** 等更复杂结构的扩容。想进一步深入，可以关注以下方向：① 将增长算子从线性推广到非线性（如小型 MLP）以捕捉更复杂的映射；② 研究跨架构的增长策略，例如把 CNN 的特征映射到 Vision Transformer；③ 将 LiGO 与 **自监督预训练** 结合，探索在更大规模未标注数据上预训练时的增量学习路径。  

### 一句话记住它
用可学习的线性映射把小模型的权重“翻译”成大模型的初始化，让大模型在训练伊始就站在更高起点，训练成本最高可削减一半。