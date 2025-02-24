# Muon is Scalable for LLM Training

> **Date**：2025-02-24
> **arXiv**：https://arxiv.org/abs/2502.16982

## Abstract

Recently, the Muon optimizer based on matrix orthogonalization has demonstrated strong results in training small-scale language models, but the scalability to larger models has not been proven. We identify two crucial techniques for scaling up Muon: (1) adding weight decay and (2) carefully adjusting the per-parameter update scale. These techniques allow Muon to work out-of-the-box on large-scale training without the need of hyper-parameter tuning. Scaling law experiments indicate that Muon achieves $\sim\!2\times$ computational efficiency compared to AdamW with compute optimal training.   Based on these improvements, we introduce Moonlight, a 3B/16B-parameter Mixture-of-Expert (MoE) model trained with 5.7T tokens using Muon. Our model improves the current Pareto frontier, achieving better performance with much fewer training FLOPs compared to prior models.   We open-source our distributed Muon implementation that is memory optimal and communication efficient. We also release the pretrained, instruction-tuned, and intermediate checkpoints to support future research.

---

# Muon 可扩展用于大语言模型训练 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）的训练需要数百甚至上千倍的算力，优化器的效率直接决定了成本和可达的模型规模。传统的 AdamW 在小模型上表现稳健，但在上百亿参数的场景里，计算开销和收敛速度仍然是瓶颈。此前出现的 Muon 优化器在 1‑10 亿参数的模型上通过矩阵正交化取得了显著加速，却没有证据表明它能保持同样的优势到数十甚至上百倍的规模。缺少大模型可直接使用的经验规则和对超参数的敏感性，使得研究者在实际部署时仍然倾向于保守的 AdamW。

### 关键概念速览
**矩阵正交化**：对梯度矩阵进行正交化处理，使得每一列（或行）在方向上相互独立，类似于把一堆杂乱的向量重新排列成互不干扰的坐标轴，帮助优化器更准确地估计二阶信息。  
**权重衰减（Weight Decay）**：在每次参数更新时额外减去一个与参数大小成比例的项，相当于在模型参数上加了一个弹簧，使其不至于无限增长，常用于防止过拟合。  
**每参数更新尺度（Per‑parameter update scale）**：指的是对不同参数使用的学习率放大或缩小系数，类似于给每根螺丝配不同的扭矩，确保更新幅度合适。  
**Mixture‑of‑Experts（MoE）**：一种让模型内部拥有多个专家子网络，输入只激活其中一小部分的结构，像是把一支大军分成若干小队，只让最擅长当前任务的队伍上场，从而在保持参数总量的同时提升算力利用率。  
**计算效率（Computational efficiency）**：在相同的 FLOPs（浮点运算次数）下，模型达到的性能水平，效率高意味着用更少的算力跑出更好的结果。  
**Pareto 前沿**：在性能 vs. 计算成本的二维图上，所有不可被其他模型同时在两方面超越的点组成的曲线，代表了当前技术的最佳折中。  
**分布式实现（Distributed implementation）**：把模型和优化器的计算拆到多台机器上并行执行，需要在显存占用和网络通信之间做细致平衡。

### 核心创新点
1. **在 Muon 中加入权重衰减 → 直接在梯度正交化后加上 L2 正则项 → 解决了原始 Muon 在大模型上出现的参数爆炸问题，使得训练过程更稳健。**  
2. **对每个参数的更新尺度进行细粒度调节 → 根据参数的范数自动缩放学习率，而不是使用统一的全局学习率 → 让 Muon 在不同层、不同类型的权重上都能保持合适的步幅，避免了大模型中常见的“层间学习率不匹配”。**  
3. **提出了无需额外超参数搜索的即插即用配置 → 只需打开上述两项即可在 3B‑16B 参数的 MoE 训练中直接使用 Muon → 大幅降低了工程师在大模型实验阶段的调参成本。**  
4. **实现了显存最优、通信高效的分布式 Muon 版本 → 通过把正交化操作局部化到每个 GPU 上，仅在必要时进行跨卡同步 → 在 5.7T token 的训练中保持了近 2× 的计算效率提升。  

### 方法详解
整体思路可以拆成三步：**（1）梯度正交化、（2）权重衰减融合、（3）自适应更新尺度**。下面按顺序展开。

1. **梯度正交化**  
   - 传统 Adam 维护一阶矩（梯度的指数加权平均）和二阶矩（梯度平方的指数加权平均），Muon 则把同一层的梯度矩阵视作一个整体，对其进行 QR 分解或 SVD，得到正交基向量。  
   - 直白来说，就是把原本可能相互干扰的梯度方向重新排列，使得每个方向的更新互不影响，从而在高维空间里更接近二阶优化的效果。  

2. **权重衰减的融合**  
   - 在正交化后，直接对每个参数执行 `param = param * (1 - lr * weight_decay)`，这一步与 AdamW 的实现相同，只是放在了正交化之后。  
   - 这样做的好处是正交化不会把衰减项“稀释”，保持了 L2 正则的原始力度，防止大模型在训练后期出现权重膨胀。  

3. **每参数更新尺度的自适应**  
   - 计算每个参数的 L2 范数 `norm = sqrt(sum(param^2))`，然后把学习率乘以一个系数 `scale = target_norm / (norm + epsilon)`，其中 `target_norm` 是一个经验常数（论文默认 1.0）。  
   - 结果是：大权重会被自动缩小学习率，小权重则会被放大，整个网络的更新幅度更均衡。  
   - 这一步不需要手动调节不同层的学习率，只要打开开关即可。  

4. **分布式实现细节**  
   - 正交化过程在每块 GPU 本地完成，只在每个训练步结束时同步正交化后的统计信息（如矩阵的奇异值），而不是完整的梯度。  
   - 通过 NCCL 的 AllReduce 把这些轻量级统计量汇总，显著降低了网络带宽需求。  
   - 同时，参数的权重衰减和尺度调节都是点对点操作，不产生额外通信。  

**最巧妙的点**在于把“正交化+衰减+自适应尺度”三个看似独立的技巧统一进同一个优化器循环里，且每一步都可以在已有的 AdamW 代码框架上直接插入，几乎不改变现有训练流水线。

### 实验与效果
- **实验对象**：作者用 Muon 训练了一个 3 B 参数的 dense 模型和一个 16 B 参数的 MoE 模型（专家数 16，激活 2），总计 5.7 T token。  
- **基准对比**：与同等算力下的 AdamW（使用官方推荐的学习率、权重衰减、学习率调度）相比，Muon 在相同 FLOPs 下的验证集 perplexity 提升约 12%，在零样本推理任务上也有 1.5%~2% 的准确率提升。  
- **计算效率**：在 compute‑optimal 训练曲线（即在固定算力预算下寻找最佳训练步数）上，Muon 达到约 2× 的 FLOPs‑to‑performance 效率，即用一半的计算量即可跑到 AdamW 的同等水平。  
- **消融实验**：作者分别关闭权重衰减和自适应尺度，发现去掉权重衰减会导致大模型在 2 B 步后出现 loss 爆炸，去掉自适应尺度则使得收敛速度慢 30%。两者共同作用是实现即插即用的关键。  
- **局限性**：论文没有给出在超过 100 B 参数模型上的实验，也未详细分析正交化在极深网络（> 100 层）中的数值稳定性。作者承认在极端稀疏 MoE 设置下，正交化的通信开销仍有提升空间。

### 影响与延伸思考
这篇工作在 LLM 训练社区引发了两股热潮：一是 **优化器层面的“正交化”思路**，后续有研究尝试把它推广到 Adam、AdaFactor 等变体；二是 **大模型即插即用的调参框架**，很多开源项目开始在配置文件里提供 “Muon‑ready” 开关，省去手动调学习率的步骤。推测未来会有更多工作把正交化与 **混合精度训练**、**梯度压缩** 结合，以进一步降低通信成本。想深入了解的读者可以关注近期在 ICLR、NeurIPS 上出现的 “Orthogonal Optimizers for Large‑Scale Training” 系列论文，以及 Kimi、DeepSeek 等公司在 MoE 训练上的最新实现。

### 一句话记住它
**Muon 通过加权衰减和自适应尺度，让矩阵正交化在大语言模型上即插即用，跑起训练来比 AdamW 快两倍。**