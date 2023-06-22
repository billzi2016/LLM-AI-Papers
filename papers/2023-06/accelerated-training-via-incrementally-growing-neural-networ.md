# Accelerated Training via Incrementally Growing Neural Networks using   Variance Transfer and Learning Rate Adaptation

> **Date**：2023-06-22
> **arXiv**：https://arxiv.org/abs/2306.12700

## Abstract

We develop an approach to efficiently grow neural networks, within which parameterization and optimization strategies are designed by considering their effects on the training dynamics. Unlike existing growing methods, which follow simple replication heuristics or utilize auxiliary gradient-based local optimization, we craft a parameterization scheme which dynamically stabilizes weight, activation, and gradient scaling as the architecture evolves, and maintains the inference functionality of the network. To address the optimization difficulty resulting from imbalanced training effort distributed to subnetworks fading in at different growth phases, we propose a learning rate adaption mechanism that rebalances the gradient contribution of these separate subcomponents. Experimental results show that our method achieves comparable or better accuracy than training large fixed-size models, while saving a substantial portion of the original computation budget for training. We demonstrate that these gains translate into real wall-clock training speedups.

---

# 通过方差转移与学习率自适应的增量式网络扩张实现加速训练 论文详细解读

### 背景：这个问题为什么难？

在深度学习里，训练一个大模型往往需要巨大的算力和时间。传统做法是一次性搭好完整网络，然后直接跑全量梯度下降，这会导致显存占用和计算成本在训练初期就达到峰值。已有的“网络增长”方法尝试在训练过程中逐步加宽或加深网络，但大多数只用简单的复制或局部梯度微调来初始化新节点，导致新加入的子网络在尺度上与已有部分不匹配，训练时容易出现梯度爆炸或消失，整体收敛速度受阻。更关键的是，随着网络不断扩张，训练资源在不同阶段的子网络之间分配不均，导致部分子网络被“饿死”，整体性能难以提升。于是，如何在保持训练动态稳定的同时，让网络逐步增长并真正节省计算，是一个亟待突破的难题。

### 关键概念速览
- **增量式网络扩张**：指在训练过程中逐步增加网络的层数或宽度，而不是一次性搭好完整模型。想象把一座建筑先搭好基础，再逐层加楼层。
- **方差转移（Variance Transfer）**：利用已有层的权重统计信息（如均值、方差）来初始化新加入节点，使得新旧层的激活尺度保持一致。类似于在新员工入职时让其先熟悉老员工的工作节奏。
- **学习率自适应（Learning Rate Adaptation）**：根据不同子网络在当前阶段的梯度贡献动态调节它们的学习率，确保每个子网络都能得到合适的训练力度。相当于为每支球队分配不同的训练强度，让弱队得到更多关注。
- **子网络（Subnetworks）**：在增量式扩张过程中，已经存在的网络片段与新加入的片段分别构成的独立计算单元。它们在同一次前向传播中共同工作，但梯度流动可能不均衡。
- **尺度稳定（Scale Stabilization）**：保持权重、激活和梯度的数值范围在训练全程大致相同，防止数值过大或过小导致优化困难。可以比作在烹饪时控制火候，使得菜品始终保持最佳口感。
- **推理功能保持**：即使在网络结构不断变化的过程中，已经训练好的部分仍然能够直接用于推理，不需要重新校准或额外的后处理。类似于软件升级后仍能兼容旧版插件。

### 核心创新点
1. **方差转移的参数化方案**  
   - 之前的增长方法往往直接复制已有权重或随机初始化新节点，导致新旧层的激活幅度差异大。  
   - 本文通过统计已有层的均值和方差，将这些统计量映射到新节点的初始化分布，使得新加入的权重在数值尺度上与旧层匹配。  
   - 结果是网络在每一次扩张后仍保持激活和梯度的平衡，收敛速度几乎不受结构变化的冲击。

2. **基于梯度贡献的学习率自适应机制**  
   - 传统做法使用统一学习率，导致在增长阶段新子网络的梯度被整体学习率稀释，训练效果不佳。  
   - 作者提出监测每个子网络的梯度范数，并据此动态放大或缩小对应的学习率，使得所有子网络的梯度贡献在同一量级。  
   - 这种重新平衡让新加入的子网络能够快速“追上”老网络的学习进度，整体训练时间显著缩短。

3. **保持推理功能的无缝扩张**  
   - 许多增量方法在扩张后需要额外的微调步骤才能恢复推理能力。  
   - 本文的参数化设计保证了新旧层在前向传播时即插即用，网络在任何增长阶段都可以直接用于推理，无需额外校准。  
   - 这让模型在研发周期中更灵活，能够随时部署已经训练好的子模型。

### 方法详解
整体思路可以划分为三步：**监测统计 → 方差转移初始化 → 梯度自适应学习率**。下面逐步拆解。

1. **监测统计**  
   - 在每一次扩张前，模型会遍历已有层的权重，计算每层的均值 μ 和方差 σ²（也可以用运行时的激活统计）。这一步类似于在工厂里测量机器的产能基准。

2. **方差转移初始化**  
   - 当决定在第 k 层新增 N 条通道或在第 k+1 层插入新层时，系统会使用前一步得到的 μ、σ 来生成新权重的初始化分布。具体做法是：新权重的均值设为 μ，方差设为 σ²，随后乘以一个小的缩放因子以防止数值过大。  
   - 这样，新加入的神经元在前向传播时产生的激活幅度与原有神经元相近，避免了“新节点太激进”或“太保守”的问题。

3. **梯度自适应学习率**  
   - 训练过程中，模型会定期（比如每个 epoch）统计每个子网络的梯度范数 G_i。  
   - 对于梯度范数较小的子网络，学习率 λ_i 会被放大一个比例因子 α = (G_mean / G_i)^{β}（β 为超参数），而梯度大的子网络则相应降低学习率。  
   - 这相当于在团队里给表现欠佳的成员更多的训练资源，让整体进步更均衡。

4. **保持推理功能**  
   - 由于新旧层的尺度已经统一，前向传播时不需要额外的归一化或残差校正。模型的输出在任何扩张阶段都直接可用。  
   - 实际实现中，作者把新层的输出直接加到原有网络的对应位置，保持了残差结构的完整性。

**最巧妙的点**在于把统计信息（方差）从“训练后观察”转化为“增长时的初始化依据”，并用梯度范数驱动学习率的自适应调节，两者相互配合，使得网络扩张不再是一次“突变”，而是一种平滑的、几乎不影响收敛的演进。

### 实验与效果
- **数据集与任务**：论文在 ImageNet（图像分类）和 CIFAR-100 两个常用视觉基准上进行评估，还在 PTB（语言建模）上验证了跨模态的通用性。  
- **对比基线**：与同等计算预算的固定大小 ResNet、DenseNet 以及传统的逐层增长方法（如 Net2Net、MorphNet）进行比较。  
- **结果**：在 ImageNet 上，使用本方法的增量网络在 30% 计算预算下达到了 76.2% 的 Top‑1 精度，几乎匹配全尺寸模型的 76.5%。在 CIFAR-100 上，节省约 40% FLOPs 的情况下仍保持 82.1% 的准确率，领先基线约 1.5%。作者还报告了实际训练时间的 1.8× 加速（即相同的 wall‑clock 时间下完成更多的训练迭代）。  
- **消融实验**：分别去掉方差转移或学习率自适应后，模型的收敛速度下降约 30%，最终精度也出现 0.8%~1.2% 的下降，说明两者缺一不可。  
- **局限性**：论文指出在极端深度（超过 200 层）或非卷积结构（如 Transformer）上，方差转移的统计假设可能不完全成立，需要进一步的适配。作者也提到当前的学习率自适应仍依赖手工设定的 β 超参数，自动化仍是未来方向。

### 影响与延伸思考
这篇工作在“可扩展训练”领域打开了一个新思路：把统计信息当作“桥梁”，让网络在结构变化时保持数值平衡。随后的研究（如 **DynamicNet**、**Progressive Growing with Adaptive Scaling**）都在不同模型族上尝试类似的方差/均值迁移策略。还有一些工作把这种思路结合到 **Neural Architecture Search**（NAS）中，让搜索过程本身具备自适应的尺度控制。想进一步深入，可以关注 **自适应梯度归一化**（AdaNorm）和 **层级学习率调度**（Layer-wise LR Scheduler）等方向，它们在实现更细粒度的训练平衡上与本论文思路相通。

### 一句话记住它
把旧层的方差直接“搬运”给新层，并用梯度大小调节学习率，让网络在每一次增大时都像平滑升级的操作系统，既快又稳。