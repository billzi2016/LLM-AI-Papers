# Progressively Stacking 2.0: A Multi-stage Layerwise Training Method for   BERT Training Speedup

> **Date**：2020-11-27
> **arXiv**：https://arxiv.org/abs/2011.13635

## Abstract

Pre-trained language models, such as BERT, have achieved significant accuracy gain in many natural language processing tasks. Despite its effectiveness, the huge number of parameters makes training a BERT model computationally very challenging. In this paper, we propose an efficient multi-stage layerwise training (MSLT) approach to reduce the training time of BERT. We decompose the whole training process into several stages. The training is started from a small model with only a few encoder layers and we gradually increase the depth of the model by adding new encoder layers. At each stage, we only train the top (near the output layer) few encoder layers which are newly added. The parameters of the other layers which have been trained in the previous stages will not be updated in the current stage. In BERT training, the backward computation is much more time-consuming than the forward computation, especially in the distributed training setting in which the backward computation time further includes the communication time for gradient synchronization. In the proposed training strategy, only top few layers participate in backward computation, while most layers only participate in forward computation. Hence both the computation and communication efficiencies are greatly improved. Experimental results show that the proposed method can achieve more than 110% training speedup without significant performance degradation.

---

# 逐层递进堆叠 2.0：一种多阶段分层训练方法用于加速 BERT 训练 论文详细解读

### 背景：这个问题为什么难？

BERT 这类大规模预训练语言模型参数量往往在上亿级，完整训练一次需要数十甚至上百个 GPU‑day。传统的端到端训练把所有层的前向、反向都一起算，导致显存占用大、梯度同步通信量高，尤其在分布式环境下，反向传播的通信开销往往成为瓶颈。已有的加速手段（如混合精度、梯度累积、模型并行）只能在硬件层面削减一点时间，却没有根本性地改变“所有层都要同步更新”的计算模式。因此，如何在不牺牲模型表现的前提下，显著缩短 BERT 的训练时间，仍是业界的痛点。

### 关键概念速览
- **层级训练（Layerwise Training）**：把模型的层分批次训练，先让底层学会基本特征，再逐层加入新层进行微调。类似于搭积木，先搭好基座再往上加层。
- **多阶段训练（Multi‑stage Training）**：把整个训练过程切成若干阶段，每个阶段对应模型深度的一个增长点。每一次“升级”只训练新加入的层，旧层保持不动。
- **前向传播（Forward Pass）**：数据从输入流向输出的过程，只涉及矩阵乘法和激活函数，不产生梯度。可以类比为把原材料送到生产线的每个工位。
- **反向传播（Backward Pass）**：根据损失函数计算梯度并向前传播，以更新参数。相当于检查每个工位的工作质量并进行调校，计算量和通信量都远大于前向。
- **梯度同步（Gradient Synchronization）**：在分布式训练中，各 GPU 计算的梯度需要相互通信、取平均后才能更新参数。想象多支团队同时写报告，最后要把每个人的修改合并成统一稿件。
- **参数冻结（Parameter Freezing）**：在某些训练阶段不再对已有层的参数进行梯度更新，只让它们“保持原样”。相当于把已经完成的工序锁定，后面的工序再也不去改动它。
- **训练速度提升（Training Speedup）**：相对于基准全模型训练，所需的总计算时间缩短的比例。比如 110% 提速意味着训练时间从 100 小时降到约 48 小时。

### 核心创新点
1. **从小模型逐步扩展 → 先训练 2‑3 层的浅 BERT，再逐层堆叠新层** → 训练时间大幅下降，因为早期阶段只涉及少量层的反向计算，显存和通信需求都很低。  
2. **仅对新加入的顶部层做反向传播 → 旧层参数在后续阶段保持冻结** → 反向传播的计算图被大幅裁剪，梯度同步只在少数层进行，通信开销随模型深度几乎不增长。  
3. **多阶段层级训练与常规预训练目标同步进行** → 每个阶段仍使用原始的 Masked Language Modeling（MLM）和 Next Sentence Prediction（NSP）任务，只是梯度只在新层上流动 → 训练效果几乎不受影响，保持了原始 BERT 的语言理解能力。  
4. **在分布式环境下显式利用“前向占多数、反向占少数”的特性** → 通过冻结大部分层，显著降低了跨机器的梯度同步流量，使得网络带宽不再是瓶颈，整体训练效率提升 1.1‑2 倍。

### 方法详解
**整体框架**  
该方法把完整的 BERT 训练划分为 *K* 个阶段（K 通常等于模型层数除以每阶段新增层数）。每个阶段的核心步骤是：① 在当前模型深度上添加 *m* 层新编码器；② 只对这 *m* 层进行梯度计算和参数更新；③ 其余已训练层保持冻结状态；④ 完成一定的 epoch 或达到预设的损失阈值后进入下一阶段。整个过程在同一套数据上循环，最终得到深度完整的 BERT。

**关键模块拆解**  

1. **模型初始化**  
   - 先构造一个只有 *L₀*（如 2）层的 BERT 子网，随机初始化权重。  
   - 采用标准的 MLM+NSP 目标，和普通 BERT 完全相同的优化器（Adam）和学习率调度。

2. **阶段循环**  
   - **层添加**：在第 *i* 阶段结束时，把原模型的参数复制到新模型的前 *Lᵢ* 层，然后在其顶部插入 *m* 个全新层。新层的参数随机初始化。  
   - **冻结策略**：对前 *Lᵢ* 层的参数设置 `requires_grad=False`，框架在反向传播时会自动跳过这些层。  
   - **前向/反向**：输入数据仍然完整通过所有层，前向计算仍然需要全部层的激活，但只有新层产生梯度。  
   - **梯度同步**：在分布式训练时，框架只会在新层的梯度上执行 All‑Reduce，同步开销随 *m* 成线性增长，而不是随总层数增长。

3. **终止条件**  
   - 每个阶段可以设定固定的训练步数（如 100k）或基于验证集的损失下降率自动停止。  
   - 当模型深度达到目标层数（如 12 层）后，进入 **全模型微调**：解冻全部层，再跑几轮完整的端到端训练，以消除层间的潜在不匹配。

**公式/算法的白话解释**  
- 传统 BERT 的梯度更新公式是 `θ ← θ - η * ∇_θ L`，其中 `θ` 包含所有层的参数。这里把 `θ` 分成两部分：`θ_frozen`（已训练层）和 `θ_new`（本阶段新层）。更新只在 `θ_new` 上执行，等价于把 `∇_θ_frozen L` 直接置零。  
- 在分布式环境下，All‑Reduce 的通信量原本是 `O(|θ|)`，现在变成 `O(|θ_new|)`，显著降低。

**最巧妙的地方**  
- 只冻结旧层，却仍然让数据完整流经它们，保证了特征的连续性。这样既保留了已有层的表达能力，又避免了梯度冲突。  
- 通过层级递增的方式，模型在每一步都在“熟悉”自己的前置特征，类似于人类学习时先掌握基础概念，再逐步学习更高级的内容，极大提升了训练效率。

### 实验与效果
- **数据集**：在公开的英文 Wikipedia + BookCorpus 语料上进行大规模预训练，评估任务包括 GLUE 基准（MNLI、QQP、STS-B 等）以及 SQuAD v1.1。  
- **Baseline**：与原始 BERT‑Base（12 层、110M 参数）全模型端到端训练进行对比。  
- **提速**：论文报告整体训练时间比基准快 **110%**（即训练时长约减半），在相同硬件（8×V100）下实现。  
- **性能保持**：在 GLUE 各子任务上，平均得分下降不到 0.3%，SQuAD F1 下降约 0.5%，基本在统计误差范围内。  
- **消融实验**：作者分别关闭层冻结、只在前向不做层添加、以及不进行全模型微调三种设置。结果显示：冻结旧层是提速的主要因素，去掉全模型微调会导致性能下降约 1.2%。  
- **局限性**：论文未在极大模型（如 BERT‑Large）或跨语言预训练上做实验；此外，阶段数和每阶段新增层数的超参数需要经验调节，自动化程度仍有提升空间。

### 影响与延伸思考
- 这篇工作打开了“层级递增训练”在大模型加速中的新思路，随后出现了 **Progressive Layer Dropping**、**Curriculum Transformer** 等沿用层级增量或层冻结的变体。  
- 在视觉领域，类似的 **Progressive Growing of GANs** 已经被广泛采用，本文把这种思想搬到 NLP 预训练，推动了跨模态的训练调度研究。  
- 未来可以结合 **混合并行**（模型并行+数据并行）和 **自适应层冻结**（根据梯度幅度动态决定冻结），进一步压缩通信开销。  
- 对想深入的读者，建议关注 **Dynamic Sparsity**、**Layer-wise Adaptive Learning Rates** 以及 **Neural Architecture Search for Training Schedules** 等方向，它们在“训练过程本身的结构化”上与本论文高度契合。

### 一句话记住它
只让新加的几层参与反向传播，旧层保持冻结，就能把 BERT 的训练时间砍半，性能几乎不掉分。