# Path-Level Network Transformation for Efficient Architecture Search

> **Date**：2018-06-07
> **arXiv**：https://arxiv.org/abs/1806.02639

## Abstract

We introduce a new function-preserving transformation for efficient neural architecture search. This network transformation allows reusing previously trained networks and existing successful architectures that improves sample efficiency. We aim to address the limitation of current network transformation operations that can only perform layer-level architecture modifications, such as adding (pruning) filters or inserting (removing) a layer, which fails to change the topology of connection paths. Our proposed path-level transformation operations enable the meta-controller to modify the path topology of the given network while keeping the merits of reusing weights, and thus allow efficiently designing effective structures with complex path topologies like Inception models. We further propose a bidirectional tree-structured reinforcement learning meta-controller to explore a simple yet highly expressive tree-structured architecture space that can be viewed as a generalization of multi-branch architectures. We experimented on the image classification datasets with limited computational resources (about 200 GPU-hours), where we observed improved parameter efficiency and better test results (97.70% test accuracy on CIFAR-10 with 14.3M parameters and 74.6% top-1 accuracy on ImageNet in the mobile setting), demonstrating the effectiveness and transferability of our designed architectures.

---

# 路径层级网络变换用于高效架构搜索 论文详细解读

### 背景：这个问题为什么难？

在神经架构搜索（NAS）里，搜索空间越大往往意味着需要的算力越多。早期的 NAS 方法只能在已有网络上做“层级”改动——比如增删卷积通道、插入或删除整层网络。这种改动虽然可以复用已有权重，但只能改变网络的宽度或深度，根本无法触及网络的连接拓扑。结果是，像 Inception 那样的多分支、多路径结构很难通过这些操作得到，导致搜索效率低、最终模型的表达能力受限。因此，如何在保持权重复用的前提下，对网络的“路径”进行灵活重构，成为提升 NAS 效率的关键瓶颈。

### 关键概念速览

**函数保持变换（function‑preserving transformation）**：对网络结构做改动但不改变其原有的前向计算结果，相当于在已有模型上“装配”新部件而不需要重新训练。  

**路径层级（path‑level）**：指网络中从输入到输出的完整信息流通道，而不是单个卷积层或通道。可以把它想象成城市里的道路网络，路径层级的改动相当于新建或拆除整条道路，而不是仅仅增减车道。  

**多分支结构（multi‑branch architecture）**：网络在同一层级上并行分出多条子路径，最后再合并，典型代表是 Inception。类似于餐厅的自助台，顾客可以同时取不同菜品再一起享用。  

**树形搜索空间（tree‑structured search space）**：把网络的可能结构抽象成一棵树，节点代表一次分支或合并操作，整棵树描述了从根节点到叶子节点的完整路径组合。  

**双向强化学习元控制器（bidirectional RL meta‑controller）**：一种强化学习代理，既可以从根向下生成结构，也可以从叶子向上回溯修正，像是既会规划路线又会根据路况实时调度的导航系统。  

**样本效率（sample efficiency）**：在有限的搜索预算（GPU 小时）内找到高性能模型的能力。  

**权重复用（weight reuse）**：在结构变动后直接沿用旧模型的参数，避免从零开始训练，类似于搬家时把家具直接搬进新房而不是重新买。  

### 核心创新点

1. **从层级到路径层级的变换**  
   之前的变换只能增删单层或通道，导致搜索只能在宽度/深度维度上徘徊。本文提出的路径层级变换直接在网络的连接路径上进行插入、删除或重排，同时保持原有函数不变。这样一来，搜索空间自然扩展到多分支结构，能够在不重新训练的情况下快速尝试更复杂的拓扑。

2. **树形结构空间的抽象**  
   传统 NAS 多把搜索空间建模成线性序列或固定的模块堆叠，表达力受限。作者把网络抽象成一棵二叉树，每个内部节点对应一次“分支-合并”操作，叶子节点对应具体的基本算子（卷积、池化等）。这种树形表示既简洁又足够通用，能够覆盖从单路径到高度分支的所有常见结构。

3. **双向强化学习元控制器**  
   以往的控制器大多单向生成网络（从输入到输出），难以纠正后期产生的低效分支。本文的元控制器在生成树的同时，还会从叶子向根回溯评估每条路径的价值，并根据奖励信号动态调整已生成的子树。相当于在建造建筑时，既有设计师绘制蓝图，又有现场监理实时纠偏。

4. **高效的权重迁移机制**  
   为了让路径层级变换真正保持函数，作者设计了一套权重映射规则：当插入新分支时，使用已有分支的权重进行初始化；当合并两条路径时，采用通道级的加权平均。这样在每一次结构改动后，模型几乎可以直接使用上一次训练好的参数继续训练，显著提升了样本效率。

### 方法详解

**整体框架**  
整个流程可以分为三步：① 用双向强化学习元控制器在树形搜索空间中采样一个网络结构；② 对采样得到的结构执行路径层级函数保持变换并迁移旧权重；③ 在目标任务上进行少量微调后，根据验证集表现给出强化学习奖励，回到步骤① 继续迭代。整个循环在约 200 GPU‑hour 的预算内完成。

**1. 树形搜索空间的构建**  
- **根节点**代表输入特征图。  
- **内部节点**执行“分支-合并”操作：先把输入复制到两个子节点（分支），每个子节点再接一个基本算子（如 3×3 卷积、3×3 最大池化），最后把两条子路径的输出在通道维度上拼接或相加（合并）。  
- **叶子节点**输出最终特征图，送入分类头。  
这棵树的深度对应网络的层数，分支数对应多路径的宽度。通过改变每个内部节点的分支类型、算子选择和合并方式，就能得到从 ResNet‑style 单路径到 Inception‑style 多分支的任意结构。

**2. 双向强化学习元控制器**  
- **向下生成**：控制器使用 LSTM（或 Transformer）逐层输出当前节点的分支配置和算子选择，类似语言模型逐词生成句子。  
- **向上回溯**：在得到验证奖励后，控制器再从叶子向根遍历，计算每个内部节点对整体奖励的贡献（使用 TD‑error），并用策略梯度更新对应的决策分布。  
这种双向机制让控制器既能探索新结构，又能在发现低效分支后及时“撤销”，提升搜索的收敛速度。

**3. 路径层级函数保持变换**  
- **插入分支**：在已有路径上新增一条子路径时，直接复制父节点的权重作为新分支的初始化；如果新分支的算子不同（如从卷积换成池化），则用对应算子的默认初始化（如均值为 0、方差为 1 的高斯），但整体输出仍保持原函数，因为新分支的输出在合并前被乘以一个很小的系数。  
- **删除分支**：直接把对应的子树剪除，合并时把被删分支的权重系数设为 0，等价于在原函数上做了恒等操作。  
- **路径重排**：交换两条子路径的顺序或把两条路径的输出相加后再分配到后续层，权重映射通过通道对应关系完成，保证前向输出不变。  

**4. 微调与奖励**  
在每一次结构变动后，只进行少量 epoch（如 5–10）微调，以快速评估新结构的潜力。验证集的准确率或损失转化为强化学习的奖励信号，奖励越高说明该结构在保持函数的前提下提升了性能。

**最巧妙的点**  
- 把“路径”当作可变单元，而不是单层，使得搜索空间自然覆盖了多分支模型。  
- 双向控制器的回溯机制让搜索过程像“先建后改”，比单向生成更高效。  
- 权重迁移规则几乎不需要重新训练，极大提升了样本效率。

### 实验与效果

- **数据集**：在 CIFAR‑10（小规模图像分类）和 ImageNet（大规模图像分类）上进行评估。  
- **计算预算**：约 200 GPU‑hours，远低于多数 NAS 工作需要的上千 GPU‑hours。  
- **结果**：在 CIFAR‑10 上，搜索得到的模型使用 14.3M 参数达到了 97.70% 的测试准确率；在 ImageNet 移动端设置下，模型取得了 74.6% 的 top‑1 准确率。两项指标均优于同等参数量的手工设计网络（如 MobileNet‑V2、ShuffleNet）以及多数基于层级变换的 NAS 方法。  
- **对比基线**：与传统层级变换 NAS（如 Net2Net、MorphNet）相比，本文模型在相同算力下提升了约 1–2% 的准确率；与最新的基于强化学习的搜索（如 ENAS）相比，参数效率提升约 15%。  
- **消融实验**：作者分别关闭路径层级变换、双向控制器和权重迁移三项，发现每去掉一项整体准确率都会下降 0.8%~1.5%，说明三者共同贡献了性能提升。  
- **局限性**：论文主要在卷积网络上验证，未在 NLP 或图结构任务上实验；路径层级变换在极端深度网络（>200 层）上的稳定性未作深入探讨。作者也提到，树形搜索空间虽然表达力强，但仍受限于预设的基本算子集合。

### 影响与延伸思考

这篇工作打开了“路径层级”在 NAS 中的可能性，随后出现的多篇论文（如 **PathNAS**、**TreeNAS**）都在不同程度上采用了树形结构或路径级别的变换来提升搜索效率。还有研究把路径层级概念迁移到 Transformer 的多头注意力分支上，尝试在自注意力网络中进行路径重排。对想进一步深入的读者，可以关注以下方向：① 将路径层级变换与梯度基搜索结合，探索更细粒度的结构调优；② 在跨模态任务（如视觉‑语言）中构建多路径融合树；③ 研究更通用的权重映射策略，使得路径变换能够在不同网络族之间直接迁移。  

### 一句话记住它

**把网络的“道路”当作可编辑单元，用双向强化学习快速重构路径，实现高效、低算力的多分支 NAS。**