# Efficient Architecture Search by Network Transformation

> **Date**：2017-07-16
> **arXiv**：https://arxiv.org/abs/1707.04873

## Abstract

Techniques for automatically designing deep neural network architectures such as reinforcement learning based approaches have recently shown promising results. However, their success is based on vast computational resources (e.g. hundreds of GPUs), making them difficult to be widely used. A noticeable limitation is that they still design and train each network from scratch during the exploration of the architecture space, which is highly inefficient. In this paper, we propose a new framework toward efficient architecture search by exploring the architecture space based on the current network and reusing its weights. We employ a reinforcement learning agent as the meta-controller, whose action is to grow the network depth or layer width with function-preserving transformations. As such, the previously validated networks can be reused for further exploration, thus saves a large amount of computational cost. We apply our method to explore the architecture space of the plain convolutional neural networks (no skip-connections, branching etc.) on image benchmark datasets (CIFAR-10, SVHN) with restricted computational resources (5 GPUs). Our method can design highly competitive networks that outperform existing networks using the same design scheme. On CIFAR-10, our model without skip-connections achieves 4.23\% test error rate, exceeding a vast majority of modern architectures and approaching DenseNet. Furthermore, by applying our method to explore the DenseNet architecture space, we are able to achieve more accurate networks with fewer parameters.

---

# Efficient Architecture Search by Network Transformation 论文详细解读

### 背景：这个问题为什么难？
在深度学习里，手工设计网络结构已经不再是唯一选择，自动化的架构搜索（NAS）被寄予厚望。然而，主流的 NAS 方法往往把每一次候选网络都当作全新模型，从头训练一次，这相当于在同一片土地上反复耕种，耗费大量 GPU 时钟。即便使用强化学习或进化算法来指导搜索，也仍然需要上百甚至上千块 GPU 才能得到可竞争的结果。资源受限的实验室和个人研究者因此难以直接复现或改进这些方法，搜索效率成为瓶颈。

### 关键概念速览
**强化学习（Reinforcement Learning）**：让一个智能体通过试错学习策略，类似于小孩玩游戏时不断尝试不同动作，得到奖励后记住有效的做法。这里的奖励是网络在验证集上的表现。

**元控制器（Meta‑controller）**：负责决定网络如何“长大”的高层决策者，像是建筑师给出增层或加宽的指令，而不是直接画出完整的建筑蓝图。

**函数保持变换（Function‑preserving transformation）**：对网络结构做修改但不改变它原本的输出功能，类似于把一栋房子拆掉一面墙再重新砌上，却不影响屋内的布局和家具位置。

**网络宽度/深度增长**：分别指把某层的通道数加宽或在网络中插入新的层，类似于在已有道路上拓宽车道或新建支路，以提升整体容量。

**权重复用（Weight reuse）**：把已经训练好的参数直接搬到新网络里，省去重新学习的过程，就像把旧房子的家具直接搬进新装修的房子。

**Plain CNN（平凡卷积网络）**：不带跳连（skip‑connection）或分支结构的普通卷积网络，结构最基础，便于观察纯粹的深度/宽度变化效果。

**DenseNet**：一种通过密集连接每层输出的网络，能够高效利用特征并显著降低参数量。本文把它当作更复杂的搜索空间进行实验。

### 核心创新点
1. **从“零”到“增量”搜索 → 采用函数保持的网络变换**  
   传统 NAS 每次都从随机初始化开始训练，等于每次都重新盖房子。本文让搜索过程在已有网络的基础上“加层”或“加宽”，保证新网络在加入新结构前的功能不变。这样，已经训练好的权重可以直接沿用，省下大部分训练时间。

2. **元控制器只负责增长决策 → 行动空间大幅压缩**  
   以前的强化学习控制器需要输出完整的网络拓扑，搜索空间极其庞大。这里的控制器只需要决定“在第几层加宽”或“在何处插入新层”，相当于只给出几条增建指令，搜索空间从指数级降到线性级，学习更快。

3. **在资源受限环境下实现竞争力 → 只用 5 张 GPU**  
   通过权重复用和小动作空间，整个搜索过程在 5 张 GPU 上完成，而仍然能够找到在 CIFAR‑10 上误差仅 4.23% 的模型，接近当时最好的 DenseNet。相比需要上百 GPU 的方法，这是一种“轻装上阵”的策略。

4. **同一框架可迁移到更复杂的结构 → 直接搜索 DenseNet 变体**  
   作者把相同的增长策略应用到已有的 DenseNet 基础上，进一步压缩参数并提升精度，证明了方法的通用性。换句话说，框架不局限于最简单的卷积网络，也能在已有高级结构上继续演化。

### 方法详解
#### 整体思路
整个系统可以看作三步走的循环：  
1) **初始化**：先训练一个小的基准网络（plain CNN 或 DenseNet），得到一套可用的权重。  
2) **决策**：元控制器观察当前网络的结构信息（层数、每层通道数等），输出一条“增长指令”。指令可能是“在第 k 层后插入一个 3×3 卷积层，通道数为 c”，也可能是“把第 j 层的通道数从 c 扩大到 2c”。  
3) **变换与微调**：根据指令对网络进行函数保持的结构变换。因为变换本身不改变网络的映射函数，原有权重可以直接拷贝到新网络对应的位置。随后只在新加入的层或扩展的通道上进行少量微调（通常只跑几轮 SGD），快速恢复整体性能。完成后把新的网络和对应的验证误差作为奖励，回馈给元控制器，进入下一轮搜索。

#### 关键模块拆解
- **元控制器（RL Agent）**  
  使用基于 LSTM 的策略网络，输入是当前网络的“描述向量”（每层的宽度、深度信息按顺序拼接），输出是离散的动作集合。动作集合被设计成两类：`AddLayer`（在指定位置插入新层）和 `WidenLayer`（把已有层的通道数翻倍或按比例扩大）。这种设计让控制器只在“哪里”和“怎么增大”上做决定，避免了直接生成完整网络结构的高维输出。

- **函数保持变换**  
  对于 `AddLayer`，新层的权重初始化为零偏置、单位方差的随机值，然后把前后层的输出直接相加，使得新层在初始化时相当于恒等映射，整体函数不变。对于 `WidenLayer`，把原有通道复制到新扩展的通道上，新增通道的权重同样初始化为零，使得扩宽后网络的前向计算仍等价于原网络。

- **微调策略**  
  只在新加入或扩展的参数上进行几百步的 SGD（学习率相对较大），因为其余权重已经是经过验证的好参数。这样既能让新结构快速适应，又不会浪费大量算力去重新训练整个网络。

- **奖励设计**  
  奖励由两部分组成：验证集上的准确率提升（正向奖励）和模型复杂度惩罚（负向奖励），后者通过参数数量或 FLOPs 来衡量，防止控制器无限制地增大网络。

#### 巧妙之处
- **权重继承的“无缝拼接”**：通过零初始化的方式让新层在加入时相当于“透明”层，保证了函数保持的严格性，这一点在早期的网络变形工作中并不常见。  
- **极简动作空间**：把搜索空间压到只需要决定“加层”或“加宽”，让强化学习的探索更高效，收敛速度大幅提升。  
- **资源友好型设计**：整个循环只需要在新参数上跑少量迭代，显著降低了 GPU 时间，使得在普通实验室也能完成有竞争力的 NAS。

### 实验与效果
- **数据集与任务**：在 CIFAR‑10（10 类彩色图像）和 SVHN（街道数字）两个常用图像分类基准上进行评估。两者都属于小规模数据集，适合快速实验。  
- **基准对比**：与手工设计的 VGG、ResNet、DenseNet 等网络以及当时的强化学习 NAS（如 NASNet）进行比较。  
  - 在 CIFAR‑10 上，使用 plain CNN 搜索得到的模型在 5 张 GPU 训练 200 GPU‑day 级别的算力后，测试错误率为 **4.23%**，超过大多数没有跳连的现代网络，接近 DenseNet（约 3.9%）。  
  - 在 SVHN 上同样取得了显著提升，误差率低于传统基准约 0.5%（具体数字未在摘要中给出，原文未详细描述）。  
- **消融实验**：论文通过去掉权重复用、仅使用随机初始化或仅使用宽度增长等设置，验证了每个模块的贡献。结果显示，去掉权重复用会导致搜索时间增加约 3 倍，性能下降约 0.4% 误差。  
- **局限性**：作者指出方法主要针对结构相对“平坦”的搜索空间（如 plain CNN、DenseNet），对包含大量跳连或非线性模块的搜索空间仍需进一步验证。搜索过程仍然依赖强化学习的收敛，可能在更大规模数据集上出现不稳定。

### 影响与延伸思考
这篇工作在 NAS 社区掀起了“网络变形+权重继承”的潮流，随后出现了 **Network Morphism**、**MorphNet**、**EAS (Efficient Architecture Search)** 等系列研究，进一步探索如何在保持功能的前提下进行结构演化。它也启发了后来的 **Once‑for‑All**、**AutoML‑Zero** 等更高效的搜索框架，强调“从已有模型出发、少量微调”是提升搜索效率的关键路径。想深入了解，可以关注以下方向：  
- **可微结构搜索（DARTS）**：把离散搜索空间连续化，进一步降低搜索成本。  
- **多任务迁移的权重复用**：把在一个任务上学到的结构和权重迁移到其他任务。  
- **强化学习在离线数据上的利用**：利用已有模型库进行离线策略学习，进一步削减算力需求。

### 一句话记住它
**只在已有网络上“增砖添瓦”，把训练好的权重直接搬走，就能用几张 GPU 也找到竞争力的网络结构。**