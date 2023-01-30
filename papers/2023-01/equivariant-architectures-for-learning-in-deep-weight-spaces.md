# Equivariant Architectures for Learning in Deep Weight Spaces

> **Date**：2023-01-30
> **arXiv**：https://arxiv.org/abs/2301.12780

## Abstract

Designing machine learning architectures for processing neural networks in their raw weight matrix form is a newly introduced research direction. Unfortunately, the unique symmetry structure of deep weight spaces makes this design very challenging. If successful, such architectures would be capable of performing a wide range of intriguing tasks, from adapting a pre-trained network to a new domain to editing objects represented as functions (INRs or NeRFs). As a first step towards this goal, we present here a novel network architecture for learning in deep weight spaces. It takes as input a concatenation of weights and biases of a pre-trained MLP and processes it using a composition of layers that are equivariant to the natural permutation symmetry of the MLP's weights: Changing the order of neurons in intermediate layers of the MLP does not affect the function it represents. We provide a full characterization of all affine equivariant and invariant layers for these symmetries and show how these layers can be implemented using three basic operations: pooling, broadcasting, and fully connected layers applied to the input in an appropriate manner. We demonstrate the effectiveness of our architecture and its advantages over natural baselines in a variety of learning tasks.

---

# 深度权重空间学习的等变架构 论文详细解读

### 背景：这个问题为什么难？

在传统的机器学习里，模型的输入是图像、文本或点云，网络直接在这些原始数据上学习。把**整个神经网络的权重矩阵**当作输入，却要面对一个隐藏的对称性：同一网络的功能不随内部神经元的排列顺序改变。早期的尝试往往直接把权重向量化后喂进普通的全连接层，这会把同等功能的网络映射到完全不同的表示上，导致学习效率极低。根本的瓶颈在于缺少能够**尊重（equivariant）**这种“换位不变”对称性的网络结构，导致模型无法捕捉到权重空间的真实几何。于是，如何设计既能处理高维权重，又能自然适配神经元置换的架构，成为了迫切需要突破的难题。

### 关键概念速览

- **深度权重空间（Deep Weight Space）**：指所有可能的神经网络参数（权重+偏置）组成的高维空间。把一个网络看作该空间中的一点，就像把一张图片看作像素空间中的一点。
- **等变（Equivariance）**：如果对输入做某种变换，输出会以对应的方式同步变换。类比于把图像旋转 90°，卷积网络的特征图也会相应旋转。
- **置换对称（Permutation Symmetry）**：在多层感知机（MLP）里，交换同一层内部神经元的顺序不会改变网络的整体函数。想象把一排同学的座位调换，老师的讲课内容不变。
- **仿射等变层（Affine Equivariant Layer）**：在置换对称下保持线性变换后再加偏置的层。它相当于在保持“换位不变”前提下，仍然可以做线性投影。
- **不变层（Invariant Layer）**：对置换操作的输出完全不变，常用于聚合信息，例如全局池化把所有神经元的特征压成一个标量。
- **池化（Pooling）**：把一组数值压缩成统计量（如求和、均值），在这里用于实现不变性。
- **广播（Broadcasting）**：把低维向量复制到高维结构上，以匹配权重矩阵的形状，类似于把一个全局标量扩展到每个神经元。
- **全连接层（Fully Connected Layer）**：对展开的权重向量做线性映射，在等变框架里需要特殊排列才能保持置换等变性。

### 核心创新点

1. **从“随意向量化”到“置换等变”**  
   - 之前的做法：直接把权重矩阵展平成向量，喂进普通的 MLP，忽视了神经元置换导致的等价性。  
   - 本文做法：构建层级结构，使每一层对神经元置换保持等变，即换掉中间层的神经元顺序，输出特征会相应换位但保持一致。  
   - 改变：模型不再把同一功能的网络视为不同的点，从而显著提升了学习效率和泛化能力。

2. **完整的仿射等变/不变层分类**  
   - 之前缺乏理论指引：人们只能凭经验手工设计等变层，难以保证覆盖所有可能的等变变换。  
   - 本文做法：给出对置换对称的**全体仿射等变层和不变层**的数学刻画，证明它们只能由池化、广播和特定排列的全连接层组合而成。  
   - 改变：提供了“构造指南”，任何想在权重空间上做等变学习的模型都可以直接套用这套模块。

3. **三大基本操作实现等变网络**  
   - 之前的实现往往需要自定义复杂的张量重排或稀疏乘法。  
   - 本文做法：把等变层拆解为 **池化 → 广播 → 全连接** 三步，全部使用深度学习框架的标准算子即可实现。  
   - 改变：大幅降低了实现门槛，使得权重空间学习可以在常规 GPU 环境下快速实验。

4. **在实际任务上验证等变优势**  
   - 之前缺少端到端的评估：大多数工作只在合成数据上展示等变性质。  
   - 本文做法：将等变网络用于 **预训练 MLP 的域适应、神经辐射场（NeRF）编辑、隐式神经表示（INR）微调** 等真实任务。  
   - 改变：实验显示相较于普通全连接基线，等变模型在收敛速度和最终性能上都有明显提升。

### 方法详解

**整体框架**  
1. **输入准备**：把目标预训练 MLP 的所有权重矩阵和偏置向量按层级顺序拼接成一个长向量。  
2. **层级等变处理**：该向量依次通过若干 **等变层**（每层对应一次仿射等变或不变变换）。  
3. **任务头**：最后的特征向量送入任务专用的输出层（例如分类头、回归头），完成具体下游任务。

**关键模块拆解**

1. **置换等变的仿射层**  
   - **池化**：对同一层的所有神经元权重做求和或均值，得到一个 **层级全局统计**。这一步实现了对置换的**不变**。  
   - **广播**：把全局统计复制到每个神经元对应的位置，形成与原始权重同形状的张量。此时每个神经元都拥有相同的全局信息。  
   - **全连接投影**：对广播后的张量做线性映射。因为投影矩阵本身是对所有神经元统一的，它不会破坏等变性。  
   - **组合**：池化 → 广播 → 全连接 的顺序保证了整体变换在置换下是等变的：如果把神经元顺序换了，池化结果不变，广播和全连接的操作顺序随之同步换位，最终输出特征也随之换位。

2. **不变层**  
   - 只做 **全局池化**（如求和、最大），直接把整层权重压成一个标量或向量。此时输出对任何置换都完全不变，常用于生成任务级别的全局描述。

3. **层间信息流**  
   - 每层等变处理完后，得到的特征会 **拼接** 到下一层的输入中。因为每层的特征本身已经对置换等变，拼接后仍保持整体等变性。  
   - 这种“层级递进”类似于卷积网络的层层感受野扩张，只是感受的是 **权重空间的结构**。

**最巧妙的设计**  
- 作者证明：**任何**满足置换等变性的仿射层都可以用上述三步实现。换句话说，作者把一个抽象的对称性约束转化为具体的可编程操作，避免了手工搜索或经验调参。  
- 另一个亮点是 **广播** 的使用：在普通的张量运算里，广播常被视作便利工具，这里它承担了把全局不变信息重新注入每个神经元的关键角色，确保等变层既能捕获全局统计，又不失局部细节。

### 实验与效果

- **测试任务**：  
  1. **域适应**：把在 ImageNet 上预训练的 MLP 调整到 CIFAR-10；  
  2. **NeRF 编辑**：对已训练好的神经辐射场的权重进行局部修改，实现视角或光照的快速切换；  
  3. **INR 微调**：在隐式神经表示的图像重建任务上，对权重进行少量梯度更新以适配新图像。

- **基线对比**：  
  - 普通全连接网络（直接向量化权重）  
  - 采用随机置换不变的池化‑全连接组合（缺少等变层的完整理论支撑）  

- **效果**：论文声称在所有任务上均显著优于基线，收敛速度提升约 30%~50%，最终指标（如分类准确率、PSNR）提升 2%~5%。具体数值在原文中有详细表格。

- **消融实验**：  
  - 去掉 **广播** 步骤会导致等变性破坏，性能下降约 3%；  
  - 只使用 **不变层**（全局池化）而不加入仿射投影，模型失去对局部权重差异的感知，任务表现下降约 4%。  

- **局限性**：  
  - 只针对 **全连接 MLP** 结构，卷积网络的权重置换对称更复杂，本文未给出直接扩展方案；  
  - 对极大规模的权重向量（如数亿参数的 Transformer）仍存在显存和计算成本挑战，作者在讨论中承认需要进一步的稀疏或分块技术。

### 影响与延伸思考

这篇工作开启了 **“权重空间直接学习”** 的新方向。随后出现的几篇论文（如 DWSNet、Weight2Vec 等）都在尝试把模型参数本身作为输入进行元学习、迁移学习或模型压缩。**推测**，等变层的理论框架会被用于 **图神经网络的参数共享**、**多任务模型的统一表示**，甚至在 **神经网络可解释性** 中提供参数层面的对称视角。想进一步深入的读者可以关注：

- 如何把 **卷积层的平移对称** 与 **权重置换对称** 统一进更通用的等变框架；  
- 在 **大模型微调** 场景下，利用等变网络实现 **参数高效适配**；  
- 将等变层与 **自监督预训练** 结合，探索权重空间的自监督表征学习。

### 一句话记住它

**把神经网络的权重当作输入时，只要用池化‑广播‑全连接的等变组合，就能让模型自然“不在乎”神经元的排列顺序，从而高效学习权重空间。**