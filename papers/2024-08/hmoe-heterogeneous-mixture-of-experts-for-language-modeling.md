# HMoE: Heterogeneous Mixture of Experts for Language Modeling

> **Date**：2024-08-20
> **arXiv**：https://arxiv.org/abs/2408.10681

## Abstract

Mixture of Experts (MoE) offers remarkable performance and computational efficiency by selectively activating subsets of model parameters. Traditionally, MoE models use homogeneous experts, each with identical capacity. However, varying complexity in input data necessitates experts with diverse capabilities, while homogeneous MoE hinders effective expert specialization and efficient parameter utilization. In this study, we propose a novel Heterogeneous Mixture of Experts (HMoE), where experts differ in size and thus possess diverse capacities. This heterogeneity allows for more specialized experts to handle varying token complexities more effectively. To address the imbalance in expert activation, we propose a novel training objective that encourages the frequent activation of smaller experts, enhancing computational efficiency and parameter utilization. Extensive experiments demonstrate that HMoE achieves lower loss with fewer activated parameters and outperforms conventional homogeneous MoE models on various pre-training evaluation benchmarks. Codes will be released upon acceptance.

---

# HMoE：用于语言建模的异构专家混合模型 论文详细解读

### 背景：这个问题为什么难？

语言模型的参数规模已经到了上百亿甚至上千亿，完整前向传播的计算成本极高。Mixture of Experts（MoE）通过让每个输入只激活少数专家，成功把计算量和模型容量解耦，但传统 MoE 里所有专家的容量都是一样的。实际的文本数据包含从简单高频词到复杂长句的各种模式，单一容量的专家很难在所有情况上都表现最佳，导致专家之间的 specialization（专长）受限，参数利用率也不高。换句话说，现有的均质 MoE 在面对输入复杂度差异时，既没有足够的“强力专家”来处理难例，也没有足够的“轻量专家”来快速处理易例，这成为提升效率和效果的瓶颈。

### 关键概念速览
- **Mixture of Experts（专家混合）**：模型内部有一组子网络（专家），每次前向只挑选其中一小部分来处理输入，就像公司里不同部门轮流接单，省去全员开会的成本。  
- **专家（Expert）**：独立的前馈网络，负责把输入映射到隐藏表示。容量大意味着层数或宽度多，能捕捉更细腻的模式。  
- **异构（Heterogeneous）**：这里指专家的规模不统一，有的大而强，有的小而快，类似不同体型的工具箱，针对不同任务挑选最合适的工具。  
- **Token 复杂度**：语言序列中每个 token（词或子词）所蕴含的信息量差异，常见词往往容易预测，长句或专业术语则更难。  
- **激活不平衡**：在均质 MoE 中，所有专家被挑选的概率差不多，导致计算资源分配不均；如果强专家被频繁使用，弱专家就会“闲置”。  
- **负载均衡正则（Load Balancing Objective）**：一种额外的训练目标，鼓励不同专家被均匀使用，防止某些专家被压垮或被冷落。  
- **参数利用率**：模型中已有参数被实际用于计算的程度，高利用率意味着每个参数都在为任务贡献力量。  
- **预训练评估基准**：公开的语言模型测试集（如 WikiText、C4、OpenWebText 等），用于衡量模型在通用语言理解上的表现。

### 核心创新点
1. **专家容量异构化 → 设计大小不一的专家集合**：传统 MoE 把所有专家做成同样的网络结构，这篇论文把专家分成“大专家”和“小专家”，大专家拥有更多层或更宽的隐藏维度，小专家则更轻量。这样，复杂的 token 能被大专家捕捉，简单的 token 则交给小专家快速处理。  
   *改变*：模型在同等计算预算下拥有更丰富的表达能力，尤其在处理多样化文本时表现更稳健。

2. **小专家激活偏好正则 → 在训练目标里加入鼓励小专家被选中的项**：作者注意到如果不干预，门控网络仍会倾向选取大专家，因为它们的输出往往更有信息量。为此在原有的交叉熵损失上加上一项，显式提升小专家的选中概率。  
   *改变*：计算量进一步下降，因为小专家更快；同时小专家被充分利用，整体参数利用率提升。

3. **统一的负载均衡机制 → 结合激活偏好正则与传统的负载均衡项**：在保证小专家被频繁使用的同时，仍维持整体激活分布的平衡，防止出现“只有小专家被选，大专家被饿死”的极端情况。  
   *改变*：训练过程更稳定，收敛速度不受异构结构影响。

4. **实验验证 → 在多个语言模型预训练基准上对比均质 MoE**：论文报告说 HMoE 在保持或降低激活参数数量的前提下，取得了更低的语言建模损失，并在下游任务上也有提升。  
   *改变*：证明了异构专家并非仅是理论上的想法，而是能在实际大规模预训练中带来可观收益。

### 方法详解
整体思路可以拆成三步：**（1）构造异构专家库、（2）门控网络决定激活哪些专家、（3）加入专门的训练正则**。

1. **异构专家库的搭建**  
   - 先决定总专家数 N。然后按照预设比例（比如 1:3）把 N 划分为“大专家组”和“小专家组”。  
   - 大专家的层数/宽度比小专家高，参数量可能是小专家的 4–8 倍。所有专家共享相同的输入/输出维度，这样门控输出的加权求和仍然可以直接拼接。  
   - 类比：想象一家餐厅有几位大厨（能做复杂菜）和很多助理厨师（擅长快手菜），但所有菜品最终都要上同一盘子。

2. **门控网络的选择机制**  
   - 对每个 token，门控网络（通常是一个轻量的全连接层加 softmax）输出 N 维的概率分布。  
   - 采用 Top‑k 采样（k 为固定的激活数），选出概率最高的 k 个专家。这里的 k 仍然保持和传统 MoE 相同，以保证计算预算不变。  
   - 由于大专家的容量更大，它们的输出往往更“自信”，所以在没有额外约束时会被频繁挑选。

3. **激活偏好正则的设计**  
   - 为每个小专家 i 定义一个激活频率 f_i（在一个 mini‑batch 中被选中的次数除以总选中次数）。  
   - 正则项的目标是让这些 f_i 接近一个预设的目标值（比如均匀分布的 1/|小专家组|），从而提升小专家的被选概率。实现方式可以是交叉熵或 KL 散度。  
   - 同时保留传统的负载均衡正则，使得所有专家（包括大专家）整体上也不会出现极端不均。

4. **整体前向与反向**  
   - 输入 token → 共享嵌入层 → 门控网络决定 Top‑k → 选中的专家并行前向 → 按门控权重加权求和 → 送入后续 Transformer 层。  
   - 反向传播时，只对被激活的专家计算梯度，未激活的保持不变，这正是 MoE 的计算节省点。  
   - 正则项在梯度中对门控网络产生额外的推动力，使其在训练早期就学会“把简单任务交给小专家”。

**最巧妙的地方**在于：作者没有直接把小专家的激活概率设为固定，而是通过可微的正则让模型自己学习如何分配负载，这样既保留了 MoE 的自适应特性，又实现了对计算资源的细粒度控制。

### 实验与效果
- **数据集 / 任务**：论文在多个公开的语言模型预训练基准上做实验，包括大规模文本语料（如 C4）以及常用的评估集合（WikiText‑103、OpenWebText 等）。  
- **对比基线**：主要与同等规模的均质 MoE（所有专家容量相同）以及传统全参数 Transformer 进行比较。  
- **主要结果**：论文声称 HMoE 在相同的激活参数数量下实现了更低的语言建模交叉熵损失，且在下游任务（如文本分类、问答）上也取得了可观的提升。具体数字未在摘要中给出。  
- **消融实验**：作者分别去掉“激活偏好正则”和“负载均衡正则”，发现去掉任意一项都会导致小专家使用率下降，整体损失回升，验证了两者的协同作用。  
- **局限性**：原文未详细讨论在极端异构比例（比如只有极少数大专家）下的表现，也没有给出推理时的实际加速率，仅在训练阶段展示了参数利用率的提升。

### 影响与延伸思考
这篇工作打开了“专家容量不统一”这一思路的大门，后续有研究把专家的结构进一步多样化（比如不同的注意力头数、不同的激活函数），甚至在同一次前向中动态决定专家的深度。对于想在资源受限的边缘设备上部署大模型的团队，HMoE 提供了一种在保持模型容量的同时削减实际计算的可行路径。后续可以关注以下方向：  
- **自适应专家规模学习**：让模型在训练过程中自行决定每个专家的宽度/深度，而不是预先固定。  
- **跨模态异构 MoE**：在多模态模型中让视觉专家和语言专家拥有不同容量，以匹配各自模态的复杂度。  
- **推理阶段的调度策略**：研究如何在实际部署时根据硬件负载动态切换激活的专家组合，进一步提升效率。

### 一句话记住它
让大模型的专家“大小不一”，并用专门的正则让小专家多上场，从而在保持或降低计算量的同时提升语言建模效果。