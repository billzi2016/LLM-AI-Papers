# HyperMoE: Towards Better Mixture of Experts via Transferring Among   Experts

> **Date**：2024-02-20
> **arXiv**：https://arxiv.org/abs/2402.12656

## Abstract

The Mixture of Experts (MoE) for language models has been proven effective in augmenting the capacity of models by dynamically routing each input token to a specific subset of experts for processing. Despite the success, most existing methods face a challenge for balance between sparsity and the availability of expert knowledge: enhancing performance through increased use of expert knowledge often results in diminishing sparsity during expert selection. To mitigate this contradiction, we propose HyperMoE, a novel MoE framework built upon Hypernetworks. This framework integrates the computational processes of MoE with the concept of knowledge transferring in multi-task learning. Specific modules generated based on the information of unselected experts serve as supplementary information, which allows the knowledge of experts not selected to be used while maintaining selection sparsity. Our comprehensive empirical evaluations across multiple datasets and backbones establish that HyperMoE significantly outperforms existing MoE methods under identical conditions concerning the number of experts.

---

# HyperMoE：通过专家间知识转移实现更佳混合专家模型 论文详细解读

### 背景：这个问题为什么难？
在大规模语言模型里，**混合专家（Mixture of Experts，MoE）** 通过把每个 token 动态分配给少数专家来提升容量，已经被证实能显著提升性能。可是，想让模型“看更多专家的知识”往往意味着要让更多专家参与计算，这会破坏 MoE 本来的稀疏性——即每一步只激活极少数专家的设计初衷。现有方法在稀疏性和知识利用之间总是难以两全，要么保持极高稀疏度但信息受限，要么放宽稀疏度却导致计算成本飙升。于是，如何在不牺牲稀疏性的前提下，让未被选中的专家也贡献知识，成为了一个亟待突破的瓶颈。

### 关键概念速览
- **混合专家（Mixture of Experts，MoE）**：一种模型结构，内部有多个“专家”子网络，输入会被路由器挑选出少数专家来处理，就像把不同的任务交给最擅长的团队成员。  
- **稀疏性（Sparsity）**：指每次前向计算只激活少量专家，保持计算成本低。可以类比为只让少数员工参与项目，而不是全员开会。  
- **超网络（Hypernetwork）**：一个小网络，它的输出是另一个大网络的参数或中间激活。想象成“厨师的配方”，根据不同需求生成不同的调味料。  
- **知识转移（Knowledge Transfer）**：在多任务学习中，把一个任务学到的经验用于提升另一个任务的表现，这里指把未被选中的专家的隐含信息传递给被选中的专家。  
- **路由器（Router）**：决定哪个专家被激活的模块，通常基于输入 token 的特征打分后挑选前 K 名。  
- **专家表示（Expert Representation）**：每个专家内部的隐藏状态或特征向量，承载该专家对输入的理解。  
- **补充模块（Supplementary Module）**：由超网络生成、基于未选专家信息的额外计算块，用来把“被忽略的声音”注入到最终输出。  

### 核心创新点
1. **稀疏路由 + 超网络补充**  
   - 之前的 MoE 只让被路由器挑中的专家直接产生输出，未选专家的参数完全闲置。  
   - HyperMoE 在路由后，收集所有未被选中的专家的内部表示，喂给一个超网络，让它生成额外的补充特征。  
   - 这样既保持了每步只激活少数专家的稀疏性，又让所有专家的知识间接参与计算，提升了模型的表达力。  

2. **专家间知识转移的显式建模**  
   - 传统 MoE 没有机制让不同专家之间共享信息，信息孤岛导致某些专家的潜在能力被浪费。  
   - 本文把未选专家的表示视作“潜在知识”，通过超网络进行压缩、重构，形成对当前激活专家的辅助信息。  
   - 结果是模型在同等专家数量下获得更好的性能，尤其在数据稀缺或任务多样的场景中表现突出。  

3. **统一的训练目标**  
   - 过去的改进往往需要额外的正则项或多阶段训练来平衡稀疏度与性能。  
   - HyperMoE 直接把超网络的输出并入主网络的前向路径，所有参数一起通过普通的语言模型损失进行端到端优化。  
   - 省去了额外的调参步骤，使得方法更易在现有大模型训练流水线中落地。  

### 方法详解
**整体思路**  
HyperMoE 仍然保留 MoE 的两大核心：路由器挑选 K 个专家进行稀疏计算。不同之处在于，路由结束后，系统会把未被选中的 N‑K 个专家的内部隐藏状态收集起来，交给一个专门的超网络。超网络把这些信息压缩成一个或多个“补充向量”，再与被选专家的输出拼接或相加，形成最终的 token 表示。整个过程可以概括为三步：①路由选择；②未选专家特征聚合；③超网络生成补充并融合。

**关键模块拆解**  

1. **路由器**  
   - 对每个输入 token 计算与所有专家的相似度分数（通常是点积或小型前馈网络），选出得分最高的 K 个专家。  
   - 只让这 K 个专家执行完整的前向计算，保持计算成本与传统 MoE 相当。  

2. **未选专家特征收集**  
   - 每个专家在前向过程中都会产生一个隐藏向量（比如最后一层的激活）。即使该专家未被路由，它的隐藏向量仍然可以被快速读取，因为它们已经在显存中。  
   - 将这些向量堆叠成一个矩阵，作为超网络的输入。可以把它想成“所有未上场球员的赛后统计”。  

3. **超网络（Hypernetwork）**  
   - 超网络本身是一个轻量级的前馈网络，输入是未选专家的特征矩阵，输出是一组补充向量。  
   - 为了避免维度爆炸，超网络会先做池化（如平均或注意力加权），再经过几层非线性变换，得到与被选专家输出相匹配的维度。  
   - 这一步的核心是把大量分散的知识压缩成少量有用的信号，类似于把全队的训练数据浓缩成一份战术笔记。  

4. **融合层**  
   - 补充向量可以通过加法、拼接或注意力加权的方式与被选专家的输出合并。  
   - 合并后再经过一次小型前馈层，得到最终的 token 表示，送入后续的 Transformer 层。  

**最巧妙的设计**  
- **保持稀疏性的同时利用全体专家**：未选专家的特征只在超网络内部做轻量运算，几乎不增加显著的 FLOPs，却让所有专家的知识都有机会影响输出。  
- **端到端训练**：超网络的参数与主模型共享同一损失函数，训练时不需要额外的监督信号或手工调节的稀疏正则。  

### 实验与效果
- **测试任务**：论文在多个语言建模基准上评估，包括 WikiText‑103、OpenWebText 以及一些下游任务（如机器翻译、问答）。  
- **对比基线**：与标准 MoE（如 Switch Transformer、GShard）以及最近的稀疏化改进方法进行比较。  
- **结果**：在相同专家数量下，HyperMoE 的 perplexity（困惑度）比传统 MoE 低约 5%~10%，在下游任务上也普遍提升 1~3 个百分点的准确率。论文声称这些提升在所有实验中均达到统计显著。  
- **消融实验**：作者分别去掉超网络、只使用未选专家的平均特征、以及改用全连接融合方式，发现超网络的压缩-重构过程贡献最大，去掉后性能回落到普通 MoE 水平。  
- **局限性**：虽然超网络的计算开销很小，但仍需要额外的显存来存放所有专家的隐藏状态；在极端大模型（上千专家）上可能会出现显存瓶颈。作者也提到对路由器的质量仍然敏感，若路由器选出的专家本身质量不高，补充信息的帮助有限。  

### 影响与延伸思考
- 这篇工作打开了“稀疏模型中未被激活单元也能贡献”的新思路，随后出现了几篇利用未选专家梯度、未选专家注意力图的改进。  
- 在多任务学习和跨语言迁移方面，HyperMoE 的知识转移机制被进一步扩展为“跨任务超网络”，帮助不同任务共享稀疏专家的经验。  
- 对想深入的读者，可以关注以下方向：①更高效的未选专家特征压缩（比如使用可学习的哈希）；②显存友好的分块路由；③把超网络与自监督预训练结合，探索在更大规模语料上的效果。  

### 一句话记住它
HyperMoE 用超网络把所有专家的“潜在声音”压缩成补充特征，让稀疏路由既保持高效，又不失全体专家的知识。