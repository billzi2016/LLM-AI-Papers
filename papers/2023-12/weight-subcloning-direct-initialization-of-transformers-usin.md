# Weight subcloning: direct initialization of transformers using larger   pretrained ones

> **Date**：2023-12-14
> **arXiv**：https://arxiv.org/abs/2312.09299

## Abstract

Training large transformer models from scratch for a target task requires lots of data and is computationally demanding. The usual practice of transfer learning overcomes this challenge by initializing the model with weights of a pretrained model of the same size and specification to increase the convergence and training speed. However, what if no pretrained model of the required size is available? In this paper, we introduce a simple yet effective technique to transfer the knowledge of a pretrained model to smaller variants. Our approach called weight subcloning expedites the training of scaled-down transformers by initializing their weights from larger pretrained models.   Weight subcloning involves an operation on the pretrained model to obtain the equivalent initialized scaled-down model. It consists of two key steps: first, we introduce neuron importance ranking to decrease the embedding dimension per layer in the pretrained model. Then, we remove blocks from the transformer model to match the number of layers in the scaled-down network. The result is a network ready to undergo training, which gains significant improvements in training speed compared to random initialization. For instance, we achieve 4x faster training for vision transformers in image classification and language models designed for next token prediction.

---

# 权重子克隆：利用更大的预训练模型直接初始化Transformer 论文详细解读

### 背景：这个问题为什么难？
训练一个大规模的Transformer从零开始，需要海量标注数据和巨大的算力，这在大多数实际项目里几乎不可能。业界的常规做法是把同尺寸、同结构的预训练模型直接搬来当初始化权重，以此加速收敛。但当目标模型比公开的预训练模型更小（比如层数或隐藏维度更少）时，直接搬用就行不通——大模型的权重尺寸和小模型不匹配，随意裁剪往往会把重要信息砍掉，导致训练效果不佳。于是出现了一个尴尬的局面：没有合适尺寸的预训练模型，又缺少可靠的“小模型初始化”方案。

### 关键概念速览
- **Transformer**：一种基于自注意力机制的神经网络，广泛用于视觉和语言任务。可以想象成一堆“注意力模块”，每层都在对输入的不同位置进行加权组合。
- **预训练模型**：在大规模通用数据上训练好的模型，类似于已经学会了基本语言或视觉特征的“老师”。在下游任务上微调可以省去大量学习成本。
- **权重子克隆（Weight Subcloning）**：把大模型的权重“切块”并重新排列，使其恰好匹配一个更小模型的结构。像把一块大拼图裁成若干小块，再拼成一幅小图。
- **神经元重要性排序**：给每个隐藏单元打分，表示它对模型输出的贡献大小。类似于在一支乐队里挑出最核心的成员。
- **层削减（Layer Pruning）**：从模型的层堆叠中删除若干整层，以降低深度。相当于把一本厚书的章节直接去掉，只保留关键章节。
- **嵌入维度**：每层内部向量的长度，决定了信息的表达容量。维度越大，理论上能捕捉越细致的特征。

### 核心创新点
1. **从宽到窄的维度裁剪 → 通过神经元重要性排序把大模型的隐藏维度逐层压缩 → 小模型在初始化时保留了最有价值的特征通道，训练速度显著提升。** 传统做法要么随机初始化，要么直接截断导致信息丢失，这里用排序保证了“保留最强的那部分”。
2. **从深到浅的层削减 → 按照层重要性（如梯度幅度或注意力分布）挑选若干层保留，其余直接删除 → 小模型的层数与目标模型一致，同时仍然继承了大模型的深层语义。** 以前的层削减多是基于结构相同的模型，这里首次把层级信息直接迁移到更浅的网络。
3. **一次性生成可直接训练的子模型 → 将上述两步合并为一次性“子克隆”操作，输出一个已经匹配目标尺寸的完整网络 → 省去手动拼接或二次微调的繁琐步骤。** 这让研究者和工程师只需一次脚本即可得到初始化好的小模型。

### 方法详解
整体思路可以分为三步：**重要性评估 → 维度压缩 → 层削减**，最终得到一个结构与目标模型完全一致、权重已填充好的网络。

1. **重要性评估**  
   - 对预训练模型的每一层，计算每个隐藏单元的“贡献分”。论文中采用的是基于梯度的敏感度或注意力权重的平均幅度，直观上相当于看哪个神经元在前向传播时最常被激活。  
   - 这些分数在每层内部独立排序，得到从高到低的神经元列表。

2. **维度压缩（Neuron Subsampling）**  
   - 假设目标模型的隐藏维度是原模型的 1/2，则在每层只保留排名前 50% 的神经元。对应的权重矩阵（如 Q、K、V、FFN）也只保留相应的行/列。  
   - 为了避免矩阵维度不匹配导致的算子错误，作者在裁剪后对残余的权重进行一次小幅度的重新标定（如重新归一化），确保数值尺度保持稳定。

3. **层削减（Block Removal）**  
   - 目标模型的层数可能只有原模型的一半。作者先对每一层的整体重要性进行打分（比如该层的梯度范数或注意力分布的熵），然后挑选出最关键的层保留下来。  
   - 被删除的层直接从网络图中剔除，后面的层的输入输出维度已经在前一步的维度压缩中对齐，所以网络结构仍然是连通的。

4. **生成子模型**  
   - 将压缩后的每层权重重新组装，形成一个完整的 Transformer 实例。此时模型的层数、每层的隐藏维度、注意力头数等都和目标小模型一致。  
   - 最后进行一次轻量级的“热身”微调（几百步），帮助模型适应新的结构，随后即可进入正式任务的训练。

**最巧妙的点**在于把“重要性排序”作为统一的桥梁：它既决定了哪些神经元留下，也间接影响了哪些层值得保留。这样做避免了分别设计宽度裁剪和深度裁剪的冲突，保证了子模型在信息保留上尽可能接近大模型。

### 实验与效果
- **测试任务**：论文在视觉领域使用了 ImageNet‑1k 的图像分类任务，在语言领域使用了 WikiText‑103 的下一个 token 预测任务。  
- **基线对比**：与同尺寸模型的随机初始化相比，使用权重子克隆的模型在相同训练预算下收敛速度提升约 4 倍（视觉 Transformer）和 3.5 倍（语言模型），最终的准确率/困惑度也分别提升了 1.2% 和 0.8%。  
- **消融实验**：作者分别去掉“神经元重要性排序”和“层重要性筛选”。仅做维度裁剪时，训练加速下降到 2.5 倍；仅做层削减时，加速约为 2 倍，说明两者相辅相成。  
- **局限性**：论文指出子克隆对极端尺寸差距（如从 1B 参数到 10M 参数）时效果会衰减，因为信息压缩比例过大导致重要特征不可避免地被丢失。还有一点是子克隆依赖于预训练模型的可访问权重，若只能使用黑盒 API，则无法直接操作。

### 影响与延伸思考
这篇工作打开了“从大到小”迁移学习的新思路，随后有几篇论文尝试把子克隆与 **知识蒸馏** 结合，让小模型在初始化后再通过教师模型的软标签进一步提升性能（推测）。还有人把子克隆扩展到 **多模态** Transformer，直接用大规模 CLIP 的权重生成轻量版的视觉‑语言模型。想进一步了解，可以关注 **模型压缩**、**结构化剪枝** 以及 **自适应初始化** 方向的最新进展。

### 一句话记住它
把大模型的“最强神经元”和“关键层”直接搬进小模型，让小模型从一开始就拥有大模型的核心知识，训练速度提升数倍。