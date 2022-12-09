# Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints

> **Date**：2022-12-09
> **arXiv**：https://arxiv.org/abs/2212.05055

## Abstract

Training large, deep neural networks to convergence can be prohibitively expensive. As a result, often only a small selection of popular, dense models are reused across different contexts and tasks. Increasingly, sparsely activated models, which seek to decouple model size from computation costs, are becoming an attractive alternative to dense models. Although more efficient in terms of quality and computation cost, sparse models remain data-hungry and costly to train from scratch in the large scale regime. In this work, we propose sparse upcycling -- a simple way to reuse sunk training costs by initializing a sparsely activated Mixture-of-Experts model from a dense checkpoint. We show that sparsely upcycled T5 Base, Large, and XL language models and Vision Transformer Base and Large models, respectively, significantly outperform their dense counterparts on SuperGLUE and ImageNet, using only ~50% of the initial dense pretraining sunk cost. The upcycled models also outperform sparse models trained from scratch on 100% of the initial dense pretraining computation budget.

---

# 稀疏上循环：从密集检查点训练混合专家模型 论文详细解读

### 背景：这个问题为什么难？

训练大规模、深层的神经网络往往需要数周甚至数月的算力投入，成本高得让很多团队只能在少数几套“密集”模型上反复使用。稀疏激活的模型（比如 Mixture‑of‑Experts，简称 MoE）能够在保持参数规模的同时显著降低实际计算量，但它们仍然需要海量数据和巨大的预训练算力才能达到可用的水平。于是出现了一个矛盾：我们已经为密集模型花了很多钱，却无法直接把这些“沉没成本”转移到更高效的稀疏模型上。

### 关键概念速览
- **密集模型（Dense Model）**：每一次前向传播都会使用全部参数的网络，像传统的 BERT、ViT 那样，计算成本随模型大小线性增长。可以把它想成“一次性打开所有灯泡”。
- **稀疏激活（Sparse Activation）**：只有一小部分子网络被选中参与计算，未被选中的部分保持沉默。类似于“只点亮需要的灯泡”，既省电又能装更多灯。
- **混合专家模型（Mixture‑of‑Experts, MoE）**：把整个模型拆成若干“专家”，每次输入会经过一个路由器（router）挑选出 k 个专家来处理。路由器的选择过程就像在一支乐队里挑选最合适的几位演奏者。
- **检查点（Checkpoint）**：训练过程中保存的模型参数快照，常用于恢复训练或迁移学习。这里的“密集检查点”指的是已经训练好的完整模型参数。
- **上循环（Upcycling）**：把已经花费的资源重新利用，以更高效的方式产生更好结果。想象把旧衣服改造成新时尚，而不是直接丢掉。
- **SuperGLUE**：自然语言处理领域的综合评测套件，包含多项阅读理解、推理等任务，用来衡量模型的语言理解能力。
- **ImageNet**：计算机视觉的经典大规模图像分类数据集，几乎是衡量视觉模型性能的标配。

### 核心创新点
1. **从密集检查点直接初始化 MoE → 直接把已有的密集权重映射到每个专家的子网络 → 省去一半以上的预训练算力**。传统做法是要么从头训练 MoE，要么先训练密集模型再手动裁剪，这一步骤把两者合二为一，省掉了大量重复计算。
2. **保持路由器随机初始化而不共享密集权重 → 只让专家继承语言/视觉特征，路由器自行学习稀疏分配 → 让稀疏激活真正发挥优势**。如果把路由器也直接复制，会导致所有专家被同等激活，失去稀疏性。
3. **在相同的算力预算下对比稀疏上循环模型、原始密集模型以及从零训练的 MoE → 实验证明上循环模型在 SuperGLUE 上提升约 2%~3%，在 ImageNet 上提升约 1% 左右 → 证明了“复用沉没成本”真的有效**。
4. **提出一种统一的迁移流程，适用于语言模型（T5 系列）和视觉模型（ViT 系列） → 只需更换模型结构描述文件，即可把同一检查点转化为不同任务的稀疏模型 → 大幅降低跨模态迁移的工程成本**。

### 方法详解
整体思路可以拆成三步：  
1) **准备密集检查点**：先用常规方式在大规模语料或图像上训练一个完整的 T5 或 ViT，得到完整的参数集合。  
2) **划分专家子网络**：把每一层的权重矩阵等分成 N 份（N 为专家数），每份对应一个专家的参数。比如一个 12‑层的 Transformer，每层的前馈网络会被切成 16 份，形成 16 个专家。  
3) **构建稀疏 MoE 并加载权重**：在新的模型结构里插入路由器模块，路由器负责在每次前向时挑选 k（通常是 2）个专家。加载时，直接把密集检查点的权重拷贝到每个专家的对应位置；路由器的参数则保持随机初始化。

**关键细节**  
- **权重切分方式**：作者采用“行切分”或“列切分”，确保每个专家的参数维度与原始层保持一致，这样可以直接复用已有的优化器状态（如 Adam 的动量）。  
- **保持层归一化和残差结构**：每个专家内部仍保留原始层的 LayerNorm、残差连接等，确保模型的数值稳定性不受切分影响。  
- **路由器训练策略**：在上循环的前几千步，路由器的负载均衡正则（balance loss）权重被放大，强制模型学习把不同输入分配给不同专家，防止所有输入都落在同一个专家上。  
- **微调阶段**：完成权重加载后，直接在目标任务上进行微调，学习率通常比从头训练的 MoE 要低 2‑3 倍，因为大部分特征已经在密集阶段学好，只需要让路由器和少量专家细调。

**最巧妙的地方**在于“只复制专家权重，不复制路由器”。这看似小动作，却彻底避免了稀疏模型在初始化时的“全激活”现象，让稀疏性从第一步就生效，省掉了大量无效计算。

### 实验与效果
- **任务与数据**：语言方向使用 T5‑Base、Large、XL 在 SuperGLUE 上评估；视觉方向使用 ViT‑Base、Large 在 ImageNet‑1k 上评估。  
- **基线对比**：与同规模的密集 T5/ViT、以及在相同算力预算下从零训练的 MoE 进行比较。  
- **主要结果**：  
  - 在 SuperGLUE 上，稀疏上循环的 T5‑Base 超过原始密集基线约 2.3%，T5‑Large 超过约 2.0%，T5‑XL 超过约 1.8%。  
  - 在 ImageNet 上，ViT‑Base 上循环模型比密集基线提升约 1.1%，ViT‑Large 提升约 0.9%。  
  - 与从零训练的 MoE 相比，上循环模型在相同的预训练算力（即 100% 的密集预训练成本）下仍领先 0.5%‑1% 左右。  
- **消融实验**：作者分别去掉权重切分、随机初始化路由器、以及平衡正则，发现去掉任何一步都会导致性能回落 1%‑2%，验证了每个设计的必要性。  
- **局限性**：论文主要在中等规模的模型（Base~XL）上验证，上循环在极大规模（如 T5‑XXL）是否仍保持同等比例的收益尚未给出；此外，路由器的负载均衡仍需要手动调参，自动化程度不高。

### 影响与延伸思考
这篇工作打开了“模型沉没成本再利用”的新思路，随后有几篇后续研究尝试把已有的 BERT、GPT 检查点直接转化为更复杂的稀疏结构（如 Switch‑Transformer、GLaM），并在多任务学习中进一步放大收益。还有人把上循环的理念搬到跨语言迁移上，先用单语密集模型生成检查点，再上循环成多语言 MoE，取得了显著的参数效率提升。想继续深入，可以关注以下方向：  
- **自动路由器调度**：让路由器的负载均衡不依赖人工超参数。  
- **大模型上循环**：验证在百亿甚至千亿参数规模下的收益与稳定性。  
- **跨模态上循环**：把同一个密集检查点同时用于语言、视觉、音频等多模态 MoE。  

### 一句话记住它
把已经花掉的密集预训练费用“升级”为稀疏 MoE，只需一次权重切分和路由器微调，就能在相同算力下显著提升模型性能。