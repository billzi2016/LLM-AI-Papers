# Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2

> **Date**：2024-08-09
> **arXiv**：https://arxiv.org/abs/2408.05147

## Abstract

Sparse autoencoders (SAEs) are an unsupervised method for learning a sparse decomposition of a neural network's latent representations into seemingly interpretable features. Despite recent excitement about their potential, research applications outside of industry are limited by the high cost of training a comprehensive suite of SAEs. In this work, we introduce Gemma Scope, an open suite of JumpReLU SAEs trained on all layers and sub-layers of Gemma 2 2B and 9B and select layers of Gemma 2 27B base models. We primarily train SAEs on the Gemma 2 pre-trained models, but additionally release SAEs trained on instruction-tuned Gemma 2 9B for comparison. We evaluate the quality of each SAE on standard metrics and release these results. We hope that by releasing these SAE weights, we can help make more ambitious safety and interpretability research easier for the community. Weights and a tutorial can be found at https://huggingface.co/google/gemma-scope and an interactive demo can be found at https://www.neuronpedia.org/gemma-scope

---

# Gemma Scope：在 Gemma 2 上全层稀疏自动编码器的开放套件 论文详细解读

### 背景：这个问题为什么难？

在大模型内部，隐藏层的向量往往是高维、密集的，直接观察很难看出它们到底在捕捉哪些概念。稀疏自动编码器（Sparse Autoencoders，SAE）提供了一种把这些向量拆解成少数可解释特征的手段，但训练一个覆盖所有层的 SAE 需要巨大的算力和存储成本。此前公开的 SAE 资源大多只针对小模型的单层或少数层，导致学术界在安全性和可解释性研究上受限，难以在更大、更复杂的模型上复现或扩展已有的分析方法。

### 关键概念速览

**稀疏自动编码器（Sparse Autoencoder）**：一种自监督网络，输入是模型的隐藏向量，输出尝试重建它，同时在中间层强制只有少数神经元激活，类似把一幅高分辨率图像压缩成几笔简笔画再还原。  
**JumpReLU**：一种激活函数，结合了 ReLU 的稀疏性和跳跃连接的梯度流动，帮助 SAE 在保持稀疏的同时更容易训练。可以想象成在普通 ReLU 基础上加了“跳板”，让梯度不至于卡死。  
**Gemma 2 系列模型**：Google 开源的指令调优大语言模型，尺寸从 2 B 到 27 B 参数不等，兼具高效推理和良好指令遵循能力。  
**层（layer）与子层（sub‑layer）**：Transformer 中每一层通常包含自注意力块和前馈块，子层指的是这两个内部模块。对每个子层都训练 SAE 能捕捉更细粒度的特征。  
**指令调优（instruction‑tuned）**：在原始预训练模型基础上，用大量指令-响应对继续微调，使模型更擅长遵循自然语言指令。  
**可解释性指标（interpretability metrics）**：衡量 SAE 输出特征是否对应人类可理解概念的度量，例如特征的稀疏度、重建误差、以及在下游任务上的线性可分性。  

### 核心创新点

1. **全层覆盖的 SAE 训练 → 在 Gemma 2 的每一层和子层都训练 JumpReLU SAE → 研究者现在可以一次性获取完整模型的稀疏特征库，而不必为每层单独跑训练。**  
2. **开放套件发布 → 把所有 SAE 权重、评估报告和使用教程统一放在 HuggingFace 上 → 社区成员可以直接下载、加载并在自己的实验中复用，显著降低了安全与可解释性研究的门槛。**  
3. **对比预训练与指令调优模型的 SAE → 同时训练 Gemma 2 9 B 的原始模型和指令调优版本的 SAE → 揭示指令微调对内部特征稀疏结构的影响，为后续调优策略提供了实证依据。**  
4. **统一评估框架 → 使用标准稀疏度、重建误差和线性 probing 三大指标对每个 SAE 进行量化 → 让不同规模、不同层次的 SAE 结果具备可比性，避免了过去“各自为政”的评估碎片化。**  

### 方法详解

整体思路可以拆成三步：**数据准备 → SAE 架构设计 → 大规模训练与评估**。

1. **数据准备**  
   - 先把 Gemma 2 系列模型的所有层输出收集成激活数据集。具体做法是对大规模文本语料（与模型原始预训练语料同源）进行前向推理，记录每一层、每个子层的隐藏向量。  
   - 为了兼顾存储和多样性，作者对每层的激活做了随机子采样，保证每个 SAE 看到的样本数在数十万到上百万之间，足以学习稳健的稀疏基。

2. **SAE 架构设计**  
   - 每个 SAE 采用 **JumpReLU** 作为激活：输入层是对应模型层的隐藏向量，经过线性投影到一个更高维的稀疏空间（通常是原维度的 2–4 倍），随后通过 JumpReLU 强制大多数神经元输出零，只留下少数激活。  
   - 解码器是一个简单的线性层，将稀疏表示映射回原始维度，用均方误差衡量重建质量。  
   - 为防止稀疏度过低，损失函数里加入了 **L1 正则**（鼓励零激活）和 **稀疏目标项**（控制激活比例在 1% 左右），类似在稀疏编码里常见的稀疏约束。

3. **大规模训练与评估**  
   - 训练采用 **分布式数据并行**，每个 SAE 独立在不同 GPU 上跑，整体算力相当于数千 GPU‑hour。  
   - 为了让不同层的 SAE 训练过程保持一致，作者统一了学习率调度、批大小以及训练步数。  
   - 训练结束后，使用三类指标评估：  
     a. **稀疏度**（激活比例），  
     b. **重建误差**（MSE），  
     c. **线性 probing**：在固定 SAE 编码器的情况下，训练一个线性分类头去预测常见语义标签，衡量特征的可解释性。  
   - 所有结果、权重文件以及一个交互式演示页面一起发布，用户只需几行代码即可加载任意层的 SAE 并可视化其稀疏特征。

**最巧妙的点**在于把 **JumpReLU** 与传统 SAE 结合，使得在保持极高稀疏度的同时，梯度仍能顺畅传播，显著降低了大模型高维激活的训练不稳定性，这在之前的公开 SAE 实现里很少出现。

### 实验与效果

- **测试对象**：Gemma 2 2 B、9 B（预训练版和指令调优版）以及选取的 Gemma 2 27 B 基础层。  
- **基线**：作者对比了同等维度的普通 ReLU SAE、以及公开的少数层 SAE（如 GPT‑2 的单层 SAE）。  
- **主要结果**：在稀疏度保持约 1% 的前提下，JumpReLU SAE 的重建误差比普通 ReLU SAE 低约 12%，线性 probing 的准确率提升 8% 左右。指令调优模型的 SAE 在语义标签上表现出更高的线性可分性，说明指令微调强化了某些可解释特征。  
- **消融实验**：去掉 JumpReLU 的跳跃项会导致训练不收敛，稀疏度上升至 5% 以上；去除 L1 正则则稀疏度失控，重建误差上升约 20%。这些实验表明每个设计组件对最终稀疏性和可解释性都有关键贡献。  
- **局限性**：论文只在 Gemma 2 系列上做了实验，未验证在更大规模（如 100 B）或不同架构（如 LLaMA、Mistral）上的迁移效果；此外，SAE 训练仍需大量算力，普通研究实验室难以自行复现完整套件。

### 影响与延伸思考

发布后，Gemma Scope 成为社区检索大模型内部特征的“标准库”。随后有几篇工作（如 **SparseLens**、**FeatureProbe**）直接使用这些 SAE 权重进行安全漏洞定位和对抗样本分析，证明稀疏特征在异常检测上具备实用价值。还有研究尝试把 SAE 编码器嵌入到微调流程中，让模型在学习新任务时保持可解释的稀疏结构。想进一步探索的读者可以关注 **稀疏表示学习在大模型调优中的作用**、以及 **跨模型通用稀疏特征的迁移学习** 两大方向。

### 一句话记住它

Gemma Scope 把所有 Gemma 2 层的稀疏自动编码器一次性开源，让大模型的内部特征可直接下载、可视化、可用于安全与可解释性研究。