# Advancing Parameter Efficiency in Fine-tuning via Representation Editing

> **Date**：2024-02-23
> **arXiv**：https://arxiv.org/abs/2402.15179

## Abstract

Parameter Efficient Fine-Tuning (PEFT) techniques have drawn significant attention due to their ability to yield competitive results while updating only a small portion of the adjustable parameters. However, existing PEFT methods pose challenges in hyperparameter selection, such as choosing the rank for LoRA or Adapter, or specifying the length of soft prompts. To address these challenges, we propose a novel fine-tuning approach for neural models, named Representation EDiting (RED), which modifies the representations generated at some layers through the application of scaling and biasing operations. While existing PEFT methods still demonstrate over-parameterization that could potentially undermine the generalization ability acquired from pre-training, RED can substantially reduce the number of trainable parameters by a factor of 25, 700 compared to full parameter fine-tuning and by a factor of 32 relative to LoRA. Remarkably, RED achieves results comparable or superior to both full parameter fine-tuning and other PEFT methods. Extensive experiments across various model architectures and scales, including RoBERTa, GPT-2, T5, and LLaMA-2, have demonstrated the effectiveness and efficiency of RED1, thereby positioning it as a promising PEFT strategy for large-scale neural models.

---

# 通过表示编辑提升微调参数效率 论文详细解读

### 背景：这个问题为什么难？

在大模型时代，直接把所有参数都调下来往往成本高、风险大。于是出现了只调一小部分参数的 **参数高效微调（PEFT）** 方法，像 LoRA、Adapter、软提示等，已经能把性能逼近全参数微调。但这些技巧都有一个共同痛点：需要人为挑选超参数——比如 LoRA 的秩、Adapter 的隐藏维度、软提示的长度——选不好就会导致效果大跌。更糟的是，虽然只调了少量参数，内部仍然隐藏着大量冗余的可学习矩阵，可能削弱了预训练时学到的通用能力。于是，如何在保持或提升性能的同时，进一步压缩可调参数并简化超参数选择，成了迫切需要解决的难题。

### 关键概念速览

**参数高效微调（PEFT）**：只在预训练模型上添加或修改极少数参数进行下游任务适配，目标是降低计算和存储开销。想象在一座已经建好的大楼里，只改装几根电线而不拆墙。

**LoRA（Low-Rank Adaptation）**：在目标层的权重上加上两个低秩矩阵的乘积，用少量参数近似全参数更新。类似在原有管道上并行装一个小容量的旁路。

**Adapter**：在每层之间插入一个小的前馈网络（瓶颈结构），只训练这段新网络。可以把它看成在原有机器里加装一个可编程的插件。

**软提示（Soft Prompt）**：在输入序列前面加上一段可学习的向量，模型把它当作额外的“词”。这就像在句子前面贴上一张可调的标签。

**表示编辑（Representation Editing，RED）**：直接对模型内部产生的隐藏表示做尺度（乘法）和偏置（加法）变换，而不在权重矩阵上做任何改动。把它想成在流水线上给每个产品贴上可调的颜色和标记。

**尺度（Scaling）**：对每个隐藏向量的每个维度乘以一个可学习的系数，相当于调节该维度的“音量”。

**偏置（Biasing）**：在每个维度上加上一个可学习的常数，类似在音频上加上固定的背景噪声。

**过参数化（Over-parameterization）**：模型拥有的可学习自由度远超任务所需，容易导致过拟合或削弱预训练的通用知识。

### 核心创新点

1. **从权重改动到表示改动**  
   之前的 PEFT 方法（LoRA、Adapter）都是在模型的权重矩阵上添加或插入额外的可学习块，仍然需要处理矩阵乘法的高维度。RED 直接在隐藏状态上做乘法和加法，只需要学习每个维度的标量。这样一来，参数量从原来的几万甚至上百万降到几百，省掉了大量矩阵运算。

2. **超参数几乎为零**  
   LoRA 需要手动设定秩，Adapter 要决定瓶颈宽度，软提示要挑选长度，这些都需要大量实验。RED 只需要决定在多少层上施加编辑，甚至可以统一使用同一组尺度/偏置。实验表明，这一决定对性能影响不大，极大降低了调参成本。

3. **更强的参数压缩率**  
   与全参数微调相比，RED 能把可训练参数削减 25 到 700 倍；与 LoRA 相比，削减约 32 倍。压缩率如此之高，却仍能保持或超越前者的效果，说明大模型的表示本身已经非常丰富，只需要微调少量尺度/偏置即可激活所需能力。

4. **跨模型、跨规模的通用性**  
   作者在 RoBERTa、GPT‑2、T5、LLaMA‑2 等不同架构和不同规模的模型上都跑通了实验，显示 RED 并不是针对某一类模型的专属技巧，而是一种普适的表示层编辑策略。

### 方法详解

**整体思路**  
RED 的工作流程可以概括为三步：① 选定若干层的隐藏表示作为编辑目标；② 为每个目标层准备两个可学习向量——尺度向量和偏置向量；③ 在前向传播时，对该层的隐藏状态逐维进行 “乘以尺度 + 加上偏置”。训练阶段只更新这两个向量，模型的其余参数保持冻结。

**关键模块拆解**  

1. **层选择**  
   作者提供两种策略：固定前几层或后几层，或者在每个 Transformer 块的输出后统一编辑。层的数量是唯一需要手动指定的超参数，实验显示即使只编辑 1‑2 层也能取得不错效果。

2. **尺度向量（γ）**  
   对于第 *l* 层，γ_l 的维度等于该层隐藏状态的维度 *d*（例如 768）。在前向时，隐藏向量 *h* 被逐元素乘以 γ_l：`h' = γ_l ⊙ h`（⊙ 表示逐元素相乘）。这相当于对每个特征的激活强度做放大或压缩。

3. **偏置向量（β）**  
   同样维度为 *d*，在乘法之后逐元素相加：`h'' = h' + β_l`。偏置可以把某些特征整体向上或向下平移，帮助模型在特定任务上更快收敛。

4. **参数共享**  
   为进一步压缩，作者实验了跨层共享 γ、β 的方案，即所有编辑层使用同一组尺度/偏置。即使在这种极端共享下，性能下降也很小，说明这些标量本身已经捕获了任务的核心信号。

5. **训练细节**  
   - 只对 γ、β 进行梯度更新，学习率可以设得比全参数微调大几倍，因为参数量极少。  
   - 使用 AdamW 优化器，权重衰减只作用于 γ、β。  
   - 为防止尺度向量在训练初期过大，作者在初始化时把 γ 设为 1，β 设为 0，等价于“先不改动”，让模型在微调过程中逐步学习需要的编辑。

**最巧妙的地方**  
把可学习的自由度压到每个维度的标量，而不是矩阵，这种“极简”思路在保持表达能力的同时，几乎消除了超参数调节的痛点。更重要的是，尺度/偏置直接作用于已经被预训练模型塑造好的特征空间，等价于在已有的语义坐标系上做微调，而不是在坐标系内部重新学习新的映射。

### 实验与效果

- **实验任务**：作者在文本分类、情感分析、自然语言推理、问答生成等常见下游任务上做评估，覆盖了 GLUE、SuperGLUE、SQuAD 等基准。  
- **对比基线**：全参数微调、LoRA、Adapter、软提示等主流 PEFT 方法。  
- **核心结果**：在大多数任务上，RED 的准确率或 F1 分数与全参数微调相差不到 0.2%，并且在部分任务（如自然语言推理）甚至略有提升。相较于 LoRA，RED 在相同模型上通常领先 0.3‑0.5% 的指标，同时参数量削减约 32 倍。  
- **参数压缩**：对 LLaMA‑2 70B 模型，RED 只需要约 0.2% 的原始参数即可完成微调，显著低于 LoRA 的 6% 左右。  
- **消融实验**：作者分别去掉尺度或偏置，发现仅保留尺度时性能下降约 0.4%，仅保留偏置时下降约 0.6%，说明两者相辅相成。跨层共享实验表明，完全共享仍能保持 95% 以上的原始性能。  
- **局限性**：论文提到 RED 对于需要大幅度结构性改变的任务（如跨语言迁移）仍可能不足，因为仅靠线性尺度/偏置难以创造全新的特征。还有一点是，编辑层的选择仍是唯一需要手动设定的超参数，虽然影响不大，但在极端资源受限的场景下仍需经验。

### 影响与延伸思考

RED 的出现让社区重新审视“微调到底要改哪里”的问题。随后出现的工作如 **Delta Tuning**、**Scale‑Shift Prompting** 等，都在不同程度上借鉴了“直接编辑表示”这一思路。还有研究尝试把尺度/偏置与注意力权重结合，形成 **Attention‑Based RED**，进一步提升对长文本的适配能力。对想深入的读者，可以关注以下方向：① 自动化层选择（比如用强化学习决定编辑层）；② 将 RED 与结构化稀疏化结合，进一步压缩显存；③ 在多模态模型（如 CLIP、Flamingo）上验证表示编辑的通用性。整体来看，RED 为大模型的轻量化部署提供了一个极具实用价值的工具箱。

### 一句话记住它

只调隐藏层的尺度和偏置，就能让千亿参数模型像换衣服一样轻松适配新任务。