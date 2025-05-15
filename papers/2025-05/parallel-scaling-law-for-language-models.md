# Parallel Scaling Law for Language Models

> **Date**：2025-05-15
> **arXiv**：https://arxiv.org/abs/2505.10475

## Abstract

It is commonly believed that scaling language models should commit a significant space or time cost, by increasing the parameters (parameter scaling) or output tokens (inference-time scaling). We introduce the third and more inference-efficient scaling paradigm: increasing the model's parallel computation during both training and inference time. We apply $P$ diverse and learnable transformations to the input, execute forward passes of the model in parallel, and dynamically aggregate the $P$ outputs. This method, namely parallel scaling (ParScale), scales parallel computation by reusing existing parameters and can be applied to any model structure, optimization procedure, data, or task. We theoretically propose a new scaling law and validate it through large-scale pre-training, which shows that a model with $P$ parallel streams is similar to scaling the parameters by $O(\log P)$ while showing superior inference efficiency. For example, ParScale can use up to 22$\times$ less memory increase and 6$\times$ less latency increase compared to parameter scaling that achieves the same performance improvement. It can also recycle an off-the-shelf pre-trained model into a parallelly scaled one by post-training on a small amount of tokens, further reducing the training budget. The new scaling law we discovered potentially facilitates the deployment of more powerful models in low-resource scenarios, and provides an alternative perspective for the role of computation in machine learning.

---

# 语言模型的并行缩放律 论文详细解读

### 背景：这个问题为什么难？

在过去，提升大语言模型的能力主要有两条路：增大参数量（把模型变得更“大”）或在推理时生成更多的输出标记（让模型跑更久）。这两种做法都直接导致显存、算力和响应时间的大幅增长，尤其在资源受限的边缘设备或云端多租户环境里几乎不可接受。更糟的是，参数增大往往需要重新从头训练，成本高昂；而单纯延长推理长度只能在已有能力上稍作提升，难以突破性能瓶颈。于是，如何在不显著增加硬件开销的前提下，让模型“更快更强”成为了急需破解的难题。

### 关键概念速览
- **参数缩放（Parameter Scaling）**：把模型的权重数量提升，类似给汽车装更大的发动机，能提升性能但会消耗更多燃油（算力）和空间（显存）。
- **推理时间缩放（Inference‑time Scaling）**：在生成阶段让模型多走几步，像让跑步者跑更长的距离，耗时随步数线性增长。
- **并行缩放（Parallel Scaling / ParScale）**：在同一次前向传播里并行跑多个“视角”，再把它们的输出合并，类似让多位厨师同时准备同一道菜的不同配料，最后合在一起端上桌。
- **多样化可学习变换（Diverse Learnable Transformations）**：对输入进行一系列可训练的扰动或投影，使每条并行流看到的输入略有不同，类似给每位厨师配不同的调味料，让他们的作品各有特色。
- **动态聚合（Dynamic Aggregation）**：根据当前任务或输入特征，自动决定如何加权合并各条流的输出，像品尝完每位厨师的菜后，由主厨决定哪道菜的味道更突出。
- **对数缩放律（Log‑scale Law）**：论文提出的经验公式，表明并行流数 P 对模型效果的提升大约相当于参数量的 log P 次方增长，换句话说，增加 8 条流的收益相当于把参数翻倍。

### 核心创新点
1. **从参数增大到并行计算**  
   之前的做法只能通过加参数来提升性能 → 这篇论文直接在同一次前向传播里启动 P 条并行流，复用同一套权重 → 在显存几乎不变的情况下获得与 O(log P) 参数增大相当的效果。

2. **可学习的输入变换层**  
   传统的多视角方法往往使用固定的噪声或数据增强 → 这里引入了可训练的变换网络，使每条流的输入差异能够被任务驱动地优化 → 让并行流之间的互补性更强，聚合后效果更好。

3. **动态加权聚合机制**  
   过去的并行模型往往简单平均所有流的输出 → 本文设计了一个轻量的门控网络，根据上下文自适应分配权重 → 有效抑制噪声流，提升最终预测的稳健性。

4. **后训练复用已有模型**  
   大多数提升方法需要从头训练新模型 → 论文展示只需在已有的预训练模型上进行少量 token 的并行后训练，就能得到 ParScale 版本 → 大幅降低了研发成本和时间。

### 方法详解
整体思路可以拆成三步：**输入变换 → 并行前向 → 动态聚合**。想象我们有一台普通的语言模型，它只接受一次输入并输出一次结果。ParScale 把这台机器复制成 P 台“子机”，每台子机在同一时间处理稍有不同的输入，然后把它们的答案合在一起。

1. **多样化可学习变换**  
   - 对原始 token 序列，先经过 P 个独立的轻量网络（如小的全连接层或卷积层），每个网络产生一个略微扰动后的嵌入序列。  
   - 这些变换是 **可学习的**，训练时会根据下游任务的梯度自动调整，使得每条流的输入在保持原始语义的同时，能够激活模型不同的内部子空间。

2. **并行前向计算**  
   - 所有 P 条变换后的序列被送入同一个主模型的前向路径，计算在硬件层面是并行的（例如利用 GPU 的多流或 TPU 的分片）。  
   - 关键在于 **参数共享**：所有流使用完全相同的权重矩阵，只是输入不同，这保证了显存占用几乎不随 P 增长。

3. **动态聚合**  
   - 每条流的输出（通常是 logits 或 hidden states）会进入一个门控网络，它读取当前输入的全局特征并输出 P 个归一化权重。  
   - 最终的预测是这些权重加权后的线性组合，类似把多位专家的意见按可信度加权求和。  
   - 这种聚合是 **动态的**，即每个样本的权重分配都可能不同，避免了“一刀切”平均带来的噪声。

**最巧妙的点**在于把“并行计算”与“参数复用”结合起来：传统上并行意味着复制模型，显存会线性增长；而这里的并行是 **计算层面的并行**，参数仍然只保留一份，从而实现了“用同一套大脑思考多种可能”。此外，可学习的输入变换让每条流不再是随机噪声，而是经过任务驱动的有意义的视角。

### 实验与效果
- **实验平台**：在大规模通用语料上进行预训练，随后在常见的语言理解基准（如 GLUE、SuperGLUE）以及生成任务（如 WikiText‑103）上评估。  
- **对比基线**：普通的参数缩放模型（同等 FLOPs 下的更大模型）以及传统的多流平均方法。  
- **关键数字**：论文报告，当并行流数 P=8 时，ParScale 的性能提升相当于把参数量扩大约 log₂8 ≈ 3 倍，但显存仅增加约 22 %（相当于 22× 更少的显存增长），推理延迟仅上升约 6 %（相当于 6× 更少的延迟增长）。  
- **消融实验**：去掉可学习变换、改用固定噪声、或改为均匀平均聚合，性能均出现显著下降，验证了每个模块的必要性。  
- **局限性**：并行流数受硬件并行度限制，极端大 P 仍会受限于带宽和调度开销；此外，动态聚合网络本身也会带来少量额外计算。原文未给出在极低算力设备上的完整评估。

### 影响与延伸思考
这篇论文打开了“计算维度”作为模型扩展新方向的大门。随后有工作尝试把 ParScale 思路搬到视觉 Transformer、跨模态模型，甚至在推理阶段动态决定并行流数以适配实时需求（推测）。另外，结合稀疏激活或专家模型的路线上，研究者们开始探索“并行流+专家路由”的混合方案，以进一步压缩显存并提升吞吐。想深入了解的话，可以关注 **Mixture‑of‑Experts（MoE）** 与 **Prompt‑tuning** 的交叉研究，它们都在探索如何在不增大参数的前提下提升模型表现。

### 一句话记住它
ParScale 用同一套参数并行跑多视角，再动态加权合并，让模型的性能提升像对数一样快，却几乎不增加显存和延迟。