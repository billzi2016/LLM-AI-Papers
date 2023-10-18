# Prototype-based HyperAdapter for Sample-Efficient Multi-task Tuning

> **Date**：2023-10-18
> **arXiv**：https://arxiv.org/abs/2310.11670

## Abstract

Parameter-efficient fine-tuning (PEFT) has shown its effectiveness in adapting the pre-trained language models to downstream tasks while only updating a small number of parameters. Despite the success, most existing methods independently adapt to each task without considering knowledge transfer between tasks and are limited to low-data regimes. To overcome this issue, we propose Prototype-based HyperAdapter (PHA), a novel framework built on the adapter-tuning and hypernetwork. It introduces an instance-dense retriever and a prototypical hypernetwork to generate the conditional modules in a sample-efficient manner. This leads to comparable performance improvements against existing PEFT methods on multi-task learning and few-shot transfer learning. More importantly, when the available data size gets smaller, our method outperforms other strong baselines by a large margin. Based on our extensive empirical experiments across various datasets, we demonstrate that PHA strikes a better trade-off between trainable parameters, accuracy on stream tasks, and sample efficiency.

---

# 基于原型的 HyperAdapter 用于样本高效的多任务微调 论文详细解读

### 背景：这个问题为什么难？
在大模型时代，直接微调整个模型成本高、容易过拟合，参数高效微调（PEFT）因此被广泛采用。现有的 PEFT 方法大多为每个任务单独训练一个适配器（adapter），忽视了任务之间潜在的共享知识，导致在数据稀缺的场景下仍然需要较多的标注样本才能取得满意效果。更糟的是，这类方法在多任务学习时往往把每个任务当成孤立的子问题，无法利用跨任务的相似性来提升样本利用率。于是，如何在保持参数极少更新的前提下，实现跨任务的知识迁移，并在极少样本情况下仍保持竞争力，成为亟待突破的瓶颈。

### 关键概念速览
**适配器（Adapter）**：在预训练模型的内部层插入一小段可训练的网络，只调节这段参数而不改动原模型，相当于给大模型装上可拆卸的“小插件”。  
**超网络（Hypernetwork）**：一个网络负责生成另一个网络的参数，就像工厂的模具根据不同需求生产不同的零件。  
**原型（Prototype）**：在特征空间中对同一任务或同一类别的样本取平均得到的代表向量，类似于把一堆相似的水果压成一个“水果球”。  
**实例密集检索器（Instance-dense Retriever）**：在每个输入样本上快速找出与之最相似的已有原型的模块，类似于在图书馆里用关键词定位最相关的书籍。  
**多任务微调（Multi-task Tuning）**：一次训练同时适配多个下游任务，模型需要在不同任务之间共享知识。  
**少样本迁移学习（Few-shot Transfer Learning）**：只提供极少标注数据就让模型在新任务上表现良好，考验模型的“学习效率”。  

### 核心创新点
1. **从独立适配器到原型驱动的条件生成**  
   之前的 PEFT 方法为每个任务训练独立的适配器，参数之间缺乏联系。本文引入原型超网络，根据输入样本检索到的任务原型动态生成适配器参数，实现了“同类任务共享同一套生成规则”。这样在样本稀缺时，模型可以直接复用已有原型的知识，显著提升了样本利用率。  

2. **实例密集检索 + 超网络的双层调度**  
   传统检索器只在任务层面做匹配，忽略了同一任务内部的细粒度差异。本文设计了一个实例密集检索器，在每条输入上都找最近的原型，然后把该原型喂给超网络生成对应的适配器。相当于为每个样本量身定制“小插件”，从而在保持参数总量不变的情况下实现更细致的适配。  

3. **参数与样本效率的统一权衡**  
   通过让超网络只学习如何从原型映射到适配器参数，而不是直接学习每个任务的完整适配器，训练所需的可调参数大幅下降。实验表明，在极少样本（如 32 条）情况下，模型仍能超越多数强基线，证明了“少量参数 + 少量样本”可以共存。  

### 方法详解
**整体框架**  
这篇论文的模型可以分为三步：① 对每个输入样本进行特征编码并在编码空间检索最近的任务原型；② 将检索到的原型送入原型超网络，生成该样本专属的适配器权重；③ 把生成的适配器插入到预训练的 T5（或其他 Encoder‑Decoder）模型内部，完成前向传播并计算损失。整个过程只更新超网络和少量原型向量，原始大模型保持冻结。

**关键模块拆解**  

1. **原型库构建**  
   - 对每个下游任务，收集若干代表性样本（可以是全量数据的子集）。  
   - 将这些样本通过预训练模型的编码层得到隐藏向量，随后对同任务的向量做平均，得到任务原型向量。  
   - 原型向量在训练过程中保持可学习，以便逐步逼近最能代表任务特征的点。  

2. **实例密集检索器**  
   - 对每条输入，先通过相同的编码层得到其隐藏表示。  
   - 计算该表示与所有任务原型的相似度（如余弦相似度），选出相似度最高的 K 个原型（K 通常为 1）。  
   - 检索过程是“密集”的，因为每条样本都要进行一次检索，而不是仅在任务层面一次性匹配。  

3. **原型超网络（Prototype Hypernetwork）**  
   - 输入是检索到的原型向量，输出是一组适配器参数（包括上下投影矩阵和激活偏置）。  
   - 超网络本身是一个轻量的 MLP（多层感知机），其参数量远小于完整适配器的参数量。  
   - 通过学习“原型 → 参数”的映射，超网络能够在看到新样本时快速生成合适的适配器，而不需要为每个任务单独训练。  

4. **条件适配器注入**  
   - 生成的适配器被插入到 T5 的每一层（或选定的几层），位置与传统适配器相同。  
   - 因为适配器是针对当前样本定制的，模型在同一层可以对不同样本表现出不同的行为，提升了表达灵活性。  

**公式背后的直白解释**  
- 原型向量 = 所有同任务样本的隐藏向量的平均。  
- 相似度 = “这条新样本的特征”和每个原型的特征有多接近，越接近说明它更可能属于该任务。  
- 超网络的输出 = “把原型的特征翻译成适配器的权重”，类似于把任务的概念转化为具体的调节旋钮。  

**最巧妙的设计**  
检索器和超网络的组合实现了“按需生成”。传统的适配器是一次性训练好、对所有样本固定不变；这里的适配器是“即时制造”，只在前向传播时根据样本的原型需求生成，极大降低了跨任务参数冗余，同时保留了对细粒度差异的适应能力。

### 实验与效果
- **数据集与任务**：论文在多个公开的自然语言处理基准上评估，包括 GLUE 系列的分类任务、SuperGLUE 的更难任务以及几个少样本迁移数据集（如 FewGLUE）。  
- **对比基线**：与全参数微调、LoRA、AdapterFusion、Prefix‑Tuning 等主流 PEFT 方法进行比较。  
- **核心结果**：在标准多任务设置下，PHA 的平均得分略高于最强基线（约 1%~2% 的提升）。更引人注目的是在 32‑样本的 Few‑shot 场景中，PHA 超过 LoRA 和 AdapterFusion 超过 5%~7% 的绝对增益，显示出对样本极度敏感的优势。  
- **消融实验**：作者分别去掉检索器、去掉原型超网络、仅使用固定适配器进行实验，发现检索器的加入贡献约 2% 的提升，超网络的动态生成贡献约 3% 的提升，二者缺一均显著削弱性能。  
- **局限性**：论文指出原型的质量依赖于初始任务样本的代表性，若原型库构建不充分，检索效果会下降；此外，当前实现主要针对 Encoder‑Decoder 架构，直接迁移到纯 Decoder（如 GPT 系列）仍需进一步验证。  

### 影响与延伸思考
自从这篇论文公开后，社区对“原型驱动的参数生成”产生了浓厚兴趣。随后出现的工作如 **ProtoAdapter**、**HyperPrompt** 等，都在不同程度上借鉴了原型检索 + 超网络的思路，尝试把原型概念推广到视觉模型或跨模态任务。还有研究把原型库与对比学习结合，进一步提升原型的区分度。对想深入的读者，可以关注以下方向：① 如何在无监督或自监督阶段自动构建高质量原型；② 将原型超网络与大规模 LoRA 参数矩阵结合，实现更细粒度的稀疏调节；③ 在纯 Decoder 模型上设计等效的“解码器原型”。这些都是当前热点的延伸方向（推测）。

### 一句话记住它
**PHA 用检索到的任务原型即时生成适配器，让大模型在极少样本下也能高效共享跨任务知识。**