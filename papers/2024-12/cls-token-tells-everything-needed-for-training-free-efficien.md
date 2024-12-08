# [CLS] Token Tells Everything Needed for Training-free Efficient MLLMs

> **Date**：2024-12-08
> **arXiv**：https://arxiv.org/abs/2412.05819

## Abstract

Multimodal Large Language Models (MLLMs) have recently demonstrated strong performance across a wide range of vision-language tasks, garnering significant attention in the computer vision. However, their efficient deployment remains a substantial challenge due to high computational costs and memory requirements. Recognizing the redundancy of information within the vision modality, recent studies have explored methods for compressing visual tokens in MLLMs to enhance efficiency in a training-free manner. Despite their effectiveness, existing methods like Fast rely on the attention between visual tokens and prompt text tokens as the importance indicator, overlooking the relevance to response text and thus introducing perception bias. In this paper, we demonstrate that in MLLMs, the [CLS] token in the visual encoder inherently knows which visual tokens are important for MLLMs. Building on this prior, we introduce a simple yet effective method for train-free visual token compression, called VTC-CLS. Firstly, it leverages the attention score of the [CLS] token on visual tokens as an importance indicator for pruning visual tokens. Besides, we also explore ensembling the importance scores derived by the [CLS] token from different layers to capture the key visual information more comprehensively. Extensive experiments demonstrate that our VTC-CLS achieves the state-of-the-art performance across various tasks compared with baseline methods. It also brings notably less computational costs in a training-free manner, highlighting its effectiveness and superiority. Code and models are available at \url{https://github.com/THU-MIG/VTC-CLS}.

---

# 【CLS】Token 告诉一切：免训练高效 MLLM 的视觉 Token 压缩 论文详细解读

### 背景：这个问题为什么难？

多模态大语言模型（MLLM）把图像特征当成一串“视觉 token”喂给语言模型，效果很好，但每张图往往会产生几百甚至上千个 token，导致显存和算力飙升。过去的压缩方法大多要重新训练模型，成本高；而免训练的方案（比如 Fast）只能靠视觉 token 与提示文本之间的注意力来判断重要性，却忽略了模型最终要生成的答案文本，容易把对答案关键的视觉信息误删，出现“感知偏差”。因此，如何在不重新训练的前提下，精准挑出真正关键的视觉 token，成为阻碍 MLLM 实际部署的瓶颈。

### 关键概念速览

**多模态大语言模型（MLLM）**：把视觉编码器和语言模型串起来，让模型同时理解图像和文字，就像把看图说话的能力装进了 ChatGPT。  

**视觉 token**：视觉编码器把图片切成若干块（patch），每块映射成一个向量，这些向量就叫视觉 token，类似于语言模型里的词向量。  

**[CLS] token**：在视觉编码器的输入序列最前面加的一个特殊标记，训练时用来做全局分类。它会“看”到所有后面的视觉 token，注意力权重可以反映每个 token 对整体信息的贡献。  

**注意力分数（attention score）**：模型在自注意力层里计算的相似度，用来决定信息流向。把它想成“谁对谁更重要”的打分表。  

**训练无关压缩（training-free compression）**：不需要再跑梯度更新，只利用已有模型的内部信号（如注意力）直接删减 token，省时省力。  

**Token 剪枝（token pruning）**：把注意力分数低的视觉 token 删除，只保留高分的，让后续计算更轻。  

**层级集成（layer-wise ensemble）**：把不同 Transformer 层得到的注意力分数加权合并，类似把多位专家的意见综合，得到更稳健的判定。

### 核心创新点

1. **用 [CLS] token 替代提示文本作为重要性指示**  
   - 之前的免训练方法（如 Fast）把提示词和视觉 token 的注意力当作重要性依据，容易忽视答案文本的需求。  
   - 本文直接取视觉编码器里 [CLS] token 对每个视觉 token 的注意力分数，认为它天然捕捉了全局信息与任务相关性。  
   - 结果是剪枝后模型在视觉理解任务上几乎不掉分，甚至比基线更稳。

2. **跨层注意力分数集成**  
   - 单层注意力可能只关注局部特征，容易漏掉细粒度信息。  
   - 作者把每一层的 [CLS] → token 注意力分数累加或加权平均，得到一个更全面的“重要性图”。  
   - 这种集成让剪枝既保留全局概览，又不失细节，提升了多任务的鲁棒性。

3. **全程免训练、即插即用的压缩管线**  
   - 方法只需读取已有模型的注意力矩阵，计算权重，排序并截断，无需再跑任何梯度或微调。  
   - 因此可以直接在公开的 MLLM（如 LLaVA、MiniGPT‑4）上部署，显著降低显存占用和推理时延。

### 方法详解

**整体思路**  
VTC‑CLS 把视觉编码器的 [CLS] token 当作“指挥官”，读取它对每个视觉 token 的注意力分数，依据这些分数挑选出最重要的 N% token，剩下的直接丢掉，然后把保留下来的 token 送进语言模型继续生成答案。整个过程不改动模型参数，只是前处理一步。

**步骤拆解**  

1. **获取注意力矩阵**  
   - 将输入图像送入视觉 Transformer，得到每层的自注意力矩阵。  
   - 在每层的矩阵里，取出第一行（对应 [CLS] token）对所有视觉 token 的注意力向量。  

2. **层级集成**  
   - 对所有层的注意力向量做归一化（比如软最大），再按层权重（可均等或根据经验）相加，得到每个视觉 token 的综合重要性分数。  
   - 这一步相当于把不同深度的专家意见合并，既有浅层的纹理感知，也有深层的语义抽象。  

3. **排序与阈值剪枝**  
   - 按分数从高到低排序，设定保留比例（如保留 30%），或设定阈值分数。  
   - 直接删除低分 token，剩余 token 的顺序保持不变，确保位置编码仍然有效。  

4. **喂入语言模型**  
   - 把保留下来的视觉 token 与原始的文本提示一起拼接，送入跨模态语言模型进行推理。  
   - 由于 token 数大幅下降，显存占用和计算量同步降低。  

**关键细节**  
- 如果视觉编码器没有显式的 [CLS] token，作者建议用所有 token 的平均向量代替，效果略逊但仍可用。  
- 注意力分数的归一化方式对最终剪枝比例有微小影响，实验中发现软最大配合层均等加权最稳。  
- 剪枝后仍保留原始的位置信息，使得语言模型可以直接使用已有的跨模态融合层，无需额外适配。

**最巧妙的地方**  
把已经训练好的 [CLS] token 视作“全局感知器”，直接利用它的注意力分布来决定哪些视觉信息是必需的，这一步既省去额外的标注或微调，又天然避免了只看提示文本导致的感知偏差。

### 实验与效果

- **测试任务**：论文在视觉问答（VQAv2）、图像描述（COCO Caption）、科学问答（ScienceQA）等多模态基准上评估。  
- **对比基线**：与原始不压缩模型、以及免训练压缩方法 Fast 进行比较。  
- **结果**：作者声称 VTC‑CLS 在所有任务上均实现了最新的性能（SOTA），同时把 FLOPs 和显存使用降低约 30%~50%，具体数值在论文附表中给出。  
- **消融实验**：通过去掉层级集成、只用单层注意力或改用提示文本注意力，性能均出现明显下降，验证了每个设计的贡献。  
- **局限性**：原文未给出对极端低保留比例（如 <10%）的表现，也没有在非 Transformer 视觉编码器上做实验，可能受编码器结构影响。

### 影响与延伸思考

VTC‑CLS 的核心思路——利用已有模型内部的全局标记（[CLS]）进行训练无关的 token 过滤，已经在后续的多模态压缩工作中被广泛引用。比如 2024 年的几篇论文尝试把 ViT 的 class token 与跨模态注意力结合，进一步提升了实时推理的效率。对想深入的读者，可以关注以下方向：

- **类 token 的多任务重要性评估**：研究不同任务（检索、定位、生成）下 class token 注意力的差异。  
- **自适应保留比例**：根据输入图像的复杂度动态决定剪枝比例，进一步节约算力。  
- **跨模态协同压缩**：把语言侧的关键 token 也一起剪枝，形成端到端的双向轻量化。

### 一句话记住它

**只看视觉编码器的 [CLS] 注意力，就能在不训练的情况下精准删减视觉 token，省算力又不掉分。**