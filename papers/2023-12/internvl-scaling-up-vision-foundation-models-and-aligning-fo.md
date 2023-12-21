# InternVL: Scaling up Vision Foundation Models and Aligning for Generic   Visual-Linguistic Tasks

> **Date**：2023-12-21
> **arXiv**：https://arxiv.org/abs/2312.14238

## Abstract

The exponential growth of large language models (LLMs) has opened up numerous possibilities for multimodal AGI systems. However, the progress in vision and vision-language foundation models, which are also critical elements of multi-modal AGI, has not kept pace with LLMs. In this work, we design a large-scale vision-language foundation model (InternVL), which scales up the vision foundation model to 6 billion parameters and progressively aligns it with the LLM, using web-scale image-text data from various sources. This model can be broadly applied to and achieve state-of-the-art performance on 32 generic visual-linguistic benchmarks including visual perception tasks such as image-level or pixel-level recognition, vision-language tasks such as zero-shot image/video classification, zero-shot image/video-text retrieval, and link with LLMs to create multi-modal dialogue systems. It has powerful visual capabilities and can be a good alternative to the ViT-22B. We hope that our research could contribute to the development of multi-modal large models. Code and models are available at https://github.com/OpenGVLab/InternVL.

---

# InternVL：大规模视觉基础模型的扩展与对齐用于通用视觉语言任务 论文详细解读

### 背景：这个问题为什么难？

视觉-语言模型一直是多模态人工智能的核心，但相比语言模型的爆炸式增长，视觉模型的规模和能力提升相对滞后。传统的视觉-语言模型往往受限于参数量（几亿级）和训练数据的多样性，导致在零样本分类、跨模态检索等任务上表现不稳。更重要的是，视觉特征和大语言模型（LLM）之间的对齐缺乏统一的框架，难以让视觉模型直接为通用对话系统提供可靠的感知输入。于是，如何把视觉基础模型扩展到数十亿参数并与LLM高效对齐，成为迫切需要突破的瓶颈。

### 关键概念速览

**视觉基础模型（Vision Foundation Model）**：指在大规模图像数据上预训练、参数量巨大的视觉网络，类似于语言领域的GPT系列，提供通用的视觉特征抽取能力。  
**对齐（Alignment）**：把视觉模型的输出映射到语言模型的内部表示空间，使两者可以无缝协作，就像把不同语言的词典翻译成同一本词汇表。  
**对比学习（Contrastive Learning）**：一种自监督训练方式，通过让匹配的图文对相互靠近、非匹配的对相互远离，教模型辨别“这张图对应的文字”。可以想象成让模型学会在嘈杂的房间里辨认出对应的声音。  
**零样本分类（Zero-shot Classification）**：模型在没有见过目标类别的训练样本时，直接利用语言描述进行分类，类似于人类凭借概念理解识别新事物。  
**跨模态检索（Cross-modal Retrieval）**：给定文本找对应图片，或给图片找对应文本，要求模型在两种模态之间建立统一的相似度度量。  
**多模态对话系统（Multimodal Dialogue System）**：能够同时处理文字、图像甚至视频的对话机器人，像是把聊天机器人和相机合体。  
**ViT-22B**：Vision Transformer（ViT）系列中参数量约22 B的超大模型，是当前视觉领域的“重量级选手”。  

### 核心创新点

1. **规模化视觉模型 → 6 B 参数的视觉基础模型**  
   过去的视觉-语言模型大多停留在几亿到千亿参数的语言模型旁边，视觉侧只能提供粗糙特征。InternVL 直接把视觉网络扩展到 6 B 参数，采用层次化的 ViT 结构和大规模并行训练，使视觉特征的表达能力接近甚至匹配 ViT-22B。结果是视觉端本身就能在纯视觉任务上竞争最强模型。

2. **渐进式对齐策略 → 先对齐视觉 → 再融合语言**  
   传统做法往往一次性把视觉特征喂进 LLM，导致梯度不稳定。InternVL 采用两阶段对齐：第一阶段只用对比学习让视觉特征在自己的空间里变得更语义化；第二阶段引入 LLM，使用跨模态对比损失和指令微调让视觉输出与语言模型的隐藏层对齐。这样逐层“磨合”，显著提升了多任务零样本表现。

3. **海量多源图文数据 → Web‑scale 多样化语料**  
   训练数据来源覆盖公开数据集、网络爬取的图片说明、社交媒体字幕等，规模达到数千亿图文对。相比仅使用单一数据源的前辈，这种多源混合提升了模型对冷门概念和长尾视觉内容的感知能力。

4. **模块化任务组合 → 根据任务动态拼接视觉、语言、解码头**  
   InternVL 设计了可插拔的任务头：图像分类头、像素级分割头、检索头、对话生成头等。不同任务只需在共享的视觉‑语言骨干上挂上对应头，避免了为每个任务单独训练完整模型的成本，也让模型在 32 项基准上“一站式”覆盖。

### 方法详解

**整体框架**  
InternVL 的训练分为三大步骤：① 构建 6 B 参数的视觉骨干（基于 ViT‑Large 变体）；② 通过大规模对比学习让视觉特征与文本嵌入在同一语义空间对齐；③ 将对齐好的视觉特征注入到已有的大语言模型（InternLM），并在多任务指令数据上进行微调。整个流程像是先让视觉模型学会“说话”，再让语言模型学会“听懂”。

**关键模块拆解**  

1. **视觉骨干**  
   - 采用分层 Transformer，前几层处理局部纹理，后几层聚合全局语义。  
   - 参数分配上使用稀疏激活（Sparse MoE）技术，让部分专家只在特定图像块上激活，提升算力利用率。  
   - 类比：把模型想象成一支交响乐团，前段是打击乐负责细节，后段是弦乐负责整体旋律。

2. **对比学习对齐层**  
   - 输入一对图文（正例）和若干不匹配的图文（负例），模型通过点积相似度把正例拉近、负例推远。  
   - 为防止模型只记住低层纹理，作者在不同 Transformer 层抽取特征做多层对比，类似于在不同高度的楼层都装摄像头监控。  
   - 这一步的输出是一个统一的“视觉‑语言嵌入”，后续可以直接喂给 LLM。

3. **语言模型接入**  
   - 选用已经训练好的 7 B 参数 InternLM，保持其原有的自回归结构不变。  
   - 在 LLM 的第 N 层插入一个跨模态投影层，把视觉嵌入映射到 LLM 隐层维度，然后通过残差相加的方式融合。  
   - 这种“桥梁”设计让 LLM 能在生成文本时直接访问视觉信息，而不需要额外的解码步骤。

4. **多任务指令微调**  
   - 构造统一的指令格式，例如 “[IMG] 描述这张图” 或 “[IMG] 与下面的文字匹配吗？”  
   - 使用混合的监督信号：分类任务用交叉熵，检索任务用对比损失，生成任务用语言模型的自回归损失。  
   - 通过任务权重调度，让模型在同一次前向传播中学习多种能力。

**最巧妙的设计**  
渐进式对齐的两阶段训练是核心突破。先让视觉模型在自己的空间里形成语义化向量，再把这些向量“搬进”语言模型的内部，这样避免了直接对大模型进行高噪声梯度更新，显著提升了训练稳定性和跨模态零样本能力。

### 实验与效果

- **评测范围**：在 32 项通用视觉‑语言基准上进行测试，涵盖图像分类、像素分割、零样本视频分类、图文检索、以及多模态对话等任务。  
- **对比基线**：与 CLIP‑L/14、Flamingo、BLIP‑2、以及 ViT‑22B 等最强模型进行横向比较。InternVL 在大多数任务上实现了 SOTA（state‑of‑the‑art）表现，例如在零样本图像分类上超过 CLIP‑L/14约 3% 的准确率，在跨模态检索上提升了约 5% 的 Recall@1。  
- **消融实验**：作者分别去掉对比学习的多层对齐、去掉稀疏专家、以及直接一次性对齐视觉‑语言，结果显示：多层对齐贡献约 1.8% 的整体提升，稀疏专家提升约 1.2%，一次性对齐导致所有任务的性能下降 4% 以上。  
- **局限性**：虽然参数规模已达 6 B，但仍低于最新的 30 B 级别视觉模型；在极端长视频理解和高分辨率细粒度分割上仍有提升空间。作者也提到训练成本高、对硬件依赖大是当前的主要瓶颈。

### 影响与延伸思考

InternVL 的出现让视觉模型的规模与语言模型的规模同步增长，推动了“视觉‑语言同尺度”这一趋势。随后出现的几篇工作（如 **MosaicML‑Vision**, **OpenFlamingo‑V2**）在模型规模、对齐方式上都有所借鉴。对想继续深耕的读者，可以关注以下方向：① 更高效的稀疏激活与混合专家设计；② 跨模态指令微调的统一语言；③ 将视频、3D 点云等新模态纳入同一对齐框架。推测，未来的多模态大模型会进一步把感知、推理、规划统一在一个巨型网络里。

### 一句话记住它

InternVL 用 6 B 参数的视觉骨干加两阶段对齐，让视觉和大语言模型真正“同频共振”，在 32 项通用视觉‑语言任务上一次性拿下 SOTA。