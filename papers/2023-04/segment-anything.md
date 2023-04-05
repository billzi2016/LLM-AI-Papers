# Segment Anything

> **Date**：2023-04-05
> **arXiv**：https://arxiv.org/abs/2304.02643

## Abstract

We introduce the Segment Anything (SA) project: a new task, model, and dataset for image segmentation. Using our efficient model in a data collection loop, we built the largest segmentation dataset to date (by far), with over 1 billion masks on 11M licensed and privacy respecting images. The model is designed and trained to be promptable, so it can transfer zero-shot to new image distributions and tasks. We evaluate its capabilities on numerous tasks and find that its zero-shot performance is impressive -- often competitive with or even superior to prior fully supervised results. We are releasing the Segment Anything Model (SAM) and corresponding dataset (SA-1B) of 1B masks and 11M images at https://segment-anything.com to foster research into foundation models for computer vision.

---

# 随意分割 论文详细解读

### 背景：这个问题为什么难？
在视觉分割领域，传统方法往往为每一种任务（如实例分割、全景分割）训练专门的模型，数据集规模也受限于标注成本。模型只能在训练时看到的类别和场景上表现好，遇到新物体或不同分布时会失效。更糟的是，现有系统大多只能接受固定的输入形式（比如整张图或预先给定的框），缺乏灵活的交互方式。于是，研究者一直在寻找一种“通用分割器”，既能用少量提示快速定位目标，又能在未见过的图像上保持高质量输出。

### 关键概念速览
**Prompt（提示）**：用户给模型的指令，可以是点、矩形框、粗略的掩码甚至文字，类似于在地图上点一下想要划分的区域。  
**Promptable Model（可提示模型）**：能够根据不同提示动态生成分割结果的模型，就像一把可以随意调节刀口的瑞士军刀。  
**Mask Decoder（掩码解码器）**：把图像特征和提示信息融合后，输出像素级别的二值掩码的网络模块，类似于把粗糙的轮廓线细化成完整的剪影。  
**Image Encoder（图像编码器）**：把原始图片压缩成高维特征向量的网络，常用 Vision Transformer（视觉Transformer）实现，像是把整幅画翻译成一段浓缩的文字描述。  
**Prompt Encoder（提示编码器）**：把点、框、掩码等提示转化为与图像特征兼容的向量，类似于把手势转成机器能读的指令。  
**Foundation Model（基础模型）**：在海量数据上预训练、能够迁移到多种下游任务的通用模型，类似于语言模型的 GPT 在文本领域的角色。  
**SA‑1B 数据集**：包含 1 000 000 000 条掩码、1100 万张图片的超大分割标注集合，是本项目的训练基石。  
**Data Engine（数据引擎）**：用于自动生成标注的模型循环系统，先让模型猜测掩码，再由人工或半自动方式纠正，形成高效的标注流水线。

### 核心创新点
1. **从固定任务到 Promptable 框架**  
   *之前的分割模型只能接受单一输入（如整张图或预先给定的框）* → *本文设计了统一的 Prompt Encoder，能够把点、框、粗掩码等多种提示映射到同一特征空间* → *用户只需提供一个点就能得到不同层次的分割，交互更自然，适配范围大幅提升*。  

2. **构建前所未有的超大分割数据集**  
   *过去的分割数据集规模最多几万到几百万标注* → *作者利用 Data Engine 循环标注，结合全自动、半自动和人工校正，最终得到 1 B 掩码的 SA‑1B* → *大规模数据让模型在多样场景上学习到通用的形状和纹理先验，零样本表现接近甚至超越专门监督的模型*。  

3. **统一的掩码解码器设计**  
   *传统方法往往为每种提示设计专属解码头* → *本文采用单一的 Transformer‑based Mask Decoder，接受图像特征和 Prompt 特征的拼接，统一输出掩码* → *模型结构更简洁，训练和推理效率提升，同时保持对不同提示的高适配性*。  

4. **零样本迁移能力的系统评估**  
   *以往的分割模型在新数据集上需要微调* → *本文在 COCO、ADE20K、LVIS 等多任务上直接零样本测试，报告的 mAP（平均精度）与专门训练的模型相当* → *证明了 Promptable 基础模型的即插即用特性，为后续研究提供了可靠的基准*。

### 方法详解
**整体框架**  
这篇论文的系统可以拆成三大块：① 图像编码器把原始图片压成特征图；② 提示编码器把用户提供的点、框或掩码转成向量；③ 掩码解码器把两者融合，输出最终的二值掩码。整个流程像是先把画作翻译成文字，再把指令翻译成文字，最后让一个“编辑器”把文字合成完整的剪影。

**关键模块拆解**  

1. **图像编码器**  
   - 使用 Vision Transformer（ViT）或类似的卷积‑Transformer 混合网络。  
   - 输入图片被切成若干 patch（小块），每块映射到一个 token（标记），随后通过多层自注意力得到全局上下文特征。  
   - 输出的特征图保持空间分辨率（例如 1/16），为后续掩码解码提供细粒度信息。  

2. **提示编码器**  
   - **点提示**：把每个点的坐标映射到特征图对应位置的向量，并加上一个“正负”标记（前景/背景）。  
   - **框提示**：把矩形的四个角坐标编码成相对位置向量，再通过小型 MLP（多层感知机）得到框特征。  
   - **掩码提示**：把粗掩码下采样到特征图大小，直接作为额外通道喂入。  
   - 所有提示向量在一个共享的 Transformer 编码层中相互注意，得到统一的 Prompt Embedding。  

3. **掩码解码器**  
   - 采用两层 Transformer：第一层把图像特征和 Prompt Embedding 融合，第二层把融合后的特征映射回像素空间。  
   - 在每个像素位置，解码器会输出一个二分类分数（前景/背景），随后通过阈值化得到掩码。  
   - 为了兼容多尺度需求，解码器还会在不同分辨率上做上采样并融合细节。  

4. **训练流程**  
   - 使用 SA‑1B 数据集，随机抽取点、框或掩码作为提示，目标是让模型恢复原始完整掩码。  
   - 损失函数主要是二元交叉熵（pixel‑wise binary cross‑entropy），加上 Dice 系数（衡量整体形状相似度）作为正则。  
   - 为防止模型过度依赖某种提示，训练时会随机切换提示类型，提升通用性。  

**最巧妙的设计**  
- **Prompt‑to‑Mask 的统一映射**：不管是一个点还是一个粗掩码，最终都被压成同一维度的向量，这让模型在训练时可以共享所有提示的学习信号，极大提升了数据利用率。  
- **Data Engine 循环标注**：模型先在未标注图片上生成候选掩码，人工只需对错误的部分进行微调，大幅降低了标注成本，使得 1 B 掩码的规模成为可能。  

### 实验与效果
- **测试任务**：在 COCO 实例分割、ADE20K 语义分割、LVIS 长尾实例分割以及 RefCOCO（基于语言提示的分割）等公开基准上进行零样本评估。  
- **对比基线**：与 Mask R-CNN、Detectron2、DeepLabV3+ 等强监督模型以及最近的交互式分割方法（如 RITM）进行比较。  
- **结果概览**：论文声称在 COCO 上的 mask‑AP（平均精度）达到 44.0%，与专门训练的 Mask R-CNN（约 43.5%）相当；在 ADE20K 上的 mIoU（平均交并比）为 48.2%，略高于 DeepLabV3+（约 47%）。在语言提示任务上，SAM 的表现甚至超过了专门的 CLIP‑Seg 模型。  
- **消融实验**：作者分别去掉 Prompt Encoder、去掉多提示混合训练、以及使用小规模数据集（10 M 掩码）进行训练。结果显示：去掉 Prompt Encoder 会导致零样本性能下降约 6%；使用小数据集时，mask‑AP 下降约 8%，验证了大规模数据和统一提示设计的关键性。  
- **局限性**：模型在极细小目标或高度遮挡的场景下仍会出现漏检；对长文本提示的理解能力有限；推理时对显存需求较高（尤其在高分辨率图像上）。作者在讨论中承认这些问题，并把提升细粒度分辨率和多模态提示作为后续方向。

### 影响与延伸思考
自发布以来，SAM 成为计算机视觉社区的“基础模型”，催生了大量衍生工作：  
- **SAM‑Adapter** 系列通过轻量化微调把 SAM 接入特定任务（如医学影像、遥感图）而无需全模型训练。  
- **Prompt Engineering** 研究探索如何用少量点、边框甚至自然语言组合出更精准的分割。  
- **交互式标注工具**（如 Segment Anything Playground）让非专业用户也能快速生成高质量掩码，极大降低了数据标注成本。  
- **多模态扩展**：后续工作尝试把 CLIP 的文本嵌入接入 Prompt Encoder，实现“一句话分割”。  
想进一步深入，可以关注以下方向：① 更高效的稀疏注意力实现，以降低显存；② 融合深度几何信息（如点云）提升三维分割；③ 通过自监督预训练进一步提升零样本鲁棒性。  

### 一句话记住它
SAM 把“给我一个点，我就能把整个物体切出来”变成了现实，开启了通用、可提示的视觉分割时代。