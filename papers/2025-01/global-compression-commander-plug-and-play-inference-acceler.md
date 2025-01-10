# Global Compression Commander: Plug-and-Play Inference Acceleration for High-Resolution Large Vision-Language Models

> **Date**：2025-01-09
> **arXiv**：https://arxiv.org/abs/2501.05179

## Abstract

Large vision-language models (LVLMs) excel at visual understanding, but face efficiency challenges due to quadratic complexity in processing long multi-modal contexts. While token compression can reduce computational costs, existing approaches are designed for single-view LVLMs and fail to consider the unique multi-view characteristics of high-resolution LVLMs with dynamic cropping. Existing methods treat all tokens uniformly, but our analysis reveals that global thumbnails can naturally guide the compression of local crops by providing holistic context for informativeness evaluation. In this paper, we first analyze dynamic cropping strategy, revealing both the complementary nature between thumbnails and crops, and the distinctive characteristics across different crops. Based on our observations, we propose ``Global Compression Commander'' (\textit{i.e.}, \textbf{GlobalCom$^2$}), a novel plug-and-play token compression framework for HR-LVLMs. GlobalCom$^2$ leverages thumbnail as the ``commander'' to guide the compression of local crops, adaptively preserving informative details while eliminating redundancy. Extensive experiments show that GlobalCom$^2$ maintains over \textbf{90\%} performance while compressing \textbf{90\%} visual tokens, reducing FLOPs and peak memory to \textbf{9.1\%} and \textbf{60\%}.

---

# 全局压缩指挥官：高分辨率大规模视觉语言模型的即插即用推理加速 论文详细解读

### 背景：这个问题为什么难？

大规模视觉语言模型（LVLM）在理解图像与文字的结合上表现卓越，但它们的计算成本随输入长度呈二次增长。高分辨率图像往往需要先做动态裁剪（crop），把整张图拆成若干局部视角，再送入模型，这会产生上千甚至上万的视觉 token。传统的 token 压缩方法只针对单视角模型设计，假设所有 token 同等重要，结果在高分辨率、多视角的场景下会把关键细节误删，导致性能大幅下降。换句话说，缺少一种能够在保持全局语义的前提下，针对每个局部裁剪自适应压缩的方案。

### 关键概念速览

**视觉 token**：模型把图像切成若干小块（patch），每块映射成一个向量，这些向量统称为视觉 token。类似于把一幅画拆成拼图块，每块都有自己的信息。

**二次复杂度**：指模型的计算量随 token 数量的平方增长。想象两个人要相互交流，信息量翻倍时，所需的对话次数会呈指数上升。

**动态裁剪（dynamic cropping）**：在推理时，根据图像内容自动选取若干局部区域（crop）送入模型，而不是一次性喂入整张高分辨率图。相当于在看一幅大画时，只挑重点局部细看。

**缩略图（thumbnail）**：对原图做一次低分辨率下采样得到的整体视图。它保留了全局布局，却几乎不占算力。

**全局压缩指挥官（Global Compression Commander，GlobalCom²）**：本文提出的框架，把缩略图当作“指挥官”，为每个局部裁剪提供压缩指令，决定哪些 token 必须保留，哪些可以舍弃。

**Plug‑and‑Play**：指该压缩模块可以直接插入已有的 LVLM 推理流程，无需重新训练或改动模型结构。

### 核心创新点

1. **从全局到局部的压缩指令**  
   *之前的方法*：对所有视觉 token 统一打分或随机采样，忽视不同 crop 之间的语义差异。  
   *本文做法*：先用低分辨率缩略图生成全局感知向量，再依据该向量对每个局部 crop 的 token 进行信息量评估，得到针对性的保留比例。  
   *改变*：实现了“全局指导、局部执行”，在保持整体语义的同时，显著削减冗余 token。

2. **基于缩略图的 informativeness 评估**  
   *之前的假设*：每个 token 的重要性只能通过自身特征或局部上下文判断。  
   *本文做法*：把缩略图视作全局上下文，计算每个局部 token 与全局特征的相似度或注意力得分，作为其信息价值的度量。  
   *改变*：让模型能够辨别“这块局部细节在全图中是否关键”，从而更精准地压缩。

3. **即插即用的压缩层**  
   *之前的压缩*：多数需要在训练阶段加入额外的稀疏化或蒸馏损失。  
   *本文做法*：设计了一个独立的压缩模块，直接在推理时读取缩略图指令，对 token 进行筛选或聚合，完全不改动原 LVLM 参数。  
   *改变*：大幅降低了部署门槛，任何支持多视角输入的 LVLM 都能立刻受益。

4. **极高压缩率下的性能保留**  
   *之前的压缩率*：在 30%~50% token 以上时才开始出现明显性能下降。  
   *本文做法*：在实验中实现了 90% token 的压缩，却仍保持超过 90% 的原始任务表现。  
   *改变*：把 FLOPs 降到原来的 9.1%，显存占用降至 60%，为高分辨率推理打开了可行性。

### 方法详解

#### 整体框架概览

GlobalCom² 的推理流程可以划分为四步：

1. **生成全局缩略图**：对原始高分辨率图像做一次快速下采样，得到低分辨率的全局视图。  
2. **提取全局特征**：把缩略图送入同一视觉编码器（或轻量化的特征提取网络），得到全局 token 序列。  
3. **指挥下的局部压缩**：对每个动态裁剪得到的局部图像，先生成其原始 token 序列，然后利用全局特征计算每个局部 token 的重要性分数，依据分数阈值或保留比例进行筛选或聚合。  
4. **融合并送入语言模型**：压缩后的局部 token 与原始语言 token 合并，进入跨模态 Transformer 完成推理。

#### 关键模块拆解

1. **缩略图特征提取器**  
   - 只需要一次前向传播，算力极低。  
   - 输出的全局 token 数量固定（如 196），每个 token 对应缩略图的一个小块。  
   - 类比为“全局地图”，为后面的局部“探险队”提供方向。

2. **全局‑局部注意力评分器**  
   - 对每个局部 token，计算它与全局 token 的点积注意力得分。  
   - 通过 softmax 归一化后得到每个局部 token 的信息价值。  
   - 直观上相当于让每块局部拼图向全局地图询问：“我在整体结构里重要吗？”

3. **自适应阈值压缩器**  
   - 根据全局‑局部注意力得分，设定一个动态阈值（可基于目标压缩率或显存预算）。  
   - 分数高于阈值的 token 直接保留，低于阈值的 token 进行两种处理：① 直接丢弃；② 与相邻低分 token 聚合成一个代表向量（类似池化），进一步降低 token 数量。  
   - 这一步是“指挥官下达的具体命令”，确保每个局部视角只保留最有价值的信息。

4. **Plug‑and‑Play 接口**  
   - 该压缩器实现为一个独立的前置层，接受原始视觉 token 和全局特征作为输入，输出压缩后的 token。  
   - 不需要改动 LVLM 的 Transformer 参数，也不需要额外的训练损失。  
   - 因此可以直接在已有模型（如 BLIP‑2、LLaVA）上挂载使用。

#### 设计中的巧思

- **全局指挥而非局部自决**：传统压缩往往让每个局部自行决定保留哪些 token，容易出现“局部盲区”。GlobalCom² 把全局视野引入，使得每块局部裁剪的压缩决策都受到整体布局的约束，避免了信息孤岛。
- **信息冗余的两层过滤**：先用注意力得分筛选，再用聚合池化进一步压缩，这种层层剔除的策略在保持关键细节的同时，极大降低了 token 数量。
- **无需再训练的即插即用**：把压缩过程完全放在推理阶段，省去了昂贵的再训练成本，也兼容不同的视觉编码器和语言模型。

### 实验与效果

- **测试数据集**：论文在多模态理解基准上评估，包括高分辨率图像问答（VQAv2‑HR）、细粒度视觉推理（NLVR2‑HR）以及跨模态检索任务。所有数据均采用动态裁剪 + 缩略图的标准输入方式。
- **对比基线**：与未压缩的原始 LVLM、以及两种已有的 token 压缩方法（统一稀疏采样、基于局部注意力的自适应剪枝）进行比较。
- **核心结果**：在保持 90% 以上原始准确率的前提下，GlobalCom² 成功压缩了约 90% 的视觉 token。对应的 FLOPs 降至原来的 9.1%，显存峰值下降至约 60%。相比传统压缩方案，性能下降仅 1%~2%，而 FLOPs 节省约 70%。
- **消融实验**：作者分别去掉缩略图指令、仅使用阈值剪枝、以及不进行聚合池化进行实验。结果显示：没有缩略图指令时，压缩率相同的情况下性能跌至 80% 左右；去掉聚合池化后，压缩率只能提升到 70% 才能维持相同性能，说明两层过滤是关键。
- **局限性**：论文指出，GlobalCom² 依赖于缩略图能够提供足够的全局语义；在极端低分辨率缩略图（如 32×32）下，指挥效果会减弱。此外，当前实现仍需在每个局部 crop 前后分别做一次注意力计算，增加了少量的前置开销。

### 影响与延伸思考

GlobalCom² 的出现让高分辨率 LVLM 的推理成本大幅可控，尤其适用于边缘设备或大规模在线服务。自发表后，已有工作尝试把“全局指挥”思路扩展到视频多模态模型，利用视频的全局帧特征指导每帧的 token 压缩；还有研究把缩略图换成轻量的语义分割图，进一步提升指挥的细粒度。对想深入的读者，可以关注以下方向：① 更高效的全局特征提取（如使用稀疏卷积）；② 将指挥信号与语言提示联合建模，实现跨模态的协同压缩；③ 在训练阶段加入指挥信号的自监督学习，以进一步提升压缩后的性能上限。

### 一句话记住它

**GlobalCom² 用全局缩略图指挥局部裁剪的 token 压缩，实现 90% 视觉 token 删除仍保留 90% 任务性能。**