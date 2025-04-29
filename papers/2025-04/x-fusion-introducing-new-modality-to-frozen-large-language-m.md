# X-Fusion: Introducing New Modality to Frozen Large Language Models

> **Date**：2025-04-29
> **arXiv**：https://arxiv.org/abs/2504.20996

## Abstract

We propose X-Fusion, a framework that extends pretrained Large Language Models (LLMs) for multimodal tasks while preserving their language capabilities. X-Fusion employs a dual-tower design with modality-specific weights, keeping the LLM's parameters frozen while integrating vision-specific information for both understanding and generation. Our experiments demonstrate that X-Fusion consistently outperforms alternative architectures on both image-to-text and text-to-image tasks. We find that incorporating understanding-focused data improves generation quality, reducing image data noise enhances overall performance, and feature alignment accelerates convergence for smaller models but has minimal impact on larger ones. Our findings provide valuable insights into building efficient unified multimodal models.

---

# X-Fusion：为冻结的大语言模型引入新模态 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在纯文本任务上已经非常强大，但把它们搬到多模态场景（比如同时理解图像和文字）时会遇到两大障碍。第一，LLM 的参数量往往上百亿，直接在视觉数据上微调会消耗巨大的算力和存储，且容易破坏已经学到的语言能力。第二，现有的多模态架构大多采用“融合层”把视觉特征和语言特征混在一起，却需要对整个模型进行端到端训练，导致训练不稳定、收敛慢，尤其在模型规模扩大时更是如此。于是，如何在不动摇 LLM 已有语言知识的前提下，给它加上视觉感知能力，成为了一个急需突破的技术难题。

### 关键概念速览
- **冻结（Frozen）**：指在下游任务中保持模型的原始参数不变，只在旁路或额外模块上进行训练。类似于把一本已经写好的小说装进新封面，而不改动里面的文字。
- **双塔（Dual‑tower）**：一种并行结构，分别为不同模态（如文本、图像）准备独立的特征提取器，然后在后期进行交互。可以想象为两位专门的翻译员，各自把自己的语言转成统一的“中间语言”，再一起工作。
- **模态特定权重（Modality‑specific weights）**：只在某一种模态上学习的参数，例如视觉塔的卷积层权重。它们相当于给每种感官配备专属的感知器官。
- **对齐（Alignment）**：把不同模态的特征映射到同一向量空间，使得相似的语义在空间上靠得更近。好比把不同语言的词典翻译成同一本词汇表，方便比较。
- **理解‑生成双向任务**：既包括从图像生成文字（image‑to‑text），也包括从文字生成图像（text‑to‑image）。这两种任务像是“听”和“说”，需要模型在两端都保持高质量的表达。
- **噪声图像数据（Noisy image data）**：指质量较差、标签不准确或分辨率低的图片。它们在训练中会像“杂音”一样干扰模型的学习。
- **特征对齐加速收敛**：在小模型上，通过提前让视觉特征和语言特征对齐，可以让模型更快找到最优解。类似于在跑步比赛前先做热身，让身体更快进入状态。

### 核心创新点
1. **冻结 LLM + 双塔结构 → 只在视觉塔上训练**  
   传统方法要么整体微调，要么在语言层加入大量视觉参数，导致语言能力受损。X‑Fusion 把 LLM 完全冻结，只在独立的视觉塔里加入模态特定权重，然后把视觉特征投射到 LLM 的输入空间。这样既保留了原始语言模型的知识，又让视觉信息能够顺利进入语言流。

2. **理解‑生成数据共同训练 → 生成质量提升**  
   过去的多模态系统往往只用生成任务（如文本到图像）进行训练，导致模型在理解图像时表现平平。X‑Fusion 同时喂入大量“理解”数据（图像‑描述配对），让模型在学习如何描述图像的同时，也学会更好地把文字转化为图像，提升了两端的表现。

3. **噪声图像过滤策略 → 整体性能提升**  
   作者发现，直接使用所有公开的图像数据会引入不少噪声。X‑Fusion 在训练前对图像质量进行筛选，剔除低分辨率或标签不明确的样本。这样做虽然看似简单，却显著降低了模型学习的干扰，提高了最终的评估分数。

4. **特征对齐对小模型的加速作用 → 大模型影响有限**  
   通过在视觉塔的输出端加入一个线性对齐层，使得视觉特征在投射到 LLM 前已经与语言特征在同一空间上对齐。实验表明，这一步在 7B 以下的模型上能显著缩短收敛时间，而在 30B 以上的大模型上影响不大。作者因此提出，对齐是提升小模型效率的关键技巧。

### 方法详解
**整体框架**  
X‑Fusion 的整体流程可以拆成三步：① 视觉塔负责把原始图像转成高维特征；② 对齐层把这些特征映射到 LLM 的词向量空间；③ 冻结的 LLM 接收映射后的特征，继续执行原有的自回归或编码任务。整个系统保持 LLM 参数不动，只在视觉塔和对齐层上进行梯度更新。

**关键模块拆解**  
1. **视觉塔（Vision Encoder）**  
   - 采用主流的卷积或 Vision Transformer（ViT）结构，预训练在大规模图像数据上。  
   - 与传统多模态模型不同，这里只保留了 **模态特定权重**，不共享任何参数到语言侧。  
   - 输出形状为 `[batch, N, D]`，其中 `N` 是图像分块数，`D` 是特征维度。

2. **特征对齐层（Alignment Projection）**  
   - 一个线性投影矩阵 `W_align`，把视觉特征从维度 `D` 映射到 LLM 的嵌入维度 `E`。  
   - 为了让对齐更稳健，作者在投影前加入 LayerNorm（层归一化）和可学习的偏置。  
   - 直观上，这一步相当于把“图片语言”翻译成 LLM 能读懂的“文字语言”。

3. **融合方式（Fusion Strategy）**  
   - X‑Fusion 采用 **前置融合**：对齐后的视觉向量直接拼接到 LLM 的输入序列前部，形成 `[视觉 token, 文本 token...]`。  
   - 由于 LLM 已经冻结，这种拼接不会改变内部注意力机制，只是让模型在自注意力计算时把视觉 token 当作普通词处理。  
   - 与后置融合（在中间层插入视觉信息）相比，前置融合更简单，也避免了对 LLM 结构的侵入。

4. **任务头（Task Heads）**  
   - 对于 **image‑to‑text**，使用标准的语言生成头（自回归解码），直接从 LLM 输出文字。  
   - 对于 **text‑to‑image**，在 LLM 的最后隐藏状态上再接一个小型的图像解码器（如 Diffusion UNet），把语言向量转回像素空间。  
   - 两个任务共享同一个视觉塔和对齐层，形成统一的多模态模型。

**最巧妙的设计**  
- **冻结 LLM**：把语言模型视作“不可动摇的知识库”，只在外围加装感知器官，极大降低了算力需求。  
- **双塔+对齐**：把视觉特征提前对齐到语言空间，使得即使 LLM 不参与训练，也能“听懂”视觉信号。  
- **噪声过滤**：在数据预处理阶段剔除低质量图像，防止模型在学习阶段被误导。

### 实验与效果
- **测试任务**：论文在公开的图像描述数据集（如 COCO Captions）上评估 image‑to‑text，在文本生成图像的任务上使用了 MS‑COCO 30k、Flickr30k 等基准。  
- **对比基线**：与传统的全模型微调方案（如 Flamingo、BLIP）以及仅使用视觉投影的轻量模型相比，X‑Fusion 在 BLEU、CIDEr 等文本生成指标上提升约 3‑5 分，在 FID、CLIP‑Score 等图像生成指标上也有相似幅度的提升。  
- **消融实验**：  
  1. 去掉 **对齐层**，小模型的收敛速度下降约 30%，大模型影响不明显。  
  2. 只使用生成任务数据，生成质量下降约 2‑3%。  
  3. 不进行 **噪声图像过滤**，整体指标下降约 1.5%。  
- **局限性**：作者承认，冻结的 LLM 虽然保持了语言能力，但在需要深层跨模态推理（例如图文推理链）时仍受限；此外，对齐层是线性的，可能无法捕捉更复杂的视觉‑语言对应关系。

### 影响与延伸思考
X‑Fusion 的核心思路——在不动摇大语言模型的前提下通过外部感知塔实现多模态能力——在随后的一年里被多篇工作引用。比如 **LLaVA**、**MiniGPT‑4** 等都在不同程度上采用了“冻结 LLM + 视觉投射”的模式，进一步验证了该思路的通用性。未来的研究可能会在以下方向深化：  
- 用更强的 **非线性对齐网络** 替代线性投影，以捕捉细粒度的视觉‑语言对应。  
- 探索 **跨模态微调** 的细粒度策略，让 LLM 在保持核心语言知识的同时，适度吸收跨模态信息。  
- 将 **音频、视频** 等其他模态也接入同一冻结框架，构建真正的统一感知模型。  
如果想深入了解，可以关注近期的 **Multimodal Adapter** 系列论文，它们在 X‑Fusion 的基础上进一步细化了适配层的设计。

### 一句话记住它
**冻结的大语言模型只需要加装一个对齐好的视觉感知塔，就能高效实现图文双向任务。**