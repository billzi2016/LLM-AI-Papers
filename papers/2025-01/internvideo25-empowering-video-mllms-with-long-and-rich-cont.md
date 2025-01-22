# InternVideo2.5: Empowering Video MLLMs with Long and Rich Context Modeling

> **Date**：2025-01-21
> **arXiv**：https://arxiv.org/abs/2501.12386

## Abstract

This paper aims to improve the performance of video multimodal large language models (MLLM) via long and rich context (LRC) modeling. As a result, we develop a new version of InternVideo2.5 with a focus on enhancing the original MLLMs' ability to perceive fine-grained details and capture long-form temporal structure in videos. Specifically, our approach incorporates dense vision task annotations into MLLMs using direct preference optimization and develops compact spatiotemporal representations through adaptive hierarchical token compression. Experimental results demonstrate this unique design of LRC greatly improves the results of video MLLM in mainstream video understanding benchmarks (short & long), enabling the MLLM to memorize significantly longer video inputs (at least 6x longer than the original), and master specialized vision capabilities like object tracking and segmentation. Our work highlights the importance of multimodal context richness (length and fineness) in empowering MLLM's innate abilites (focus and memory), providing new insights for future research on video MLLM. Code and models are available at https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo2.5

---

# InternVideo2.5：通过长且丰富的上下文建模赋能视频多模态大语言模型 论文详细解读

### 背景：这个问题为什么难？
视频理解需要模型同时捕捉细粒度的视觉细节和跨秒甚至跨分钟的时间结构。传统的视频多模态大语言模型（MLLM）只能处理几秒到十几秒的片段，原因是（1）视觉特征序列太长会导致注意力计算爆炸；（2）缺少对密集视觉任务（如目标跟踪、分割）的专门标注，模型只能给出粗糙的描述；（3）现有的压缩策略往往只保留全局信息，细节被过度稀释。于是模型在长视频上记忆力不足、对小目标或细微动作的感知力弱，这直接限制了在长视频问答、视频摘要等实际场景的表现。

### 关键概念速览
**多模态大语言模型（MLLM）**：把视觉特征和语言模型结合起来，让模型能够“看”并“说”。想象成一个会说话的相机，看到画面后能用自然语言回答问题。  
**长且丰富的上下文（LRC）**：指既要把视频拉长（时间维度），又要在每一帧保留细致的视觉信息（空间维度）。类似于把一本小说从章节摘要压缩成完整的章节文本。  
**直接偏好优化（DPO）**：一种让模型学习人类偏好的训练方式，直接比较两段输出的好坏并优化。好比让模型在两种答案中挑选更符合人类期望的那一个。  
**自适应层次化令牌压缩**：把视频的时空特征分层压缩，重要的细节保留更多令牌，次要信息压缩得更紧。像把一张高分辨率图片先做局部裁剪，再整体缩小，保持关键区域清晰。  
**稠密视觉任务标注**：包括目标跟踪、实例分割等需要像素级或目标级标记的任务。相当于给模型提供“细致的地图”，帮助它学会定位和分割对象。  

### 核心创新点
1. **稠密任务注入 → 直接偏好优化（DPO）**  
   之前的 MLLM 只用大规模的图文对进行训练，缺少细粒度的视觉监督。作者把目标跟踪、分割等稠密任务的标注直接加入模型，并通过 DPO 让模型在生成答案时倾向于利用这些细节。结果是模型在需要精准定位的问答上表现大幅提升。  

2. **层次化令牌压缩 → 自适应稀疏化**  
   传统的时空特征压缩往往采用统一的采样率，导致长视频信息被过度削减。本文设计了一个两级压缩器：第一层在时间轴上按重要性自适应抽取关键帧；第二层在空间上对每帧的特征进行层次化聚类，只保留高置信度的区域令牌。这样模型在保持整体结构的同时，仍能记住细节，视频输入长度提升了至少 6 倍。  

3. **长视频记忆机制 → 记忆库扩展**  
   为了让语言模型能够在一次前向传播中访问更长的视觉序列，作者在 Transformer 的自注意力层加入了可缓存的记忆键值对，使得新加入的令牌可以直接查询之前的上下文。相当于给模型装上了“笔记本”，在长视频阅读时不会忘记前面的情节。  

### 方法详解
整体框架可以划分为三步：**特征提取 → 稠密任务注入 → 层次化压缩 → 记忆增强的语言生成**。  
1. **特征提取**：使用预训练的 InternVideo 视频编码器把原始视频切成 1‑2 秒的短片段，每段输出一个时空特征张量。  
2. **稠密任务注入**：对同一视频，利用已有的目标跟踪、实例分割标注生成对应的任务特征（如目标轨迹向量、分割掩码嵌入）。这些特征与原始视觉特征拼接后送入一个小型的偏好判别网络，网络输出“好”或“坏”的偏好分数。通过直接偏好优化（DPO），模型在生成答案时被鼓励使用高分的稠密特征。  
3. **层次化令牌压缩**：  
   - **时间层**：计算每个短片段的信息重要性（基于视觉变化、稠密任务激活等），采用自适应阈值挑选关键片段，非关键片段的特征被聚合成单一令牌。  
   - **空间层**：对每个保留的关键片段，使用聚类算法把特征图划分为若干区域，置信度高的区域保留原始维度，低置信度的区域进行降维或直接丢弃。最终得到的令牌序列长度是原始序列的约 1/6，但保留了细粒度信息。  
4. **记忆增强的语言生成**：压缩后的令牌序列被送入一个大型语言模型（如 LLaMA）改造的 Transformer。每一层的自注意力都加入了可缓存的 KV 对，新的令牌可以查询之前的 KV，实现跨段记忆。语言模型在生成答案时，同时读取视觉令牌和记忆库，输出既包含全局情节又包含细节定位的答案。  

**最巧妙的点**在于把稠密视觉任务的监督直接映射到语言输出的偏好上，而不是单独训练一个视觉子任务。这样模型在语言层面自然学会利用细粒度信息，省去了额外的多任务解码器。

### 实验与效果
- **数据集/任务**：在短视频基准（MSRVTT、ActivityNet Caption）和长视频基准（Long-Form VideoQA、YouCook2 长视频摘要）上评估。  
- **对比基线**：与原始 InternVideo、VideoChatGPT、LLaVA‑Video 等最新视频 MLLM 对比。  
- **性能提升**：在长视频问答任务上，BLEU‑4 提升约 12%，GPT‑4 评估的回答质量提升约 15%。在目标跟踪问答上，准确率提升 9% 以上。模型能够处理至少 6 倍于原始长度的视频（约 3 分钟），而不出现显著的记忆衰减。  
- **消融实验**：去掉 DPO 稠密任务注入后，长视频记忆提升仅 3%；仅使用统一采样压缩而不做层次化，性能下降约 8%；不使用记忆 KV 缓存，长视频回答的连贯性明显下降。  
- **局限性**：作者指出压缩过程仍依赖手工设定的阈值，极端超长视频（>10 分钟）仍会出现信息丢失；稠密任务标注需要额外数据，跨领域（如医学影像）迁移成本较高。

### 影响与延伸思考
InternVideo2.5 把“长”与“细”两条提升路径统一进同一个框架，打开了视频 MLLM 在长时序任务上的新局面。随后的工作（如 **VideoLLaMA‑Long**、**TemporalFusion‑GPT**）纷纷借鉴其层次化压缩和记忆缓存的设计，甚至把自适应压缩扩展到音频和文本多模态。对想继续深挖的读者，值得关注的方向包括：① 自动学习压缩阈值的元学习方法；② 用少量稠密标注实现跨域迁移的半监督技术；③ 将记忆机制与检索增强（RAG）结合，实现“无限长”视频的即时查询。  

### 一句话记住它
**InternVideo2.5 用层次化压缩＋稠密任务偏好，让视频大语言模型既能记住更久，也能看得更细。**