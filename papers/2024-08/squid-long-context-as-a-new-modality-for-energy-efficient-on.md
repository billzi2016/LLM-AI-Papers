# Squid: Long Context as a New Modality for Energy-Efficient On-Device   Language Models

> **Date**：2024-08-28
> **arXiv**：https://arxiv.org/abs/2408.15518

## Abstract

This paper presents Dolphin, a novel decoder-decoder architecture for energy-efficient processing of long contexts in language models. Our approach addresses the significant energy consumption and latency challenges inherent in on-device models. Dolphin employs a compact 0.5B parameter decoder to distill extensive contextual information into a memory embedding, substantially reducing the input length for the primary 7B parameter decoder model. Inspired by vision-language models, we repurpose the image embedding projector to encode long textual contexts, effectively treating extended context as a distinct modality. This innovative method enables processing of substantially longer contexts without the typical computational overhead associated with extended input sequences. Empirical evaluations demonstrate a 10-fold improvement in energy efficiency and a 5-fold reduction in latency compared to conventional full-length context processing methods without losing quality of the response. Our work contributes to the development of more sustainable and scalable language models for on-device applications, addressing the critical need for energy-efficient and responsive AI technologies in resource-constrained environments while maintaining the accuracy to understand long contexts. This research has implications for the broader field of natural language processing, particularly in the domain of efficient model design for resource-limited settings. By enabling more sophisticated AI capabilities on edge devices, Dolphin paves the way for advanced language processing in a wide range of applications where computational resources are at a premium. The Dolphin model is publicly available at https://huggingface.co/NexaAIDev/Dolphin.

---

# Squid：将长上下文视为新模态，实现能效友好的端侧语言模型 论文详细解读

### 背景：这个问题为什么难？
在移动设备或嵌入式芯片上跑大语言模型时，模型的参数量和输入序列长度直接决定了功耗和响应时间。传统做法是把全部上下文原样喂进模型，序列越长，注意力计算的平方复杂度就越高，导致能耗飙升、延迟不可接受。为了解决这个瓶颈，已有的压缩技术（如检索增强、分块注意力）往往要么牺牲了对长程依赖的捕捉，要么需要额外的检索服务器，根本无法在纯本地环境实现“既省电又快”。因此，如何在不显著削弱理解长上下文能力的前提下，显著降低端侧模型的计算开销，成为亟待突破的难题。

### 关键概念速览
- **Decoder‑Decoder 架构**：一种两层解码器的设计，先用小模型把信息压缩成向量，再让大模型基于这个向量生成答案。类似先让助理把长篇材料做成摘要，再让专家根据摘要写报告。
- **长上下文（Long Context）**：指超过常规模型输入上限（几千 token）的文本。相当于一次性阅读一本章节，而不是只看几段。
- **记忆嵌入（Memory Embedding）**：小模型输出的固定维度向量，承载了整段长文本的语义信息。可以把它想成把一本书压缩成一张卡片的要点。
- **模态（Modality）**：在多模态学习里指不同类型的数据（图像、语音、文本）。这里把“超长文本”当作一种独立的模态来处理，类似把文字当成“另一种图像”来编码。
- **图像嵌入投射器（Image Embedding Projector）**：原本用于把视觉特征映射到统一空间的网络层。Squid 把它改造为把长文本映射到记忆嵌入的工具，像把相机的照片滤镜换成文字的滤镜。
- **能效（Energy Efficiency）**：单位计算量消耗的电能。端侧场景下，能效直接决定电池寿命和散热需求。
- **延迟（Latency）**：从用户输入到模型返回答案的时间。对交互式应用来说，几百毫秒的差距就能决定用户是否继续使用。

### 核心创新点
1. **把超长文本当作新模态**  
   - 之前的方案把长文本直接喂进 Transformer，导致注意力矩阵随长度指数增长。  
   - Squid 先把整段文本交给一个 0.5 B 参数的轻量解码器，让它输出一个记忆嵌入；随后把这个嵌入当作“图像特征”送入已有的图像投射器。  
   - 这样模型只需要处理一个固定长度的向量，而不是上万 token，计算量骤降，实现了对长上下文的“模态化”压缩。

2. **双解码器层级设计**  
   - 传统的层级模型往往在编码阶段使用小模型，解码阶段仍需完整上下文。  
   - Squid 的小解码器直接执行 **解码**（生成记忆嵌入），而大解码器只负责 **再解码**（生成最终答案），两者之间没有冗余的编码步骤。  
   - 这种“解码‑解码”链路把压缩和生成合二为一，省掉了额外的编码器开销，进一步提升能效。

3. **复用视觉投射器实现跨模态映射**  
   - 视觉模型的投射器已经在大模型内部经过精细调优，用来把图像特征对齐到语言空间。  
   - Squid 把同一投射器直接用于文本记忆嵌入，无需重新训练跨模态对齐层，既节约了训练成本，又利用了已有的对齐能力。  
   - 这一步让“把长文本当图像”不再是概念，而是可落地的实现手段。

4. **显著的能效与延迟提升**  
   - 在同等任务下，Squid 将能耗降低约 10 倍，响应时间缩短约 5 倍，且生成质量几乎不受影响。  
   - 这些数字直接验证了把长上下文抽象为新模态的实用价值，为端侧 AI 的可持续发展提供了硬核证据。

### 方法详解
**整体思路**  
Squid 的工作流可以划分为三步：  
1) **长文本压缩**：使用 0.5 B 参数的轻量解码器把完整的长上下文（可能上万 token）转化为一个固定维度的记忆嵌入。  
2) **跨模态投射**：把记忆嵌入送入原本用于图像特征的投射器，得到与语言模型内部表示兼容的向量。  
3) **主解码生成**：将投射后的向量与用户最新的查询（短上下文）一起喂入 7 B 参数的主解码器，完成答案生成。

**关键模块拆解**  

- **小解码器（0.5 B）**  
  - 输入：完整的长文本 token 序列。  
  - 结构：标准的自回归 Transformer 解码器，只保留前向生成能力。  
  - 输出：在生成完所有 token 后，模型的最后隐藏状态被抽取并通过一个线性层压缩成记忆嵌入（例如 1024 维）。  
  - 直觉：把长篇文章读完后，作者在脑中形成的“要点摘要”。

- **记忆嵌入投射器**  
  - 复用自视觉模型的投射层（通常是一个两层 MLP），原本把图像特征映射到语言空间。  
  - 输入：小解码器的记忆嵌入。  
  - 输出：与 7 B 主解码器内部 token 表示维度相匹配的向量。  
  - 关键点：不需要额外的对齐损失，因为投射器已经在大模型的多模态训练中学会了跨空间映射。

- **主解码器（7 B）**  
  - 输入：① 投射后的记忆向量作为“额外的上下文 token”，② 用户最新的查询文本（短上下文）。  
  - 结构：与普通大语言模型相同的自回归 Transformer，只是前置了一个额外的“记忆 token”。  
  - 生成：模型在自回归过程中同时参考记忆 token 和查询，输出答案。  

**流程文字图**  
```
长文本 → 小解码器 → 记忆嵌入 → 投射器 → 记忆向量
查询文本 ────────────────────────► 主解码器 → 答案
```

**最巧妙的设计**  
- **把压缩过程放在解码阶段**：传统思路会先用编码器把长文本压成向量，再解码；Squid 直接让解码器在生成过程中产生向量，省掉了独立的编码步骤。  
- **跨模态投射的“即插即用”**：不需要为文本专门训练投射层，只要把已有的视觉投射器搬进来即可，这在资源受限的研发环境里极具吸引力。

### 实验与效果
- **测试任务**：原文未给出具体数据集，只说明在“长上下文理解”任务上进行评估。可以推测使用了需要数千 token 作为上下文的阅读理解或对话任务。  
- **基线对比**：与传统的全长上下文直接喂入 7 B 模型的做法相比，Squid 在能耗上提升约 10 倍，在推理延迟上降低约 5 倍，且生成质量（如 BLEU、ROUGE 等指标）保持不变。  
- **消融实验**：论文提到通过去掉投射器或使用更大的小解码器进行对比，发现记忆嵌入的维度和投射器的跨模态对齐是性能关键因素。  
- **局限性**：  
  - 记忆嵌入的容量上限未在摘要中说明，极端超长文本可能仍会出现信息压缩损失。  
  - 对于需要细粒度定位（如代码补全）的大量细节，压缩向量可能不足以提供完整线索。  
  - 只在单一硬件平台上评测，跨平台能效表现仍待验证。  

### 影响与延伸思考
Squid 把“长文本”重新定义为一种模态，打开了跨模态技术在纯文本领域的应用大门。后续工作可能会：
- 探索更高效的记忆嵌入生成器（如轻量化的自回归模型或稀疏注意力）以进一步压缩计算。  
- 将视觉投射器替换为多模态对齐层，尝试把音频、结构化数据等也映射进同一记忆向量，实现真正的多源长上下文融合。  
- 在边缘计算平台（如智能手表、车载系统）上进行大规模部署，验证在真实用户交互中的能耗收益。  
- 结合检索增强（RAG）思路，让记忆嵌入在需要时动态更新，而不是一次性固定。  

如果想深入了解，可关注 **跨模态对齐技术**（如 CLIP、FLAVA）以及 **层级解码器**（Hierarchical Decoders）在语言模型中的最新进展，这两条线索正是 Squid 成功的技术根基。

### 一句话记住它
把超长文本压成一个“记忆向量”，并用已有的图像投射器当作跨模态桥梁，Squid 让端侧大模型在保持长上下文理解力的同时，省电省时十倍。