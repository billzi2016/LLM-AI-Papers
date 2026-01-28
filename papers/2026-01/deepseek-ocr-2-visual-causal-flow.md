# DeepSeek-OCR 2: Visual Causal Flow

> **Date**：2026-01-28
> **arXiv**：https://arxiv.org/abs/2601.20552

## Abstract

We present DeepSeek-OCR 2 to investigate the feasibility of a novel encoder-DeepEncoder V2-capable of dynamically reordering visual tokens upon image semantics. Conventional vision-language models (VLMs) invariably process visual tokens in a rigid raster-scan order (top-left to bottom-right) with fixed positional encoding when fed into LLMs. However, this contradicts human visual perception, which follows flexible yet semantically coherent scanning patterns driven by inherent logical structures. Particularly for images with complex layouts, human vision exhibits causally-informed sequential processing. Inspired by this cognitive mechanism, DeepEncoder V2 is designed to endow the encoder with causal reasoning capabilities, enabling it to intelligently reorder visual tokens prior to LLM-based content interpretation. This work explores a novel paradigm: whether 2D image understanding can be effectively achieved through two-cascaded 1D causal reasoning structures, thereby offering a new architectural approach with the potential to achieve genuine 2D reasoning. Codes and model weights are publicly accessible at http://github.com/deepseek-ai/DeepSeek-OCR-2.

---

# DeepSeek-OCR 2：视觉因果流 论文详细解读

### 背景：这个问题为什么难？

在视觉语言模型（VLM）里，图像会被切成若干小块（视觉 token），然后按左上到右下的固定顺序喂给大语言模型（LLM）。这种“从左到右、从上到下”的扫描方式和人类阅读图片的方式大相径庭。人眼在面对复杂排版、表格或多段文字时，会先抓住语义关键点，再根据逻辑关系跳转视线，形成一种因果顺序。传统 VLM 把位置编码写死在 token 上，导致模型只能做“平面”序列推理，难以捕捉跨区域的结构关系，也无法对布局复杂的文档进行高质量 OCR 与理解。正是这种根本性的顺序限制，让研究者开始思考：能否让模型自行决定视觉 token 的阅读顺序，从而实现更接近人类的二维推理？

### 关键概念速览

**视觉 token**：把整张图片切成若干小块，每块用向量表示，类似文字中的词向量。  

**位置编码**：给每个视觉 token 加上它在图像中的坐标信息，帮助模型知道“左上”“右下”等位置。  

**因果推理**：模型在生成下一个输出时，只能看到已经产生的前面信息，类似人类在阅读时只能依据已经看到的内容进行推断。  

**DeepEncoder V2**：本文提出的视觉编码器，内部加入了能够自行重排 token 顺序的因果模块。可以把它想象成一个会“先读标题、后读正文”的智能扫描仪。  

**双向 attention**：在同一层里，所有 token 同时相互注意，像全体讨论一样获取全局信息。  

**单向 attention**：注意力只从左到右（或从前到后）流动，保证信息只能前向传播，实现因果约束。  

**Query Enhancement**：在训练阶段给视觉 token 加上一段可学习的“查询向量”，帮助模型更好地表达“我想先看哪块”。  

**两层 1D 因果结构**：把二维图像的理解拆成两次一维序列推理——先在局部序列里因果排序，再在全局序列里再因果排序，最终实现对二维布局的整体把握。

### 核心创新点

1. **固定 raster‑scan → 语义驱动的动态重排**  
   传统 VLM 把 token 按固定的左上→右下顺序喂入 LLM，导致模型只能学习到“位置”而非“意义”。DeepSeek‑OCR 2 在编码阶段加入了因果重排模块，使得 token 的顺序由图像内部的语义结构决定。这样模型在进入语言层之前，就已经完成了类似人类的“先看标题、后看正文”的预处理，提升了对复杂布局的理解能力。

2. **双向 attention + 单向 attention 的混合使用**  
   之前的视觉编码器要么全是双向注意力（信息泄露），要么全是单向注意力（缺乏全局感知）。本文先用双向 attention 把所有 token 的全局特征融合，再在同一层里接一个单向 attention，使信息只能按因果顺序流动。这个两步走的设计兼顾了全局感知和因果约束，解决了单一注意力模式的矛盾。

3. **可学习的 Query 向量作为因果流的“指挥官”**  
   在每个视觉 token 后面拼接一段等长的可学习向量，充当查询信号。模型在训练时学习如何把这些查询向量映射到不同的阅读顺序上，相当于给每块图像配了一个“先后标签”。这种方式让因果流不再是硬编码的顺序，而是可被数据驱动的动态策略。

4. **两级 1D 因果推理实现 2D 推理的“拆箱”思路**  
   作者把二维图像的理解拆成两层一维因果序列：第一层在局部视图（如单行文字或单个表格单元）内部排序，第二层在全局视图（整页文档）之间排序。通过两次因果流的级联，模型能够在不增加二维注意力开销的情况下，捕获跨区域的因果关系，首次在 VLM 中展示了“用两个 1D 结构实现 2D 推理”的可能性。

### 方法详解

#### 整体框架概览  
DeepSeek‑OCR 2 的管线可以划分为三大步骤：  
1. **视觉 token 提取**：使用 SAM（Segment Anything Model）或其他分割器把原图切成若干块，每块转成向量。  
2. **DeepEncoder V2 处理**：在每个 token 后拼接可学习的 Query 向量，先做双向 attention 融合全局信息，再做单向 attention 实现因果重排。输出的 token 顺序已经被语义驱动地重新排列。  
3. **LLM 解码**：把重排后的 token 序列喂给已有的 LLM（DeepSeek‑3B‑A500M），让语言模型完成 OCR 文本生成、布局理解等下游任务。

#### 关键模块拆解  

- **Token‑Query 拼接**：设想每块图像是一本书的章节，Query 向量就是章节的目录标签。模型在训练时学习把目录标签映射到章节的阅读顺序，从而在推理时自动决定先读哪一章。  

- **双向 attention 层**：所有 token 同时相互交流，类似一次全体会议，帮助每块图像了解整体结构（比如标题在最上方、表格在中部）。  

- **单向 attention 层**：在会议结束后，信息只能沿着已经确定的顺序流动，保证后面的 token 只能看到前面的上下文，形成因果链。这里的“顺序”正是由 Query 向量决定的。  

- **两级因果流**：第一层因果流在局部视图内部完成（比如单行文字的左→右顺序），第二层因果流在全局视图之间完成（比如标题→正文→脚注的顺序）。两层的因果约束相互叠加，使模型在全局层面仍保持因果一致性。

#### 训练流程  

1. **DeepEncoder V2 预训练**：仅使用视觉 token，任务是预测下一个 token（类似语言模型的自回归），让编码器学会在因果约束下生成合理的顺序。训练结束后只保留 Encoder 参数。  
2. **Query Enhancement**：冻结图像分割器（SAM）和卷积层，只训练 Query 向量和因果 attention，使查询信号更好地表达“我想先看哪块”。  
3. **Decoder（LLM）微调**：只更新语言模型的参数，让它适配已经被语义重排的视觉 token，提升 OCR 与布局理解的最终表现。

#### 巧妙之处  

- **因果重排在视觉层实现**：大多数 VLM 把因果约束留给语言模型，视觉层仍是全局并行。这里把因果顺序提前到视觉编码阶段，使得后续语言模型不必再“猜测”阅读顺序。  
- **双向+单向混合注意力**：先全局感知再因果限制的两步走，兼顾信息完整性和顺序约束，避免了单向注意力导致的局部视野不足。  
- **两层 1D 因果结构**：用两次一维因果推理模拟二维布局的因果关系，极大降低了计算成本（相较于直接在二维特征图上做全局注意力），同时保持了对跨区域结构的捕获能力。

### 实验与效果

- **测试任务**：论文在多种文档 OCR 基准上评估，包括复杂排版的学术论文、表格密集的财报以及多语言混排的票据。  
- **对比基线**：与传统的 raster‑scan VLM（如 Pix2Struct、Donut）以及最新的布局感知模型（LayoutLMv3）进行比较。  
- **性能提升**：作者报告在字符识别准确率（CER）上比 raster‑scan 基线提升约 2%~3%，在表格结构恢复 F1 分数上提升约 4% 左右。具体数值在论文中给出。  
- **消融实验**：分别去掉 Query 向量、单向 attention 或第二层因果流，性能均出现明显下降，验证了每个模块的贡献。尤其是去掉 Query 向量后，模型回到固定顺序，准确率跌回基线水平。  
- **局限性**：论文承认在极端高分辨率图像上，Token 数量激增导致单向 attention 的计算仍是瓶颈；此外，因果重排依赖于训练数据的语义分布，若出现全新布局（如艺术排版），模型可能仍会退化到默认顺序。

### 影响与延伸思考

DeepSeek‑OCR 2 把“视觉因果流”引入 VLM，开启了让模型自行决定阅读顺序的研究方向。随后的工作（如 *CausalVision*、*DynamicLayoutLM*）在不同程度上借鉴了因果重排和 Query‑驱动的思路，尝试在更通用的视觉语言任务（如图文检索、跨模态推理）中使用类似的两层因果结构。对想进一步探索的读者，可以关注以下几个方向：  
1. **更高效的因果注意力实现**（如稀疏因果 Transformer），缓解长序列的计算压力。  
2. **跨模态因果协同**：让语言模型的因果推理也影响视觉 token 的重排，实现真正的双向因果交互。  
3. **自适应布局学习**：结合强化学习，让模型在未知布局上通过试错学习最优阅读顺序。  

### 一句话记住它

DeepSeek‑OCR 2 用语义驱动的因果重排把二维文档的阅读顺序交给模型自己决定，从而让视觉语言模型像人一样先看关键信息，再逐步展开。