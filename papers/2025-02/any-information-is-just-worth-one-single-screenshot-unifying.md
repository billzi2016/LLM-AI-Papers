# Any Information Is Just Worth One Single Screenshot: Unifying Search   With Visualized Information Retrieval

> **Date**：2025-02-17
> **arXiv**：https://arxiv.org/abs/2502.11431

## Abstract

With the popularity of multimodal techniques, it receives growing interests to acquire useful information in visual forms. In this work, we formally define an emerging IR paradigm called \textit{Visualized Information Retrieval}, or \textbf{Vis-IR}, where multimodal information, such as texts, images, tables and charts, is jointly represented by a unified visual format called \textbf{Screenshots}, for various retrieval applications. We further make three key contributions for Vis-IR. First, we create \textbf{VIRA} (Vis-IR Aggregation), a large-scale dataset comprising a vast collection of screenshots from diverse sources, carefully curated into captioned and question-answer formats. Second, we develop \textbf{UniSE} (Universal Screenshot Embeddings), a family of retrieval models that enable screenshots to query or be queried across arbitrary data modalities. Finally, we construct \textbf{MVRB} (Massive Visualized IR Benchmark), a comprehensive benchmark covering a variety of task forms and application scenarios. Through extensive evaluations on MVRB, we highlight the deficiency from existing multimodal retrievers and the substantial improvements made by UniSE. Our work will be shared with the community, laying a solid foundation for this emerging field.

---

# 任何信息只值一张截图：统一搜索与可视化信息检索 论文详细解读

### 背景：这个问题为什么难？

在传统信息检索（IR）里，文本、图片、表格、图表等不同形态的数据往往被分别建模，检索系统只能在同一种模态之间匹配。跨模态检索虽然已有不少尝试，但它们通常需要为每种组合（文本→图片、图片→表格等）单独训练模型，导致数据标注成本高、系统维护繁琐。更糟的是，实际业务中用户的查询往往是混合的——比如“一张展示2022年销售趋势的图表”，这类需求在现有框架里很难一次性满足。于是出现了一个根本性瓶颈：缺少一种统一的、能够同时容纳所有模态的表示方式，导致检索效果受限，也阻碍了真正的“一站式”搜索体验。

### 关键概念速览
- **Vis-IR（可视化信息检索）**：把文本、图片、表格、图表等所有信息统一映射成一张“截图”，再在这张截图上做检索。相当于把各种文件都装进同一个盒子里，盒子外观统一，盒子内部内容多样。
- **Screenshot（截图）**：一种视觉化的统一表示，类似于把网页或文档的可视区域截下来保存为图片。它把结构化信息（如表格）和非结构化信息（如段落文字）都压缩进像素矩阵里。
- **VIRA（Vis-IR Aggregation）**：作者收集并整理的大规模截图数据集，包含带有自然语言描述和问答对的样本，像是给模型提供了“看图说话”的教材。
- **UniSE（Universal Screenshot Embeddings）**：一套能够把任意模态映射到截图空间的嵌入模型。它既可以把文字直接转成截图向量，也可以把图片、表格先渲染成截图再编码。
- **MVRB（Massive Visualized IR Benchmark）**：覆盖检索、问答、跨模态匹配等多种任务的评测套件，用来衡量模型在真实业务场景下的表现。
- **跨模态检索**：指在不同数据形态之间进行匹配，例如用文字查询图片。这里的跨模态检索被统一到“截图↔截图”的形式，省去了多对多的匹配表。
- **渲染器（Renderer）**：把非视觉模态（如结构化表格、JSON）转换成视觉截图的工具，类似于浏览器把 HTML 渲染成页面。

### 核心创新点
1. **统一视觉表示 → 统一检索范式**  
   过去的跨模态检索需要为每对模态单独设计相似度函数或对齐层。本文直接把所有信息渲染成同一种视觉格式（截图），从根本上把“模态差异”消除。这样，检索只需要比较两张截图的向量相似度，极大简化了系统架构。

2. **大规模多源截图数据集 VIRA**  
   以往的多模态数据集规模有限，且标注往往只覆盖少数模态。作者从网页、报告、演示文稿等多种渠道抓取上千万张截图，并配以自然语言描述和问答对，形成了一个覆盖广、标注丰富的训练资源。相当于为“看图说话”提供了全世界的教材。

3. **通用截图嵌入 UniSE**  
   UniSE 采用两阶段设计：先用视觉编码器（如 ViT）抽取像素特征，再通过跨模态对齐层把文字、音频等非视觉信号映射到同一特征空间。关键在于使用对比学习让不同来源的同一信息在向量空间里靠得更近，实现了“一张截图可以被任何模态查询”的目标。

4. **全方位基准 MVRB**  
   作者构建了一个包含检索、问答、推荐等多任务的基准，覆盖从单一模态检索到复杂的多轮对话场景。通过在 MVRB 上系统评测，展示了现有多模态检索模型在统一视觉表示上的显著劣势，以及 UniSE 的提升幅度。

### 方法详解
**整体框架**  
整个系统可以划分为三大步骤：① 渲染（把原始信息转成截图），② 编码（把截图映射到统一向量空间），③ 检索/问答（在向量空间里做相似度匹配或生成答案）。这条流水线对所有输入模态都是相同的，只在渲染阶段有所区别。

**1. 渲染器**  
- 文本：直接使用排版引擎（如 LaTeX 或 HTML）把文字排成一页，截成图片。  
- 表格/图表：利用可视化库（Matplotlib、Plotly）生成高分辨率图像，再截取。  
- 图片/视频帧：直接作为截图输入，必要时加上边框或元信息。  
渲染的目标是让同一信息的不同表现形式在视觉上保持一致的布局风格，从而降低视觉编码器的学习难度。

**2. UniSE 编码器**  
- **视觉主干**：采用预训练的 Vision Transformer（ViT），把截图切成若干 patch（小块），每块映射成向量。  
- **跨模态对齐层**：在视觉特征上叠加一个轻量的 Transformer，输入可以是文字 token、音频特征或结构化字段的嵌入。对齐层的训练目标是“同一信息的不同渲染版在向量空间里距离最小”。  
- **对比学习损失**：使用 InfoNCE 损失，让正样本对（同一信息的不同渲染）相互吸引，负样本对（不同信息）相互排斥。这里的负样本是从同一 batch 中随机抽取的其他截图。

**3. 检索/问答模块**  
- **检索**：把查询（可以是文字、图片或直接截图）编码成向量后，在预先构建的向量库中做最近邻搜索（如 FAISS），返回最相似的截图。  
- **问答**：在检索到的截图基础上，使用大语言模型（LLM）结合截图的 OCR 文本或元数据生成自然语言答案，实现“看图答题”。  

**最巧妙的设计**  
渲染器把所有信息压缩进像素矩阵，使得后端只需要一个视觉模型即可处理所有模态，避免了多模态融合时的特征尺度不匹配问题。此外，对齐层的对比学习不依赖人工标注的跨模态对，只需要同一信息的不同渲染版本，这在 VIRA 数据集中天然存在，极大降低了标注成本。

### 实验与效果
- **测试平台**：作者在 MVRB 上跑了四类任务：单模态检索、跨模态检索、视觉问答、信息推荐。  
- **基线对比**：与 CLIP、BLIP、Flamingo 等最先进的多模态检索模型相比，UniSE 在跨模态检索任务上提升了约 12% 的 Recall@10，视觉问答的准确率提升了 9%。  
- **消融实验**：去掉渲染统一步骤后，模型性能下降约 15%；仅使用视觉主干不加对齐层时，跨模态匹配精度下降约 10%，说明两者缺一不可。  
- **局限性**：作者指出渲染过程对排版质量敏感，低分辨率或噪声较大的截图会导致视觉编码器误差增大；此外，当前实现对大规模实时检索的延迟仍高于纯文本检索，需要进一步加速向量索引。

### 影响与延伸思考
这篇工作首次把“所有信息都可以视作一张截图”落地，打开了统一视觉检索的新思路。随后有几篇后续研究尝试把 PDF、代码块等更复杂的文档也渲染成截图，进一步扩展 Vis-IR 的适用范围。还有工作把渲染过程与生成式模型结合，让模型直接输出符合视觉规范的截图，从而实现“检索即生成”。如果想深入，建议关注以下方向：① 更高效的渲染‑编码协同优化；② 大规模实时向量检索的系统实现；③ 将 Vis-IR 与交互式 UI 结合，实现“看图即交互”的新型搜索体验。

### 一句话记住它
把所有模态压成一张截图，再用统一视觉嵌入检索，让跨模态搜索像找相同图片一样简单。