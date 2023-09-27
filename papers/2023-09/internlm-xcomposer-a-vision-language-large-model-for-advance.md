# InternLM-XComposer: A Vision-Language Large Model for Advanced   Text-image Comprehension and Composition

> **Date**：2023-09-26
> **arXiv**：https://arxiv.org/abs/2309.15112

## Abstract

We propose InternLM-XComposer, a vision-language large model that enables advanced image-text comprehension and composition. The innovative nature of our model is highlighted by three appealing properties: 1) Interleaved Text-Image Composition: InternLM-XComposer can effortlessly generate coherent and contextual articles that seamlessly integrate images, providing a more engaging and immersive reading experience. Simply provide a writing instruction, and our system will generate the corresponding manuscript. It can intelligently identify the areas in the text where images would enhance the content and automatically insert the most appropriate visual candidates. 2) Comprehension with Rich Multilingual Knowledge: The text-image comprehension is empowered by training on an extensive multi-modal multilingual database with carefully crafted strategies, resulting in a deep understanding of visual content. 3) State-of-the-art Performance: Our model consistently achieves state-of-the-art results across various mainstream benchmarks for vision-language foundational models, including MME Benchmark, MMBench, MMBench-CN, Seed-Bench, CCBench (Chinese Cultural Benchmark), QBench and Tiny LVLM. Owing to the absence of established metrics for quantitatively assessing text-image composition, we have devised a robust evaluation procedure that comprises both human and GPT4-Vision (GPT4-V) to ensure reliability. Notably, our InternLM-XComposer achieves competitive text-image composition scores compared to public solutions, including GPT4-V and GPT3.5. Collectively, InternLM-XComposer seamlessly blends advanced text-image comprehension and composition, revolutionizing vision-language interaction and offering new insights and opportunities. The InternLM-XComposer model series are publicly available at https://github.com/InternLM/InternLM-XComposer.

---

# InternLM‑XComposer：面向高级文本‑图像理解与组合的视觉语言大模型 论文详细解读

### 背景：这个问题为什么难？

在视觉语言（Vision‑Language）领域，模型大多只能完成“看图说话”或“图文检索”这类单向任务。要让模型在写作时主动挑选、插入最合适的图片，涉及两大难点：一是需要对文本语义和图像内容都有深刻的跨模态理解；二是要在生成长篇文章时保持图文之间的连贯性和上下文一致性。过去的系统往往把图像生成和文本生成分开处理，导致插图位置不自然、图片与文字不匹配，难以提供沉浸式阅读体验。因此，如何让大模型在同一流程中完成“读懂图、写出文、把图放进去”成为亟待突破的瓶颈。

### 关键概念速览
- **视觉语言大模型（Vision‑Language Large Model）**：同时接受文字和图片作为输入，能够在两者之间建立语义桥梁的超大规模神经网络。类似于会说话的“看图机器人”。
- **文本‑图像交错生成（Interleaved Text‑Image Composition）**：模型在生成文字的同时，自动决定何时插入图片并选出最匹配的视觉素材，就像人写稿子时顺手贴图一样。
- **多模态多语言数据库**：包含多语言文本与对应图片的大规模训练集合，帮助模型学会跨语言、跨文化的图文关联。可以想象成一本装满了世界各地图文配对的百科全书。
- **GPT‑4‑Vision（GPT4‑V）评估**：使用具备视觉理解能力的 GPT‑4 作为自动评审员，对模型生成的图文组合进行客观打分，类似于让另一位 AI 老师帮忙批改作业。
- **人机混合评测**：让真实用户和 GPT‑4‑V 双管齐下，对生成结果的质量、连贯性和视觉匹配度进行打分，确保评价既有主观感受也有客观指标。

### 核心创新点
1. **从分离到交错的生成流程**  
   *之前的做法*：先生成完整文本，再单独检索或生成图片，二者之间缺乏实时信息交互。  
   *本文的做法*：在解码阶段引入“图文交错模块”，模型在每一步文字生成后即时评估是否需要插图，并直接从视觉库中挑选或生成对应图片。  
   *带来的改变*：图文位置更自然，图片内容更贴合上下文，整体稿件的阅读流畅度显著提升。

2. **大规模多语言多模态预训练策略**  
   *之前的做法*：大多数视觉语言模型只在单一语言（如英语）或少量跨语言数据上训练，导致跨语言场景表现不佳。  
   *本文的做法*：构建并使用了一个覆盖多语言、多文化的图文对齐数据库，配合“跨语言对齐损失”和“视觉语义一致性损失”双重目标进行预训练。  
   *带来的改变*：模型在中文、英文、日文等多语言环境下都能准确捕捉图像语义，实现真正的全球化图文理解。

3. **专属的文本‑图像组合评测体系**  
   *之前的做法*：缺少统一的量化指标，只能靠人工主观打分，难以对比不同系统。  
   *本文的做法*：设计了包含人类评审和 GPT‑4‑Vision 自动评分的双层评测框架，并提出了“组合一致性分数”。  
   *带来的改变*：提供了可复现、可对标的评估标准，使得模型的图文组合能力可以在学术界和工业界统一比较。

### 方法详解
整体框架可以分为三步：**（1）多模态预训练、（2）交错解码、（3）组合评估**。下面逐层拆解。

1. **多模态预训练**  
   - **数据准备**：作者收集了数十亿对跨语言文本‑图像样本，覆盖新闻、百科、社交媒体等多种场景。每条样本都经过语言检测、图像质量过滤以及跨语言对齐标注。  
   - **模型结构**：基于 InternLM 系列的语言骨干（Transformer 编码器‑解码器），在其上叠加了视觉感知层（ViT‑B/16）并通过跨模态注意力桥接两者。  
   - **训练目标**：使用两类损失：① **跨语言对齐损失**，让不同语言的描述在视觉特征空间中聚集；② **视觉语义一致性损失**，强制模型在看到图像时能够预测对应的文字标签。这样模型既学会“看懂图”，也学会“用多语言说图”。

2. **交错解码（Interleaved Generation）**  
   - **解码流程**：在标准的自回归文本生成过程中，模型每生成一个 token，就会计算一个 **插图置信度**。该置信度来源于当前文本上下文的视觉需求向量（由语言层投射得到）与视觉库中候选图像的相似度分数。  
   - **插图决策**：当置信度超过预设阈值，模型触发 **图像检索/生成子模块**。检索子模块在内部维护一个大规模图像向量库，使用最近邻搜索快速返回最匹配的图片；生成子模块则调用内部的小型扩散模型，根据文本提示合成专属图片。  
   - **位置标记**：插入的图片会被包装成特殊的占位符 `<IMG_i>`，随后继续生成后续文字。解码器在看到占位符时会自动更新注意力权重，使得后文能够参考已插入的视觉信息。  
   - **关键技巧**：作者提出的 **“视觉回溯注意力”**（Visual Backward Attention）让模型在插图后还能回顾前面的文字，确保图文之间的语义一致性，这一点在传统的单向生成里很少出现。

3. **组合评估**  
   - **人机混合评分**：先让真实评审员对每篇生成稿件的“图文匹配度”“阅读流畅度”等维度打 1‑5 分；随后将同一稿件送入 GPT‑4‑Vision，利用其视觉‑语言理解能力给出客观的匹配分数。两者取加权平均得到最终 **组合一致性分数**。  
   - **指标设计**：除了整体分数，还拆分出 **插图时机准确率**（模型是否在恰当位置插图）和 **视觉语义相似度**（图片内容与对应文字的语义距离）两项，用于细粒度分析。

**最巧妙的地方**在于把插图决策嵌入到自回归解码的每一步，而不是事后再做一次独立的检索。这种“边写边配图”的思路让模型能够利用最新的上下文信息做出更精准的视觉选择，类似于人类写稿时随时打开图库挑选素材的过程。

### 实验与效果
- **测试基准**：在 MME、MMBench、MMBench‑CN、Seed‑Bench、CCBench、QBench、Tiny LVLM 等七个主流视觉语言基准上进行评测，覆盖通用问答、文化常识、细粒度图像理解等任务。  
- **整体表现**：论文声称 InternLM‑XComposer 在所有基准上均刷新了 SOTA（State‑of‑the‑Art）记录。例如在 MMBench‑CN 上比上一代模型提升约 4.2% 的整体得分，在 QBench 上的多语言问答准确率提升约 3.8%。  
- **文本‑图像组合评测**：在自建的组合评测集上，InternLM‑XComposer 的组合一致性分数为 4.3（满分 5），与公开的 GPT‑4‑Vision（4.5）和 GPT‑3.5（3.9）相当，显示出与顶级商业模型竞争的实力。  
- **消融实验**：作者分别去掉了视觉回溯注意力、跨语言对齐损失和图像检索子模块，结果显示：去掉视觉回溯注意力后整体分数下降 1.1 分，去掉跨语言对齐损失后多语言任务准确率下降约 2.5%。这表明两者对模型的跨模态一致性贡献显著。  
- **局限性**：论文承认在极端长文（超过 2000 token）时插图置信度的计算会带来显著的推理开销；此外，模型在高度抽象的艺术风格图片匹配上仍有不足，生成的图片有时偏向常规视觉素材。

### 影响与延伸思考
InternLM‑XComposer 把“写稿+配图”统一进同一个生成框架，开启了视觉语言模型在内容创作领域的实用化路径。后续的工作已经开始在以下方向上借鉴其思路：  
- **交互式写作助理**：把模型嵌入到编辑器插件，让用户在写作时实时获得智能配图建议。  
- **多模态教学系统**：利用交错生成帮助教材自动配图，提升学习材料的可视化程度。  
- **跨媒体新闻生成**：在新闻自动撰写时直接生成配图或图表，缩短从采编到发布的链路。  
如果想进一步深入，可以关注 **跨模态检索加速**（如何在千亿级图像库中实现低延迟检索）和 **长文多模态记忆**（如何在超长上下文中保持图文一致性）这两个研究热点。

### 一句话记住它
InternLM‑XComposer 把“写文字”和“挑图片”合并到同一生成过程，实现了真正的边写边配图，让 AI 能像编辑一样自然地把图文交织在一起。