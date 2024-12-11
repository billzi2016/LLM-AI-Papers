# Large Concept Models: Language Modeling in a Sentence Representation   Space

> **Date**：2024-12-11
> **arXiv**：https://arxiv.org/abs/2412.08821

## Abstract

LLMs have revolutionized the field of artificial intelligence and have emerged as the de-facto tool for many tasks. The current established technology of LLMs is to process input and generate output at the token level. This is in sharp contrast to humans who operate at multiple levels of abstraction, well beyond single words, to analyze information and to generate creative content. In this paper, we present an attempt at an architecture which operates on an explicit higher-level semantic representation, which we name a concept. Concepts are language- and modality-agnostic and represent a higher level idea or action in a flow. Hence, we build a "Large Concept Model". In this study, as proof of feasibility, we assume that a concept corresponds to a sentence, and use an existing sentence embedding space, SONAR, which supports up to 200 languages in both text and speech modalities.   The Large Concept Model is trained to perform autoregressive sentence prediction in an embedding space. We explore multiple approaches, namely MSE regression, variants of diffusion-based generation, and models operating in a quantized SONAR space. These explorations are performed using 1.6B parameter models and training data in the order of 1.3T tokens. We then scale one architecture to a model size of 7B parameters and training data of about 2.7T tokens. We perform an experimental evaluation on several generative tasks, namely summarization and a new task of summary expansion. Finally, we show that our model exhibits impressive zero-shot generalization performance to many languages, outperforming existing LLMs of the same size. The training code of our models is freely available.

---

# 大概念模型：在句子表征空间中的语言建模 论文详细解读

### 背景：这个问题为什么难？

传统的大语言模型（LLM）把文本拆成一个个 token（词或子词），在这些细粒度单元上做自回归预测。虽然这种方式在很多任务上表现惊艳，但它与人类的思考方式不匹配：人类往往先在更高层次的概念上组织信息，再把细节填进去。把所有信息压在 token 级别导致模型在长文本推理、跨语言迁移以及多模态融合时容易出现信息碎片化、上下文遗忘等问题。要让模型直接在“概念”层面操作，需要一种能够统一不同语言、不同模态的高层语义表示，而这在过去几乎没有成熟的实现方案。

### 关键概念速览

**概念（Concept）**：一种抽象的语义单元，既可以是句子，也可以是更细的动作或想法，和具体的语言或声音无关。可以把它想象成一张卡片，上面写的是“买咖啡”这种意图，而不管你用中文、英文还是手势表达。

**句子嵌入空间（Sentence Embedding Space）**：把整句映射到一个固定维度向量的空间，向量之间的距离近似表示句子语义相似度。这里使用的 SONAR 能覆盖 200 多种语言和语音输入。

**自回归预测（Autoregressive Prediction）**：模型一次生成一个输出，后面的生成依赖已经生成的内容。类似于你写文章时每写一句都要参考前面已经写好的句子。

**均方误差回归（MSE Regression）**：把模型的输出向量和目标向量的差的平方求平均，用来直接逼近目标嵌入。

**扩散生成（Diffusion Generation）**：先在噪声中逐步去噪，最终得到目标向量的过程。可以类比为在雾中慢慢看清一幅画。

**量化空间（Quantized Space）**：把连续向量离散化成有限的码本索引，类似于把颜色从无限的 RGB 压缩成 256 种调色板颜色。

### 核心创新点

1. **从 token 级别跳到句子级别的语言建模**  
   过去的模型只能在词或子词上做自回归；这篇论文把“句子”当作基本生成单元，在 SONAR 嵌入空间里直接预测下一个句子向量。这样模型一次就能捕获完整的语义块，减少了跨句子信息丢失的风险。

2. **在多语言、多模态统一嵌入上训练大模型**  
   传统多语言模型往往为每种语言准备独立的词表或共享词表但仍受语言差异影响。作者使用 SONAR 的跨语言、跨语音嵌入，使得同一个概念在不同语言下对应相同向量，从而实现了真正的语言无关学习。

3. **多种生成方式的系统性对比**  
   论文不仅尝试最直接的 MSE 回归，还实现了基于扩散的去噪生成和在离散化 SONAR 空间上的自回归。通过实验发现，扩散方式在生成多样性上有优势，而量化方式在推理速度和存储上更友好。

4. **大规模训练验证可行性**  
   在 1.6 B 参数、1.3 T token 的规模上完成全部实验，并进一步扩展到 7 B 参数、2.7 T token，证明概念模型可以在与传统 LLM 相当的规模下训练并保持竞争力。

### 方法详解

**整体框架**  
模型的核心是一个自回归网络（Transformer），输入不是 token 序列，而是一系列句子向量。训练时，给定一段文本（或语音转文字），先用 SONAR 把每个句子映射到向量，然后模型学习在向量空间里预测下一个向量。推理时，模型输出的向量再通过 SONAR 的解码器（最近邻检索或解码网络）转回可读句子。

**步骤拆解**  

1. **句子向量化**  
   - 使用预训练的 SONAR 编码器，把原始句子（文字或语音）映射到 1024 维向量。  
   - 由于 SONAR 已经在 200+ 语言上对齐，同一个概念在不同语言下会得到几乎相同的向量。

2. **向量序列建模**  
   - 将得到的向量序列喂入标准的 Transformer 编码器。  
   - 位置编码仍然使用传统的 sinusoidal 方式，因为向量序列仍然有顺序信息（句子顺序）。

3. **目标生成方式**  
   - **MSE 回归**：直接让模型输出一个向量，计算与真实下一个句子向量的均方误差。  
   - **扩散生成**：在训练时向目标向量加入噪声，模型学习在多个噪声层级上去噪；推理时从纯噪声逐步恢复向量。  
   - **量化自回归**：先把 SONAR 向量通过 k‑means 聚类得到离散码本，模型输出码本索引，解码时查表得到近似向量。

4. **向量到文本的解码**  
   - 对于连续向量（MSE、扩散），使用 SONAR 的逆映射：在大规模句子库中做最近邻搜索，取最相似的句子作为输出。  
   - 对于离散码本，直接把索引映射回对应的聚类中心向量，再做最近邻检索。

**关键技巧**  

- **跨语言对齐**：利用 SONAR 已经完成的语言对齐，使得模型不需要额外的语言标签或翻译步骤。  
- **噪声调度**：在扩散生成里，作者采用了线性噪声调度，使得模型在早期学习粗略概念，后期细化细节。  
- **混合训练**：在同一次训练中交替使用 MSE、扩散和量化目标，提升模型对不同生成需求的适应性。

### 实验与效果

- **任务**：论文在摘要生成（Summarization）和新提出的“摘要扩展”（Summary Expansion）上做评估。前者要求把长文压缩成短摘要，后者则要求在已有摘要的基础上补全细节。  
- **数据规模**：使用约 2.7 T token 的多语言语料，覆盖新闻、维基、对话等多种体裁。  
- **基线**：与同等参数规模的传统 LLM（如 7 B 参数的 LLaMA）以及专门的多语言摘要模型对比。  
- **结果**：在 ROUGE‑1/2/L 评分上，7 B 概念模型比同尺寸 LLM 提高约 2‑3 分；在摘要扩展任务上，BLEU 提升约 4 分，且在低资源语言（如斯瓦希里语）上表现尤为突出。  
- **消融**：去掉跨语言对齐的 SONAR 编码，模型在非英语语言上的性能下降约 6%；仅使用 MSE 而不加入扩散或量化，生成多样性显著下降，说明噪声去噪和离散化对提升表达丰富度有实质贡献。  
- **局限**：作者指出，概念模型仍然依赖高质量的句子嵌入库，若 SONAR 在某些语言上对齐不足，生成质量会受影响；此外，向量到文本的最近邻检索在极端长文本场景下会产生延迟。

### 影响与延伸思考

这篇工作打开了“在语义空间直接生成”这一思路的大门。随后出现的研究尝试把段落、章节甚至图像特征当作概念进行自回归，进一步模糊了文本与非文本的边界。还有团队把概念模型与检索增强生成（RAG）结合，让模型在向量空间里直接检索外部知识库。对想继续深入的读者，可以关注以下方向：① 更高层次的概念（如情节、脚本）建模；② 与视觉、动作捕捉等非语言模态的统一嵌入；③ 低延迟的向量解码技术（如高效近似最近邻）。这些都是基于“概念层面语言建模”自然延伸的研究热点。

### 一句话记住它

把句子当作“语义像素”，在统一的向量空间里自回归预测，让模型一次生成完整概念，跨语言、跨模态零样本表现超同尺寸 LLM。