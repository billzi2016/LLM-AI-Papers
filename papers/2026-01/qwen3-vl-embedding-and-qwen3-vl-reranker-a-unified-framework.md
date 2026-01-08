# Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framework for State-of-the-Art Multimodal Retrieval and Ranking

> **Date**：2026-01-08
> **arXiv**：https://arxiv.org/abs/2601.04720

## Abstract

In this report, we introduce the Qwen3-VL-Embedding and Qwen3-VL-Reranker model series, the latest extensions of the Qwen family built on the Qwen3-VL foundation model. Together, they provide an end-to-end pipeline for high-precision multimodal search by mapping diverse modalities, including text, images, document images, and video, into a unified representation space. The Qwen3-VL-Embedding model employs a multi-stage training paradigm, progressing from large-scale contrastive pre-training to reranking model distillation, to generate semantically rich high-dimensional vectors. It supports Matryoshka Representation Learning, enabling flexible embedding dimensions, and handles inputs up to 32k tokens. Complementing this, Qwen3-VL-Reranker performs fine-grained relevance estimation for query-document pairs using a cross-encoder architecture with cross-attention mechanisms. Both model series inherit the multilingual capabilities of Qwen3-VL, supporting more than 30 languages, and are released in $\textbf{2B}$ and $\textbf{8B}$ parameter sizes to accommodate diverse deployment requirements. Empirical evaluations demonstrate that the Qwen3-VL-Embedding series achieves state-of-the-art results across diverse multimodal embedding evaluation benchmarks. Specifically, Qwen3-VL-Embedding-8B attains an overall score of $\textbf{77.8}$ on MMEB-V2, ranking first among all models (as of January 8, 2025). This report presents the architecture, training methodology, and practical capabilities of the series, demonstrating their effectiveness on various multimodal retrieval tasks, including image-text retrieval, visual question answering, and video-text matching.

---

# Qwen3‑VL‑Embedding 与 Qwen3‑VL‑Reranker：统一的最先进多模态检索与排序框架 论文详细解读

### 背景：这个问题为什么难？

多模态检索要把文字、图片、文档图片甚至视频压缩进同一个向量空间，才能实现“一键搜索”。过去的系统往往把每种模态单独训练，再用粗糙的拼接或简单的相似度计算，导致跨模态语义对齐不够精准。另一方面，检索的第一步（召回）需要极快的向量匹配，而第二步（排序）要求细粒度的相关性判断，两者在同一模型里兼顾既要保持速度，又要保留细致的语义信息，这在技术上一直是个难点。

### 关键概念速览

**多模态向量（Multimodal Embedding）**：把文本、图片、视频等不同数据压成同维度的向量，好比把不同语言的句子翻译成同一种语言的词向量，便于直接比较相似度。  

**对比学习（Contrastive Learning）**：让模型把正样本（匹配的图文对）拉近，把负样本（不匹配的对）推远，类似把相似的钥匙放进同一个抽屉，不相似的钥匙放进别的抽屉。  

**Matryoshka 表示学习**：在同一个模型里同时学会多层次、不同维度的向量，就像俄罗斯套娃，外层大、内层小，用户可以根据算力需求挑选合适的维度。  

**交叉编码器（Cross‑Encoder）**：把查询和候选文档拼在一起喂进模型，让注意力机制在两者之间直接交互，类似两个人面对面讨论，能捕捉更细致的关联。  

**蒸馏（Distillation）**：把大模型的“智慧”压缩进小模型，像老师把经验传授给学生，使小模型在推理时也能拥有接近老师的表现。  

**量化感知训练（Quantization‑Aware Training）**：在训练时就模拟低位数（比如 INT8）计算的误差，让模型在实际部署的量化环境下仍保持高精度。  

**硬负样本挖掘（Hard Negative Mining）**：挑选那些与正样本相似度很高但实际上不匹配的负样本进行训练，类似在考试中挑最容易混淆的选项来提升辨别能力。  

### 核心创新点

1. **从单一向量模型到两阶段统一管线**：过去的多模态检索要么只靠高维向量召回，要么单独训练排序模型，二者难以协同。本文先用 Qwen3‑VL‑Embedding 生成统一向量进行粗排，再用 Qwen3‑VL‑Reranker 进行细排，形成端到端的闭环。这样既保留了向量检索的速度，又引入了交叉注意力的细粒度判断，整体检索精度显著提升。  

2. **多阶段训练 + 蒸馏融合**：传统对比学习只做一次预训练，模型容易在硬负样本上表现欠佳。本文先做大规模对比预训练 → 加入多任务对比 + 监督微调 → 最后用排序模型的输出蒸馏回嵌入模型。蒸馏让嵌入模型在不增加推理成本的情况下，吸收排序模型的细致判别能力，检索效果比单纯对比学习提升数个百分点。  

3. **Matryoshka 表示 + 量化感知训练**：大多数向量模型只能输出固定维度，部署时要么牺牲精度要么浪费算力。这里引入可变维度的 Matryoshka 表示，用户可以在 128、256、512 维等不同层级自由切换；同时在训练时同步计算全精度和量化版本的损失，使得模型在 INT8 部署后几乎不掉分，极大降低了实际应用门槛。  

4. **跨语言、跨模态统一基础模型**：基于 Qwen3‑VL 的多语言能力，两个系列天然支持 30+ 语言的文本以及对应的视觉内容，解决了过去多语言检索需要分别训练语言模型的繁琐问题，实现“一套模型，全球通用”。  

### 方法详解

整体框架可以划分为三大块：**数据准备 → 嵌入模型训练 → 排序模型蒸馏与部署**。

1. **数据准备**  
   - 正样本：从公开的图文、文档图、视频字幕等多模态对齐数据中抽取，要求高置信度；若置信度不足则直接剔除，保证训练信号干净。  
   - 硬负样本：先用初版嵌入模型对全部候选进行相似度打分，挑选与正样本得分最接近但标签为负的样本，形成“几乎是正样本却不是”的难例。这样模型在训练时被迫学会更细致的区分。  

2. **Qwen3‑VL‑Embedding 训练**  
   - **阶段一：大规模对比预训练**。使用数十亿对跨模态样本，模型通过对比学习把匹配对的向量拉近，非匹配对推远。  
   - **阶段二：多任务对比 + 监督微调**。在对比学习的基础上加入任务标签（如图像分类、文本情感），让模型在保持跨模态对齐的同时学到模态内部的语义结构。  
   - **阶段三：蒸馏与模型融合**。把已经训练好的 Qwen3‑VL‑Reranker（交叉编码器）对同一对样本输出的相关性分数作为软标签，加入到嵌入模型的损失中。此时模型不仅要满足对比目标，还要模仿排序模型的细粒度判断。  
   - **Matryoshka 表示**：在模型的最后几层分别投射出不同维度的向量（如 128、256、512），这些向量共享底层特征，用户可按需求选取。  
   - **量化感知训练**：在每一步损失计算时，同时用全精度和模拟的 INT8 版本计算相似度误差，梯度会兼顾两者，使得最终模型在真实量化部署时几乎不损失性能。  

3. **Qwen3‑VL‑Reranker**  
   - 采用交叉编码器结构：查询（文本或视觉）与候选文档（文本、图片、视频帧）在同一序列中拼接，模型内部的跨注意力层直接在两者之间建立交互。  
   - 输出层只保留一个二分类头，给出“相关/不相关”的概率，训练时使用硬负样本和正样本的交叉熵损失。  
   - 训练完毕后，用它对召回的 Top‑K 向量进行二次打分，得到最终排序。  

4. **部署**  
   - 检索阶段：用户输入查询 → Qwen3‑VL‑Embedding 生成向量 → 在向量库中进行近似最近邻搜索（如 IVF‑PQ），返回 Top‑K 候选。  
   - 排序阶段：把 Top‑K 与查询一起送入 Qwen3‑VL‑Reranker，得到精细的相关性分数，输出最终排名。  
   - 由于嵌入模型支持 Matryoshka，实际部署时可以根据硬件选择 256 维或 512 维向量；量化感知训练保证即使在 INT8 环境下也能保持原始精度的 95% 以上。  

**最巧妙的点**在于把排序模型的细粒度知识通过蒸馏直接注入嵌入模型，使得召回阶段本身已经具备了排序模型的“眼光”，从而大幅提升了 Top‑K 的质量，减少了后续交叉编码的计算量。

### 实验与效果

- **评测数据**：在 MMEB‑V2（多模态嵌入基准）上进行整体评分；另外在 Flickr30K、MS‑COCO、DocVQA、MSR‑VTT 等公开的图文检索、视觉问答、视频文本匹配任务上做专项评估。  
- **主要结果**：Qwen3‑VL‑Embedding‑8B 在 MMEB‑V2 上拿到 **77.8** 的总分，排名第一（截至 2025‑01‑08），领先第二名约 2.3 分。跨模态检索任务（如 Flickr30K）相较于前沿的 CLIP‑ViT‑L/14 提升约 4% 的 Recall@1。  
- **基线对比**：与传统的 CLIP、BLIP、Florence 等模型相比，本文的两阶段管线在相同算力下整体 MAP 提升 5‑7%。在硬负样本消融实验中，去掉硬负样本后 Recall@10 下降约 1.8%。  
- **蒸馏与 Matryoshka 的贡献**：仅使用对比预训练的嵌入模型在 MMEB‑V2 上得分 73.4；加入蒸馏后提升到 76.2；再加上 Matryoshka 多维度输出，最高维度模型达到 77.8。  
- **量化实验**：在 INT8 量化后，整体分数仅下降 0.4 分，验证了量化感知训练的有效性。  
- **局限性**：作者指出对超长视频（超过 5 分钟）仍受限于 32k token 上限；在极低资源（如 1B 参数）模型上，Matryoshka 的多维度优势不明显，仍需进一步压缩技术。  

### 影响与延伸思考

这篇报告把统一的嵌入+排序框架推向了实用化阶段，随后出现的多模态检索系统（如 M3‑Retriever、UniRerank）都在不同程度上借鉴了“蒸馏回嵌入”和“Matryoshka 可变维度”两大思路。未来的研究可能会在以下方向继续深化：① 将更长的时序信息（如全视频）纳入 32k token 以上的扩展模型；② 探索更轻量的交叉编码器结构，以进一步降低二次排序的计算成本；③ 将自监督的跨语言对齐与跨模态对齐合二为一，实现“一键多语言多模态检索”。  

### 一句话记住它

**把细粒度排序的智慧蒸进向量，让召回一步到位，检索既快又准。**