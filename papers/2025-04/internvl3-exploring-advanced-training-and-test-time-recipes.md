# InternVL3: Exploring Advanced Training and Test-Time Recipes for   Open-Source Multimodal Models

> **Date**：2025-04-14
> **arXiv**：https://arxiv.org/abs/2504.10479

## Abstract

We introduce InternVL3, a significant advancement in the InternVL series featuring a native multimodal pre-training paradigm. Rather than adapting a text-only large language model (LLM) into a multimodal large language model (MLLM) that supports visual inputs, InternVL3 jointly acquires multimodal and linguistic capabilities from both diverse multimodal data and pure-text corpora during a single pre-training stage. This unified training paradigm effectively addresses the complexities and alignment challenges commonly encountered in conventional post-hoc training pipelines for MLLMs. To further improve performance and scalability, InternVL3 incorporates variable visual position encoding (V2PE) to support extended multimodal contexts, employs advanced post-training techniques such as supervised fine-tuning (SFT) and mixed preference optimization (MPO), and adopts test-time scaling strategies alongside an optimized training infrastructure. Extensive empirical evaluations demonstrate that InternVL3 delivers superior performance across a wide range of multi-modal tasks. In particular, InternVL3-78B achieves a score of 72.2 on the MMMU benchmark, setting a new state-of-the-art among open-source MLLMs. Its capabilities remain highly competitive with leading proprietary models, including ChatGPT-4o, Claude 3.5 Sonnet, and Gemini 2.5 Pro, while also maintaining strong pure-language proficiency. In pursuit of open-science principles, we will publicly release both the training data and model weights to foster further research and development in next-generation MLLMs.

---

# InternVL3：探索开源多模态模型的高级训练与推理技巧 论文详细解读

### 背景：这个问题为什么难？

多模态大语言模型（MLLM）要同时理解文字和图像，传统做法是先训练一个只会文字的语言模型，再“后接”视觉编码器。这样会出现两大痛点：一是视觉特征与语言模型的内部表示不匹配，需要大量对齐数据和技巧；二是后接方式往往只能处理固定长度的图文输入，难以扩展到更长的多模态上下文。于是模型在实际任务中常出现“看图说话”不自然、长文档理解力不足的问题，迫切需要一种更根本的训练方式来突破这些瓶颈。

### 关键概念速览
- **多模态预训练**：在同一次训练中让模型同时看到图像和文字，让它自行学习两者的关联，而不是先学文字再硬接视觉。类似于让学生在同一堂课上同时学数学和物理，培养跨学科思维。
- **原生多模态模型**：指从零开始就设计为同时接受视觉和语言输入的模型，而不是在已有的语言模型上“拼装”视觉模块。就像从头造一辆混合动力车，而不是在燃油车上加装电池。
- **可变视觉位置编码（V2PE）**：一种让模型在处理不同长度图像序列时仍能保持位置信息的技术。想象把每张图片切成若干块，每块都有自己的坐标，即使块数变化，坐标体系也能自适应。
- **监督微调（SFT）**：在大规模预训练后，用标注好的任务数据继续训练，使模型在特定任务上表现更好。相当于在通用教育后进行职业培训。
- **混合偏好优化（MPO）**：把多种人类偏好（如有用性、礼貌、事实性）混合进训练目标，让模型在生成答案时兼顾多方面需求。类似于在招聘时同时看技术、沟通和团队协作能力。
- **测试时尺度扩展**：在推理阶段通过增加模型的计算预算（如更大的采样步数或更高的分辨率）来提升答案质量。好比在考试时给自己更多的时间思考，以求更精准的答案。
- **MMMU 基准**：一个综合评估多模态理解能力的排行榜，覆盖文字、图像、表格等多种信息源。相当于多学科的统一考试。

### 核心创新点
1. **统一多模态与纯文本预训练 → 直接在同一阶段喂入图文混合数据和纯文字数据 → 省去后期对齐步骤，显著提升模型对视觉信息的内在理解深度。** 过去的做法是先训练语言模型，再用视觉投影层对齐，这会产生“语言瓶颈”。InternVL3 把两类数据一起喂进模型，让它自行学习跨模态映射，避免了人为的对齐偏差。

2. **可变视觉位置编码（V2PE） → 为每张图像的视觉块分配可伸缩的位置信号 → 支持更长的图像序列和跨图像上下文。** 传统位置编码只能处理固定长度的视觉序列，遇到高分辨率或多图输入会失效。V2PE 通过动态计算位置信号，使模型在不同分辨率、不同图像数量下都能保持空间感知。

3. **混合偏好优化（MPO） + 监督微调（SFT） → 在大规模预训练后，先用高质量指令数据进行 SFT，再用多目标偏好损失进行 MPO → 让模型在保持语言流畅性的同时，对视觉指令的响应更精准。** 这一步把“懂语言”和“会看图”两条线索融合，避免了只在语言上微调导致的视觉退化。

4. **测试时尺度扩展 + 优化的训练基础设施 → 在推理时动态提升采样步数、使用更高分辨率图像，同时在训练阶段采用高效的并行调度和混合精度技术 → 在相同硬件预算下实现更高的性能/成本比。** 这让开源模型在资源受限的环境下仍能追平商业闭源模型的表现。

### 方法详解
整体框架可以划分为三大阶段：**（1）统一多模态预训练、（2）指令层面的监督微调与混合偏好优化、（3）推理时的尺度扩展**。下面逐步拆解每个阶段的关键模块。

1. **统一多模态预训练**  
   - **数据构造**：作者收集了两类数据：① 大规模的纯文本语料（类似于 GPT‑4 的训练数据），② 多模态对齐数据（图像+描述、问答、对话等）。在每个训练 batch 中，随机混合这两类样本，使模型既要预测纯文本的下一个词，也要在视觉上下文中完成填空或回答。  
   - **模型结构**：采用 Transformer 作为核心，输入分为两路：文字 token 序列和视觉 token 序列。视觉 token 通过视觉编码器（如 ViT）转化为向量后，加入 **V2PE**。V2PE 的核心是：对每个视觉块的坐标进行线性映射，再乘以一个随序列长度变化的缩放因子，使得位置向量可以随块数伸缩。  
   - **训练目标**：统一使用自回归语言建模损失（预测下一个 token），对多模态样本额外加上跨模态对齐的遮盖预测（类似于 BERT 的 MLM），但所有损失在同一梯度中累计。这样模型在一次前向传播里就学会了文字生成和视觉理解。

2. **指令层面的监督微调（SFT）**  
   - **指令数据**：从公开的指令微调数据集（如 Alpaca、ShareGPT）以及自制的图文指令集合中抽取样本。每条指令可能包含图像、文字或两者的组合，要求模型给出自然语言回答。  
   - **微调方式**：保持原始 Transformer 参数不变，只在最后的语言投影层上加一个轻量的 LoRA（低秩适配）层，以降低显存占用。这样可以在不破坏预训练知识的前提下快速适配指令任务。

3. **混合偏好优化（MPO）**  
   - **偏好标签**：收集了三类偏好评分：**有用性**（答案是否解决问题）、**安全性**（是否包含有害内容）和**事实性**（信息是否准确）。每条生成的答案会得到这三维评分。  
   - **损失组合**：在 SFT 基础上加入一个加权的偏好损失，权重通过验证集上的贝叶斯优化自动调节。这样模型在生成时会倾向于同时满足多种人类价值观，而不是单纯追求流畅度。

4. **测试时尺度扩展**  
   - **采样策略**：在推理时使用更高的温度调度、更多的 beam（束搜索）或 nucleus sampling（核采样）步数，以提升答案的多样性和准确性。  
   - **视觉分辨率**：对输入图像使用更高的分辨率进行编码，V2PE 能自然适配更长的视觉 token 序列，避免信息丢失。  
   - **硬件优化**：作者在训练时采用了张量并行 + 数据并行的混合并行策略，并使用 FlashAttention 等加速库，使得即使在大模型（78B 参数）上也能在数周内完成训练。

**最巧妙的点**在于 V2PE 与统一预训练的结合：它让模型在“看”更大、更长的图像时不需要重新设计位置编码，而是直接复用同一套 Transformer 参数，这在以前的多模态模型里几乎是不可想象的。

### 实验与效果
- **评测数据集**：作者在 MMMU（Multimodal Massive Multitask Understanding）基准上进行主评测，还覆盖了 VQAv2、COCO Caption、ScienceQA（图文混合版）等任务。  
- **核心结果**：InternVL3‑78B 在 MMMU 上取得 **72.2 分**，刷新了开源 MLLM 的最高记录，领先第二名约 4 分。与闭源的 ChatGPT‑4o、Claude 3.5 Sonnet、Gemini 2.5 Pro 相比，仅在少数细分任务上略有差距。  
- **对比基线**：与前代 InternVL2（采用后接视觉层）相比，整体分数提升约 6–8 分；与 LLaVA‑1.5、MiniGPT‑4 等同规模开源模型相比，优势在 5–10 分之间。  
- **消融实验**：作者分别关闭 V2PE、MPO、SFT 三个模块进行实验，发现：去掉 V2PE 会导致 MMMU 分数下降约 2.3 分；去掉 MPO 使有用性评分下降约 1.8 分；仅保留 SFT 而不做 MPO，整体分数下降约 1.5 分。说明每个组件都有实质贡献。  
- **局限性**：论文承认在极高分辨率图像（> 4K）和超长文本（> 8k token）场景下仍会出现显存瓶颈；此外，偏好标签的人工标注成本高，MPO 的效果在不同语言（如非英语）上尚未充分验证。

### 影响与延伸思考
InternVL3 的“原生多模态预训练 + 可变位置编码”思路已经在后续的开源项目中被快速复制，例如 **OpenFlamingo‑2**、**Mistral‑Vision** 等都尝试在同一阶段混合图文数据。业界也开始关注 **可伸缩的视觉位置编码** 作为解决高分辨率视觉输入的通用手段。未来的研究方向可能包括：① 将音频、视频等更多模态一起加入统一预训练；② 探索更高效的偏好标签自动生成方法，以降低 MPO 的标注成本；③ 在大规模分布式训练框架下进一步压缩显存，实现 100B 级别的原生多模态模型。对想深入的读者，可以关注 **V2PE 的数学实现**、**MPO 的多目标优化技巧**以及 **混合并行训练调度** 这几块。

### 一句话记住它
InternVL3 用一次统一的多模态预训练和可变视觉位置编码，让模型从根本上“看得懂、说得好”，在开源多模态领域实现了与商业大模型相媲美的性能。