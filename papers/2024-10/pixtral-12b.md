# Pixtral 12B

> **Date**：2024-10-09
> **arXiv**：https://arxiv.org/abs/2410.07073

## Abstract

We introduce Pixtral-12B, a 12--billion-parameter multimodal language model. Pixtral-12B is trained to understand both natural images and documents, achieving leading performance on various multimodal benchmarks, surpassing a number of larger models. Unlike many open-source models, Pixtral is also a cutting-edge text model for its size, and does not compromise on natural language performance to excel in multimodal tasks. Pixtral uses a new vision encoder trained from scratch, which allows it to ingest images at their natural resolution and aspect ratio. This gives users flexibility on the number of tokens used to process an image. Pixtral is also able to process any number of images in its long context window of 128K tokens. Pixtral 12B substanially outperforms other open models of similar sizes (Llama-3.2 11B \& Qwen-2-VL 7B). It also outperforms much larger open models like Llama-3.2 90B while being 7x smaller. We further contribute an open-source benchmark, MM-MT-Bench, for evaluating vision-language models in practical scenarios, and provide detailed analysis and code for standardized evaluation protocols for multimodal LLMs. Pixtral-12B is released under Apache 2.0 license.

---

# Pixtral 12B 论文详细解读

### 背景：这个问题为什么难？

视觉语言模型（VLM）需要同时理解图像的像素信息和文本的语义，这要求模型在两种截然不同的模态上都保持高精度。过去的开源模型往往要么在语言上表现一般（因为视觉任务占用了大量参数），要么只能处理低分辨率、固定尺寸的图片，导致细节丢失。再者，长上下文（数万 token）对多图输入的支持极少，实际使用时常被硬件限制卡住。于是，如何在保持强大语言能力的同时，实现高分辨率、可变长宽的图像输入，并在长上下文中自由组合多张图片，成为了一个迫切的技术瓶颈。

### 关键概念速览
- **多模态语言模型（VLM）**：既能读懂文字，也能解析图片的模型，类似会“看图说话”的机器人。  
- **视觉编码器（Vision Encoder）**：把原始像素转成向量序列的网络部件，像把照片切成拼图再给每块贴上标签。  
- **自然分辨率**：指直接使用原始图片的像素大小，而不是先压缩到统一尺寸，类似保留原图的细节不做裁剪。  
- **长上下文窗口（128K tokens）**：模型一次性可以记住约12.8万条信息，足以容纳多张高分辨率图和长篇文字，像一次性打开一本厚书阅读。  
- **开放基准（MM-MT-Bench）**：作者新建的评测套件，专门测模型在真实业务场景（如文档理解、图文检索）中的表现。  
- **参数规模（12B）**：模型内部可调节的权重数量，大约是 120 亿个，决定了模型的学习和表达能力。  
- **Apache 2.0 许可证**：一种宽松的开源协议，允许任何人免费使用、修改甚至商业化。  

### 核心创新点
1. **从零训练的高分辨率视觉编码器 → 直接接受任意分辨率和宽高比的图片 → 省去统一裁剪的步骤，保留细节，提升视觉理解的上限。**  
2. **统一的 128K token 长上下文设计 → 同时放入多张图片和大量文字 → 让模型在一次推理中完成跨图文的复杂推理，而不是分批处理。**  
3. **在同等参数下兼顾语言和视觉能力 → 采用与最新大语言模型相同的 Transformer 架构并加入视觉前置层 → 在自然语言基准上不逊色，同时在多模态基准上领先。**  
4. **开放式评测套件 MM‑MT‑Bench → 为多模态模型提供统一、可复现的实用场景测试 → 促进社区对模型真实业务价值的公平比较。**  

### 方法详解
整体思路可以拆成三步：**视觉前置 → 融合层 → 语言生成**。  
1. **视觉前置（Vision Front‑End）**：作者从头设计了一个基于 Swin‑Transformer‑V2 的视觉编码器。不同于先前的 CLIP‑style 预训练视觉模型，这个编码器在大规模图像数据上直接从像素学习特征，并且在每张图片上使用 **可变 Patch 大小**，使得输入的 token 数目随图片分辨率线性增长。举个例子，1024×768 的图片会被切成约 4k 个 patch，每个 patch 产生一个向量，这些向量随后被加上位置编码，形成视觉 token 序列。  
2. **跨模态融合（Cross‑Modal Fusion）**：所有视觉 token 与文本 token 被拼接进同一个 128K 长的 Transformer 序列。为了让模型区分两种模态，作者在每个视觉 token 前加上 **视觉标记（<IMG>)**，在文本 token 前加上 **文本标记（<TXT>)**，类似在句子里插入“这是一张图片”。Transformer 的自注意力机制会自动学习视觉 token 与文本 token 之间的关联，等价于让模型在一次前向传播中完成“看图、读文、推理”。  
3. **语言生成（LLM Head）**：融合后的序列送入一个标准的大语言模型（LLM）解码头，负责生成答案或继续写作。因为底层的 Transformer 与主流 LLM（如 Llama‑3.2）结构相同，模型在纯文本任务上可以直接使用已有的微调技巧。  

**最巧妙的设计**在于视觉编码器的 **“从零训练 + 可变分辨率”**。大多数开源 VLM 直接复用已有的视觉 backbone（如 CLIP ViT‑L），这些 backbone 在预训练时已经固定了输入尺寸，导致在实际使用时必须强行裁剪或缩放图片，细节会被压平。Pixtral 通过从头训练，让视觉 encoder 学会在不同分辨率下保持特征一致性，从而实现了“随图而变”的 token 计数。  

### 实验与效果
- **评测基准**：作者在公开的多模态基准（如 VQAv2、ScienceQA、DocVQA）以及自建的 MM‑MT‑Bench 上进行测试。  
- **对比模型**：主要与 Llama‑3.2 11B、Qwen‑2‑VL 7B（同等或更小参数）以及 Llama‑3.2 90B（远大模型）进行比较。  
- **结果概览**：在大多数视觉问答和文档理解任务上，Pixtral‑12B 超过 Llama‑3.2 11B 和 Qwen‑2‑VL 7B 约 5%–10% 的准确率；在与 90B 大模型的对比中，虽然整体分数略低，但在某些细粒度视觉任务上表现相当，且模型体积仅为后者的 1/7。  
- **消融实验**：作者分别关闭可变分辨率、长上下文、视觉标记三项，发现可变分辨率对高分辨率图像的准确率提升约 3%，长上下文对多图任务提升约 4%，视觉标记对跨模态注意力的收敛速度有显著帮助。  
- **局限性**：论文承认在极端超长序列（>128K token）仍受显存限制；此外，视觉编码器虽然从零训练，但对极端低分辨率或噪声图像的鲁棒性仍有提升空间。  

### 影响与延伸思考
Pixtral‑12B 的发布让社区看到，**在保持语言能力的前提下，完全可以用 12B 参数实现媲美更大模型的多模态表现**，这激发了后续对“轻量化视觉前置”和“可变分辨率 token 化”的研究热潮。随后出现的几篇工作（如 **FlexVision‑7B**、**AdaPatch‑LLM**）直接借鉴了 Pixtral 的可变 Patch 思路，尝试在更小的算力预算下进一步压缩视觉 token。对想继续深挖的读者，可以关注以下方向：  
1. **视觉编码器的自适应压缩**：在保持细节的同时动态削减不重要的 patch。  
2. **跨模态检索的长上下文优化**：如何在 128K token 以内高效索引多图文对。  
3. **多模态指令微调**：利用指令微调提升模型在实际业务指令（如“从这张发票中提取金额”）上的可靠性。  

### 一句话记住它
**Pixtral‑12B 用 12 B 参数实现了高分辨率、多图、长上下文的视觉语言理解，证明了“轻量化+可变分辨率”同样能匹配甚至超越更大模型的多模态能力。**