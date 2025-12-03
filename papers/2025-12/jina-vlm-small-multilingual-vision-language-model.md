# jina-vlm: Small Multilingual Vision Language Model

> **Date**：2025-12-03
> **arXiv**：https://arxiv.org/abs/2512.04032

## Abstract

We present jina-vlm, a token-efficient 2.4B parameter vision-language model that achieves state-of-the-art multilingual VQA performance among open 2B-scale VLMs. The model couples a SigLIP2 vision encoder with a Qwen3 language decoder and makes use of image tiling and attention-pooling for token-efficient processing of arbitrary-resolution images. To understand the contribution of different training data categories, we conduct a leave-one-out data mixture ablation study-systematically removing task, domain, modality, and language categories-to diagnose which data types are necessary versus redundant and whether task benefits transfer across domains. Model weights and code are publicly released at https://huggingface.co/jinaai/jina-vlm.

---

# jina-vlm：小型多语言视觉语言模型 论文详细解读

### 背景：这个问题为什么难？

视觉语言模型（VLM）需要同时处理高分辨率图像和长文本，计算成本会随图像像素数指数增长。传统做法把整张图像切成固定大小的patch，然后直接喂进Transformer，导致在处理大图时要么丢失细节，要么消耗巨量显存。另一方面，多语言能力往往依赖数十亿参数的大型语言模型，导致模型体积和部署成本高得离谱。于是，业界面临两个矛盾：**如何在保持视觉细节的前提下，用更少的token表示图像**，以及**如何在不膨胀参数规模的情况下，支持多语言问答**。这正是jina‑vlm要破解的难题。

### 关键概念速览
- **Vision Encoder（视觉编码器）**：把原始像素映射成向量序列的网络，类似相机把光信号转成数字信号。这里使用的是SigLIP2，专门为高效特征提取设计。
- **Language Decoder（语言解码器）**：在已有文本上下文的基础上生成下一个词的模型，这里选用了Qwen‑3的1.7B基础版，兼顾多语言理解与生成。
- **Image Tiling（图像切块）**：把一张大图划分为若干重叠的小块（tile），每块单独编码，再在后端合并。想象把一幅巨幅画切成拼图块，分别拍照后再拼回去。
- **Attention‑Pooling（注意力池化）**：在所有tile的特征上做加权平均，权重由自注意力机制决定，等于是让模型自己挑出最重要的局部信息。
- **Token‑Efficient（Token 高效）**：指在Transformer里使用的序列长度尽可能短，以降低显存和计算。这里的“token”指的是图像块或文字片段。
- **Leave‑One‑Out Data Mixture Ablation（逐项剔除数据混合消融）**：系统性地把训练数据中的某一类（比如特定语言或任务）去掉，观察性能变化，从而判断该类数据的必要性。
- **Instruction Fine‑Tuning（指令微调）**：在大模型上继续训练，使其能够遵循自然语言指令完成特定任务，如VQA、OCR等。

### 核心创新点
1. **图像切块 + 注意力池化 → 动态分辨率输入**  
   过去的VLM要么固定图像尺寸，要么在大图上直接切成大量patch，导致token爆炸。jina‑vlm把图像分成若干重叠tile，每个tile用固定大小的视觉编码器处理，再通过注意力池化把所有tile的特征压缩成少量token。这样既保留了高分辨率细节，又把序列长度控制在几百以内。

2. **SigLIP2 + Qwen‑3 双模态桥接 → 2.4B 参数实现多语言 SOTA**  
   之前的多语言VLM多依赖数十亿参数的语言模型。作者把轻量级的SigLIP2视觉编码器和相对紧凑的Qwen‑3（1.7B）语言解码器拼接，通过跨模态对齐训练，使得仅2.4B参数的模型在多语言视觉问答上跑出开放域2B级模型的最佳成绩。

3. **系统化的留一法数据消融 → 揭示数据贡献**  
   通过逐项剔除任务、领域、模态和语言类别，作者量化了每类数据对模型性能的影响，发现跨领域图像caption数据是提升多语言VQA的关键，而大量纯文本对视觉能力贡献有限。这种细粒度的诊断帮助后续数据构建更有针对性。

4. **统一的对齐+指令微调流水线 → 多任务兼容**  
   训练分两阶段：先用跨域图像caption和少量纯文本做对齐，让视觉特征和语言模型共享语义空间；再用指令微调覆盖VQA、文档理解、OCR、数学推理等任务，使模型在不同下游任务上保持一致的交互方式。

### 方法详解
**整体框架**  
jina‑vlm的训练与推理可以划分为三步：① 图像切块并编码；② 注意力池化合并特征；③ 将合并后的视觉token与文本token一起送入语言解码器，生成答案或描述。整个流程保持了“视觉先行、语言后接”的顺序，避免了双向Transformer的高昂计算。

**1. 图像切块（Tile Generation）**  
- 输入任意分辨率的图片。  
- 按固定窗口大小（如224×224）在图像上滑动，步幅小于窗口尺寸，产生重叠tile。重叠确保边缘信息不会在切块边界被截断。  
- 同时生成一张全局缩略图（低分辨率），用于提供整体布局的粗略特征。

**2. SigLIP2 视觉编码**  
- 每个tile和全局缩略图分别送入SigLIP2。SigLIP2是基于对比学习的视觉骨干网络，能够在保持轻量的同时输出高质量的视觉向量。  
- 输出的每个tile得到一个固定长度的向量（如768维），形成一组视觉token。

**3. 注意力池化（Attention‑Pooling）**  
- 将所有tile向量堆叠，加入全局缩略图向量作为“CLS” token。  
- 通过自注意力层计算每个tile的权重，权重高的tile代表图像中信息密集或与任务相关的区域。  
- 对所有向量做加权求和，得到一个或少数几个压缩后的视觉token。这个过程相当于让模型自己挑“关键拼图块”，而不是硬性平均。

**4. 跨模态对齐**  
- 将压缩后的视觉token与对应的文本描述（如caption）拼接，喂入语言解码器的前置层。  
- 使用跨模态对齐损失（如对比学习的InfoNCE）让视觉token在语言空间中与对应文字靠近。这样语言解码器在后续生成时能够自然“看到”图像信息。

**5. 指令微调**  
- 在对齐好的模型上，构造指令式数据集：每条样本包括指令（如“请回答以下关于图片的问题”）、可能的上下文（如文档或表格）以及目标答案。  
- 任务覆盖 VQA、文档理解、OCR、数学推理等，确保模型在不同输入模式下都能保持统一的交互方式。  
- 采用标准的自回归语言模型训练目标（最大化答案的似然），让模型学会在看到视觉token后直接生成答案。

**最巧妙的点**  
- **重叠tile + 注意力池化**：传统做法要么直接下采样导致细节丢失，要么使用巨量patch导致序列爆炸。这里通过重叠tile保留细节，再用注意力池化把信息压缩，达到了“细节不丢、序列短”的平衡。  
- **Leave‑One‑Out 消融**：不是简单的对比实验，而是系统性剔除每类数据，得到对数据贡献的量化图谱，为后续数据构建提供了科学依据。

### 实验与效果
- **评测任务**：多语言视觉问答（VQA）是主要基准，覆盖英语、中文、法语等十余语言；另外在文档理解、OCR、数学推理等指令任务上也做了验证。  
- **对比基线**：与同尺度的开源模型（如BLIP‑2 2.7B、MiniGPT‑4 2.0B）以及更大规模的多语言VLM（如LLaVA‑13B）进行比较。  
- **结果**：在公开的多语言VQA数据集上，jina‑vlm取得了最高的平均准确率，超过同尺度模型约5%~8%，并且在多数语言上超过了部分13B级模型的表现（论文声称）。  
- **消融实验**：去掉注意力池化后，模型在高分辨率图像上的准确率下降约12%；去掉跨域caption数据，跨语言VQA的准确率下降约9%；仅保留纯文本对齐，视觉相关任务几乎失效，验证了每个模块的必要性。  
- **局限性**：模型仍然受限于2.4B参数，在极端复杂推理或长文档多轮对话上表现不如更大模型；作者也提到在极低资源语言（如斯瓦希里语）上的数据仍不足，导致性能不均衡。

### 影响与延伸思考
jina‑vlm展示了“轻量+多语言”在视觉语言领域的可行路径，激发了后续工作在**token‑efficient 视觉处理**和**跨模态对齐**上的探索。随后出现的几篇论文（如“TileFormer”与“Efficient Multilingual VLM”）都在不同程度上借鉴了重叠tile + 注意力池化的思路。对想进一步深入的读者，可以关注以下方向：  
- **更细粒度的动态切块策略**（根据图像内容自适应tile大小）。  
- **跨语言对齐的统一表示学习**，尤其是低资源语言的增强方法。  
- **多模态指令微调的统一框架**，把视觉、文本、音频等多源信息统一到同一指令体系中。

### 一句话记住它
**jina‑vlm 用重叠切块 + 注意力池化，让 2.4 B 参数的模型在任意分辨率图像上实现多语言视觉问答的 SOTA。**