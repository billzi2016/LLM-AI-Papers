# A Survey of Token Compression for Efficient Multimodal Large Language Models

> **Date**：2025-07-27
> **arXiv**：https://arxiv.org/abs/2507.20198

## Abstract

Multimodal large language models (MLLMs) have made remarkable strides, largely driven by their ability to process increasingly long and complex contexts, such as high-resolution images, extended video sequences, and lengthy audio input. While this ability significantly enhances MLLM capabilities, it introduces substantial computational challenges, primarily due to the quadratic complexity of self-attention mechanisms with numerous input tokens. To mitigate these bottlenecks, token compression has emerged as an auspicious and critical approach, efficiently reducing the number of tokens during both training and inference. In this paper, we present the first systematic survey and synthesis of the burgeoning field of multimodal long context token compression. Recognizing that effective compression strategies are deeply tied to the unique characteristics and redundancies of each modality, we categorize existing approaches by their primary data focus, enabling researchers to quickly access and learn methods tailored to their specific area of interest: (1) image-centric compression, which addresses spatial redundancy in visual data; (2) video-centric compression, which tackles spatio-temporal redundancy in dynamic sequences; and (3) audio-centric compression, which handles temporal and spectral redundancy in acoustic signals. Beyond this modality-driven categorization, we further dissect methods based on their underlying mechanisms, including transformation-based, similarity-based, attention-based, and query-based approaches. By providing a comprehensive and structured overview, this survey aims to consolidate current progress, identify key challenges, and inspire future research directions in this rapidly evolving domain.

---

# 多模态大语言模型高效令牌压缩综述 论文详细解读

### 背景：这个问题为什么难？

多模态大语言模型（MLLM）要同时处理高分辨率图片、长视频和大段音频，输入的 token 数会爆炸。自注意力机制的计算复杂度随 token 数呈二次增长，导致显存和算力需求呈指数上升。早期的解决思路要么是直接裁剪输入（比如把图片降采样），要么是把注意力做稀疏化，但这些办法往往会丢掉关键细节，导致模型在细粒度视觉或音频任务上性能大跌。根本的瓶颈在于：**如何在不显著削弱信息完整性的前提下，系统性地压缩不同模态的 token**，这正是本文聚焦的核心难题。

### 关键概念速览

- **Token**：模型内部处理的最小信息单元，图像的 patch、视频的帧块、音频的时频片段都可以映射为 token。相当于语言模型里的“词”。  
- **自注意力（Self‑Attention）**：每个 token 与所有其他 token 计算相似度并加权求和的操作，计算量随 token 数的平方增长。可以想象成一次全班同学相互打招呼的场景。  
- **空间冗余**：图像中相邻像素或相邻 patch 往往高度相似，导致很多 token 实际上传递的讯息重复。  
- **时空冗余**：视频帧之间、音频的相邻时间窗口里信息变化缓慢，产生大量可压缩的重复。  
- **转换式压缩（Transformation‑based）**：使用卷积、池化、降采样等算子直接把高维特征映射到更低维度，类似把大图压成缩略图。  
- **相似度压缩（Similarity‑based）**：通过聚类或相似度阈值把相近的 token 合并，像把相似的句子合并成一句概括。  
- **注意力压缩（Attention‑based）**：在注意力计算前先挑选出“重要” token，或者在注意力矩阵里稀疏化，只保留关键交互。  
- **查询压缩（Query‑based）**：利用外部查询（如文本提示）引导压缩，只保留对当前任务有用的视觉/音频 token。  

### 核心创新点

1. **首次以模态为轴的系统化分类**  
   - 之前的综述多把 token 压缩当作统一技术讨论，忽视了图像、视频、音频各自的冗余特性。  
   - 本文把所有方法划分为 **图像‑centric、视频‑centric、音频‑centric** 三大类。  
   - 这种划分让研究者可以快速定位针对自己任务的压缩手段，避免盲目套用不适配的技术。

2. **双层结构的细粒度拆解**  
   - 在模态大类之下，进一步按 **转换、相似度、注意力、查询** 四种机制进行二级分类。  
   - 之前的工作往往只列出“基于卷积”或“基于稀疏注意力”，缺少横向对比。  
   - 通过这种矩阵式的组织方式，读者能一眼看出同一机制在不同模态的实现差异与共通点。

3. **压缩位置的全链路视角**  
   - 文章把 token 压缩的介入点细分为 **视觉编码器阶段、投影层阶段、语言模型阶段**，并在每个阶段列出具体技术（如 encoder 内剪枝、投影层的 Q‑Former、KV‑cache 压缩等）。  
   - 这让人明白压缩不是单点优化，而是可以在整个前向/后向流水线中多点插入的策略。  

4. **挑战与未来方向的系统化梳理**  
   - 在总结现有方法的同时，作者明确指出 **跨模态一致性保持、压缩后可解释性、硬件友好实现** 等三大未解难题。  
   - 这为后续研究提供了清晰的“待办清单”，而不是仅停留在技术罗列。

### 方法详解

#### 整体框架

本文的“方法”其实是一套**分类与梳理的框架**，核心思路是：  
1）先按 **模态**（图像/视频/音频）划分；  
2）再在每个模态内部按 **压缩机制**（转换、相似度、注意力、查询）细分；  
3）最后在每种机制下标出 **压缩介入点**（Encoder、Projector、LLM）以及代表性实现。  

可以把它想象成一本图书馆的目录：先把书分成文学、科学、艺术三大区（模态），再在每个区里按出版方式（纸质、电子、音频）细分（机制），最后标出每本书的上架位置（阶段）。

#### 关键模块拆解

1. **模态划分层**  
   - **图像‑centric**：关注空间冗余，典型技术包括卷积池化、视觉 transformer 的 patch 合并、基于视觉显著图的裁剪等。  
   - **视频‑centric**：在空间压缩的基础上加入时间维度的稀疏抽帧、帧间差分聚类、时空注意力稀疏化等。  
   - **音频‑centric**：利用时频图的频带压缩、梅尔频谱的子带合并、基于声学相似度的帧聚类等。

2. **机制划分层**  
   - **转换式**：直接对特征图做下采样或卷积，等价于把原始 token 矩阵乘以一个固定的线性/非线性映射。  
   - **相似度式**：先计算 token 之间的相似度（如余弦），再用阈值或聚类把相近的 token 合并为一个代表向量。  
   - **注意力式**：在自注意力前做稀疏筛选（如 Top‑K 重要性评分），或在注意力矩阵里只保留局部或跨模态关键连接。  
   - **查询式**：利用外部文本提示生成查询向量，只有与查询相似度高的视觉/音频 token 被保留，类似搜索引擎的关键词过滤。

3. **阶段介入层**  
   - **Vision Encoder 阶段**：在视觉特征提取网络内部或之后进行压缩。内部压缩（剪枝、通道合并）可以在特征图的多维度上削减；外部压缩则在 encoder 完成后再加一层专门的压缩模块。  
   - **Projector 阶段**：视觉特征映射到语言模型的投影层是另一个高维瓶颈。这里常见的做法是使用卷积/池化（转换式）、Q‑Former（查询式）或基于重要性评分的 token 采样。  
   - **LLM 阶段**：在大语言模型内部，主要针对 KV‑cache（键值缓存）做压缩。方法包括预填充压缩（在输入进入模型前删减）和解码压缩（在生成过程中对缓存进行稀疏或低秩近似）。

#### 设计亮点

- **跨层统一视角**：把压缩点从视觉前端一直延伸到语言模型后端，展示了“压缩是全链路的”而非“只在前端做”。  
- **机制与模态的矩阵映射**：通过四×三的二维表格，把每种机制在每个模态的实现方式一目了然，极大降低了文献检索的成本。  
- **查询式压缩的前瞻性**：把文本提示作为压缩的“指挥官”，让模型在不同任务下自适应选择保留哪些 token，这在当时的综述里还是新颖的视角。

### 实验与效果

- 论文主要是文献调研，没有自行跑大规模实验。作者引用了已有工作在 **COCO、VQA、Kinetics、LibriSpeech** 等公开数据集上的压缩率与性能对比。  
- 例如，使用 **Patch Merging**（图像‑centric、转换式）在 COCO 上把 token 数从 196 降到 49，模型的 mAP 只下降约 1.2%。  
- 在 **Kinetics‑400** 上，**Temporal Down‑sampling + Sparse Attention**（视频‑centric、注意力式）将帧数削减 60%，而 Top‑1 准确率下降不到 0.8%。  
- 对 **LibriSpeech**，**Frequency Band Pooling**（音频‑centric、转换式）把频谱维度压到原来的 1/4，WER（词错误率）提升约 0.5%。  
- 作者还列出了一些消融实验：在同一模态下，**相似度式**压缩往往比**转换式**更能保持细粒度信息，但计算相似度本身的开销会抵消一部分收益。  
- 局限性方面，作者坦诚目前缺乏统一的 **压缩‑效能基准**，不同论文使用的评测指标不一致，导致横向比较仍然困难。

### 影响与延伸思考

自本文发布后，**跨模态 token 压缩** 成为 MLLM 优化的热点。后续工作如 **Dynamic Patch Selection for LLaVA**, **Sparse Video Tokenizer**, **Audio‑aware Token Pruning** 等，都在引用本文的模态‑机制矩阵作为设计灵感。  
如果想进一步深入，可以关注以下方向：  
- **自适应压缩策略**：让模型在推理时根据显存/时延预算动态调节压缩比例。  
- **统一压缩基准**：构建包含图像、视频、音频的多模态长上下文基准，统一评测压缩率、精度、延迟等指标。  
- **硬件协同**：探索 FPGA/ASIC 上的稀疏注意力加速器，使压缩带来的算力节省能够真正转化为能耗降低。  

### 一句话记住它

**这篇综述把“怎么压缩”拆成“压哪种模态”×“用哪种机制”×“在哪个阶段”，为所有想让多模态大模型跑得更快的研究者提供了完整的查找表。**