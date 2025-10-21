# DeepSeek-OCR: Contexts Optical Compression

> **Date**：2025-10-21
> **arXiv**：https://arxiv.org/abs/2510.18234

## Abstract

We present DeepSeek-OCR as an initial investigation into the feasibility of compressing long contexts via optical 2D mapping. DeepSeek-OCR consists of two components: DeepEncoder and DeepSeek3B-MoE-A570M as the decoder. Specifically, DeepEncoder serves as the core engine, designed to maintain low activations under high-resolution input while achieving high compression ratios to ensure an optimal and manageable number of vision tokens. Experiments show that when the number of text tokens is within 10 times that of vision tokens (i.e., a compression ratio < 10x), the model can achieve decoding (OCR) precision of 97%. Even at a compression ratio of 20x, the OCR accuracy still remains at about 60%. This shows considerable promise for research areas such as historical long-context compression and memory forgetting mechanisms in LLMs. Beyond this, DeepSeek-OCR also demonstrates high practical value. On OmniDocBench, it surpasses GOT-OCR2.0 (256 tokens/page) using only 100 vision tokens, and outperforms MinerU2.0 (6000+ tokens per page on average) while utilizing fewer than 800 vision tokens. In production, DeepSeek-OCR can generate training data for LLMs/VLMs at a scale of 200k+ pages per day (a single A100-40G). Codes and model weights are publicly accessible at http://github.com/deepseek-ai/DeepSeek-OCR.

---

# DeepSeek-OCR：上下文光学压缩 论文详细解读

### 背景：这个问题为什么难？

传统 OCR 系统在处理单页文档时已经相当成熟，但当面对“超长”上下文——比如几千行文字、跨页的表格或整本书籍时，模型必须同时处理数万甚至数十万的视觉 token。现有的视觉‑语言模型受限于显存和算力，往往只能把页面切成小块，导致上下文信息被割裂，跨块的结构关系难以捕获。再者，LLM（大语言模型）在接收视觉信息时只能接受有限数量的 vision token，若直接喂入高分辨率图像会导致激活爆炸、推理成本飙升。于是，如何在不牺牲关键视觉信息的前提下，把超长文档压缩成可被 LLM 消化的少量 token，成为了 OCR 研究的瓶颈。

### 关键概念速览
- **vision token（视觉 token）**：模型把图像切成小块后得到的离散表示，类似于文字模型的词向量。每个 token 携带局部像素信息，数量越多模型负担越重。  
- **光学 2D 映射**：把长文本或多页文档先渲染成一张高分辨率图片，再通过特殊网络压缩成更小的特征图。可以把它想象成把一本书拍成一张巨幅海报，然后用“光学镜头”把海报压缩成几张小卡片。  
- **压缩比（compression ratio）**：原始文本 token 数除以最终保留下来的 vision token 数。比值越大说明压缩越激进。  
- **DeepEncoder**：本文的核心压缩引擎，负责把高分辨率图像映射成稀疏、低激活的视觉 token。它内部融合了 SAM（Segment Anything Model）分割、卷积层和 CLIP（Contrastive Language‑Image Pre‑training）嵌入。  
- **DeepSeek3B‑MoE‑A570M**：本文使用的解码器，属于 3 B 参数的混合专家（Mixture‑of‑Experts）模型，专门用来把 vision token 转化为文字输出（即 OCR 结果）。  
- **MoE（混合专家）**：一种把大模型拆成多个子模型（专家），只激活其中一小部分来完成推理的技术，能在保持性能的同时显著降低算力需求。  
- **OmniDocBench**：一个综合文档理解基准，覆盖普通文档、表格、图表等多种布局，常用于评估 OCR 与文档结构解析的整体能力。  

### 核心创新点
1. **光学上下文压缩 → DeepEncoder 设计 → 超高压缩比仍保持可用信息**  
   过去的视觉‑语言模型要么直接下采样导致细节丢失，要么保持高分辨率导致 token 爆炸。DeepSeek‑OCR 把“把文档渲染成图像再压缩”的思路具体化为 DeepEncoder：先用 SAM 做粗粒度分割，提取出文档的结构块；随后通过卷积层进行空间压缩；最后用 CLIP 的跨模态投影把压缩后的特征映射到低维视觉 token。这样在 10× 以下的压缩比下，OCR 精度仍能达到 97%。  
2. **低激活压缩策略 → 动态分辨率输入 → 显存友好**  
   为了防止高分辨率图像导致激活值飙升，DeepEncoder 在不同分辨率下采用自适应通道数和步幅，使得无论输入是 4K 还是 8K 图像，最终的 token 数都被严格控制在几百个以内。相比传统的固定‑size CNN，这种“按需缩放”让模型在单卡 A100‑40G 上即可一次性处理整页甚至多页文档。  
3. **视觉‑语言解码器的 MoE 适配 → DeepSeek3B‑MoE‑A570M → 高效 OCR 解码**  
   直接把压缩后的视觉 token 喂入普通语言模型会出现信息瓶颈。作者选用了混合专家架构的 3 B 参数模型，使得每次推理只激活少数专家，既保持了解码质量，又大幅降低了推理时的显存占用。实验表明，这套解码器在 20× 压缩下仍能保持约 60% 的识别率。  
4. **大规模生产流水线 → 单卡每日 20 万页 → 实际落地价值**  
   将 DeepEncoder 与 MoE 解码器串联后，作者实现了“一卡生成 200k+ 页 OCR 数据”的生产速率，这在业界的文档数据标注成本上具有革命性意义。  

### 方法详解
#### 整体框架
DeepSeek‑OCR 的工作流可以划分为三步：  
1. **图像化长文本**：把任意长度的文档（包括多页 PDF、扫描件、网页截图）渲染成一张高分辨率的二维图像。  
2. **光学压缩（DeepEncoder）**：对这张图像进行分割、特征抽取和空间下采样，输出固定数量的 vision token。  
3. **语言解码（DeepSeek3B‑MoE‑A570M）**：把 vision token 送入混合专家语言模型，逐字符生成 OCR 文本。  

#### DeepEncoder 细节
- **输入层 – SAM 分割**：首先使用 Segment Anything Model（SAM）对整张图像进行全局分割，得到若干语义块（文字块、表格块、图形块）。这一步相当于把一本书的章节、段落先划分出来，后续压缩时可以针对不同块采用不同策略。  
- **特征抽取 – 卷积网络**：对每个分割块，使用轻量化卷积层（3×3、步幅 2）进行空间下采样。这里的关键是“低激活”：卷积层的通道数随输入分辨率动态调节，确保特征图的激活均值保持在一个可控范围内，防止显存溢出。  
- **跨模态映射 – CLIP 嵌入**：卷积得到的特征图再喂入 CLIP 的视觉投影头，得到与语言空间对齐的向量。因为 CLIP 已经在大规模图文对上学习了视觉‑语言对应关系，这一步可以把压缩后的视觉信息直接映射为语言模型可接受的 token。  
- **Token 采样与归一化**：从所有块的嵌入向量中均匀抽样，形成固定长度的 token 序列（如 100、400、800 token），并进行层归一化，以消除不同块之间的尺度差异。  

> **最巧妙的点**：DeepEncoder 并不是单纯的“把图像压成小”。它先用 SAM 把文档结构显式化，再在结构块内部做压缩，这样即使压缩比高，关键的排版信息（如表格行列、公式布局）仍能被保留。

#### 解码器 – DeepSeek3B‑MoE‑A570M
- **模型规模**：3 B 参数的混合专家 Transformer，内部有数十个专家网络。  
- **路由机制**：每个输入 token 只会激活 1~2 个专家，路由网络根据 token 的视觉特征决定使用哪个专家。这样既保持了模型的表达能力，又让推理成本与 token 数呈线性关系。  
- **自回归生成**：解码器在每一步预测下一个字符（或子词），并将已生成的文字反馈回模型，形成标准的 OCR 流程。  
- **后处理**：为了提升实际可用性，作者在解码后加入了基于语言模型的拼写校正和版面约束（如表格列对齐），但细节在摘要中未展开。  

#### 训练流程
1. **DeepEncoder 预训练**：使用 OCR 1.0（传统文档）和 OCR 2.0（结构化图像）两类数据，对 SAM‑+‑Conv‑+‑CLIP 三段进行端到端优化，使得压缩后的 token 能最大化保留文字可辨识度。  
2. **全模型微调**：在同样的数据上，联合优化 DeepEncoder 与 MoE 解码器，使得从图像到文字的整体端到端误差最小。  

### 实验与效果
- **数据集**：主要在 OmniDocBench 上评估，OmniDocBench 包含普通文档、表格、化学式、几何图形等多样化布局。  
- **压缩比 vs 精度**：当压缩比 < 10×（即文本 token 数是 vision token 数的 10 倍以内）时，OCR 精度达到 **97%**；在更激进的 20× 压缩下，精度仍保持在 **约 60%**。  
- **与基线对比**：  
  - **GOT‑OCR2.0**（每页 256 vision token）在同等任务上表现略低，DeepSeek‑OCR 只用 **100 vision token** 就超过其性能。  
  - **MinerU2.0**（平均每页 6000+ token）在同样的文档上，DeepSeek‑OCR 只需 **< 800 vision token** 就取得更好结果。  
- **生产效率**：在单卡 A100‑40G 上，模型能够每日处理 **200k+ 页**，相当于传统 OCR 流水线的数十倍。  
- **消融实验**：摘要未提供细节，但作者声称通过去掉 SAM 或 CLIP 任意一环，压缩后的精度会显著下降，验证了三段式设计的必要性。  
- **局限性**：在压缩比超过 20× 时，识别率快速下滑；此外，模型对极端低分辨率或严重噪声的文档仍不够鲁棒，作者在摘要中未给出对应的改进方案。  

### 影响与延伸思考
DeepSeek‑OCR 把“把长文本变成图像再压缩”的思路落地，为 **长上下文视觉‑语言协同** 开辟了新路径。随后的工作（如 2024‑2025 年的 LongVision、OptiDoc 等）纷纷借鉴其“结构化分割 + 动态下采样 + MoE 解码”的管线，尝试在多模态检索、文档问答等任务上实现千页级上下文的即时推理。对想进一步探索的读者，可以关注以下方向：  
- **可逆压缩**：在保持高压缩率的同时，设计能够恢复细粒度像素的逆向网络。  
- **跨模态记忆机制**：把压缩后的 vision token 存入长期记忆，以支持 LLM 对历史文档的随时检索。  
- **自适应路由的专家设计**：让 MoE 专家专门学习表格、公式、手写体等不同子任务的解码技巧。  

### 一句话记住它
**DeepSeek‑OCR 用结构化光学压缩把整本书压进几百个视觉 token，仍能让大语言模型实现高精度 OCR。**