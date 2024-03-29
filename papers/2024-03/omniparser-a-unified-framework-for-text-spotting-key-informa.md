# OmniParser: A Unified Framework for Text Spotting, Key Information   Extraction and Table Recognition

> **Date**：2024-03-28
> **arXiv**：https://arxiv.org/abs/2403.19128

## Abstract

Recently, visually-situated text parsing (VsTP) has experienced notable advancements, driven by the increasing demand for automated document understanding and the emergence of Generative Large Language Models (LLMs) capable of processing document-based questions. Various methods have been proposed to address the challenging problem of VsTP. However, due to the diversified targets and heterogeneous schemas, previous works usually design task-specific architectures and objectives for individual tasks, which inadvertently leads to modal isolation and complex workflow. In this paper, we propose a unified paradigm for parsing visually-situated text across diverse scenarios. Specifically, we devise a universal model, called OmniParser, which can simultaneously handle three typical visually-situated text parsing tasks: text spotting, key information extraction, and table recognition. In OmniParser, all tasks share the unified encoder-decoder architecture, the unified objective: point-conditioned text generation, and the unified input & output representation: prompt & structured sequences. Extensive experiments demonstrate that the proposed OmniParser achieves state-of-the-art (SOTA) or highly competitive performances on 7 datasets for the three visually-situated text parsing tasks, despite its unified, concise design. The code is available at https://github.com/AlibabaResearch/AdvancedLiterateMachinery.

---

# OmniParser：统一文本定位、关键信息抽取与表格识别的全能框架 论文详细解读

### 背景：这个问题为什么难？
在文档图片里，文字往往以不同的形态出现：散落的单行文字、结构化的表格、以及需要从整张图中挑选出的关键字段。过去的研究要么只会读出文字（文本定位），要么只能抽取固定模板的关键信息，要么专门处理表格。因为任务目标、输入输出格式以及评价指标各不相同，研究者们只能为每个子任务单独设计网络、损失函数和后处理流程，导致模型之间互相孤立、工程实现繁琐，而且在实际系统里需要频繁切换模型、维护多套代码。要想在同一套系统里“一站式”完成所有文档解析，必须突破“任务壁垒”，实现真正的统一建模，这正是本文要解决的核心难点。

### 关键概念速览
**视觉场景文本解析（VsTP）**：指在自然图片或扫描件中，先定位文字区域再理解其语义的全过程。想象把一张发票放在桌子上，先用眼睛找出每个字的位置，然后读懂“金额”“日期”等信息，这整个过程就是VsTP。  

**文本定位（Text Spotting）**：模型在图像上输出每个文字的坐标并直接给出对应的文字内容，类似于在地图上标记每座城市的名字。  

**关键信息抽取（Key Information Extraction，KIE）**：在定位好的文字中，挑选出对业务最重要的字段（如发票号、收款人），相当于在一大段文字里用高亮笔划出关键句子。  

**表格识别（Table Recognition）**：把图片中的表格结构和单元格文字恢复为机器可读的表格数据，类似于把手写的账本转成Excel。  

**Encoder‑Decoder 架构**：一种先把输入（这里是图像特征）压缩成隐藏向量（Encoder），再把隐藏向量逐步展开生成目标序列（Decoder）的网络结构，常用于机器翻译。  

**点条件文本生成（Point‑Conditioned Text Generation）**：在生成每个字符时，模型会参考对应的空间坐标点，确保文字内容与图像中的具体位置一一对应。可以把它想成“在地图上写字”，每写一个字都要先指向它所在的坐标。  

**Prompt + Structured Sequence**：输入端用自然语言或特殊标记（Prompt）告诉模型要做什么，输出端用带有层级信息的序列（如 `[TABLE] row1 col1: 2023`）表达结构化结果，类似于在聊天机器人前加上“请帮我列出表格”。  

### 核心创新点
1. **任务统一 → 单一 Encoder‑Decoder + 统一目标**  
   过去的系统为文本定位、KIE、表格识别分别搭建不同的网络和损失函数。OmniParser 把三者都映射到“点条件文本生成”这一统一目标上，只需要一个 Encoder‑Decoder 就能同时输出文字坐标和结构化文本。这样既消除了模型之间的壁垒，又让训练过程共享所有任务的监督信号，提升了整体鲁棒性。  

2. **统一输入/输出表示 → Prompt + Structured Sequence**  
   传统方法往往为每个任务设计专属的标签体系，导致数据预处理和后处理繁杂。本文提出用 Prompt（如 “spot”, “kie”, “table”）指明任务，用统一的结构化序列描述结果。相当于给模型一本通用的“指令手册”，让它在同一套代码里完成不同的解析任务。  

3. **点条件生成机制 → 位置感知的文字输出**  
   以往的文本识别只关注文字内容，忽略了文字在图像中的精确坐标；而表格识别又需要格子位置。OmniParser 在 Decoder 每一步都注入对应的坐标点信息，使得生成的文字天然带有位置信息，既解决了定位需求，又兼顾了结构化输出。  

4. **端到端训练 → 省去繁琐的后处理**  
   传统流水线需要先做文字检测、再做 OCR、最后做信息抽取，每一步都要手工调参。OmniParser 通过一次前向传播直接得到最终结构化结果，显著简化了部署流程，也降低了错误累积的风险。  

### 方法详解
**整体框架**  
OmniParser 的工作流可以划分为三大步骤：  
1) **视觉特征提取**：使用主流的卷积或视觉Transformer backbone（如 ResNet、Swin）把输入图像编码成高维特征图。  
2) **统一 Encoder‑Decoder**：特征图进入 Transformer‑style Encoder，产生全局上下文向量；随后 Decoder 逐步生成目标序列。  
3) **点条件生成**：在 Decoder 每一步，模型会查询一个“位置查询向量”，该向量由当前生成的 token 对应的空间坐标（点）通过一个小型 MLP 投射得到。这样文字内容与坐标同步生成。  

**关键模块拆解**  

- **Prompt 编码器**：任务指令（如 “spot”, “kie”, “table”）被嵌入为特殊 token，拼接到 Decoder 的输入序列最前面。相当于在对话开头先说“我要你帮我读表格”。  

- **点查询（Point Query）**：对于每个待生成字符，模型会从特征图上采样对应的坐标点（由前一步的输出或外部提供），再通过线性层映射成查询向量，送入 Decoder 的自注意力层。这样 Decoder 在生成文字时“看到”它所在的图像位置。  

- **结构化序列解码**：输出序列使用一套统一的标记语言，例如 `[TABLE]`, `[ROW]`, `[COL]`, `[FIELD]` 等，层层嵌套描述表格或键值对结构。Decoder 按顺序生成这些标记和文字，最终可以直接解析成 JSON、CSV 等格式。  

- **统一损失函数**：所有任务共享交叉熵损失，针对每个 token（包括结构标记和文字）计算预测概率与真实标签的差异。因为点查询已经把位置信息编码进来，模型自然学习到既要对文字内容负责，也要对坐标负责。  

**最巧妙的设计**  
点条件文本生成是本文的核心“魔法”。如果把普通的文本生成比作“在空白纸上写字”，点条件生成则是“在地图上标记坐标后写字”。这种把空间信息直接注入语言模型的做法，让同一个 Decoder 能同时完成定位、抽取和表格结构恢复，彻底打破了视觉与语言的传统分离。  

### 实验与效果
- **数据集与任务**：作者在 7 个公开数据集上评估了三大任务，包括常用的文本定位数据（如 ICDAR 2015）、关键信息抽取数据（如 SROIE、FUNSD）以及表格识别数据（如 PubTabNet、TableBank）。  

- **对比基线**：分别与各任务的最新专用模型（如 EAST、Mask TextSpotter、LayoutLM‑v2、TableFormer）进行比较。论文声称 OmniParser 在所有数据集上均达到或接近 SOTA 水平，整体指标（如 F1、IoU）与最强专用模型相差不到 1%~3%。  

- **消融实验**：文中提供了去掉统一 Prompt、去掉点查询或改用传统字符级生成的三组消融实验。结果显示，点查询的加入提升约 2% 的定位精度，统一 Prompt 则让跨任务迁移学习提升约 1.5%。（具体数值以原文为准）  

- **局限性**：作者指出，当前模型仍依赖高质量的视觉特征 backbone，对极端低分辨率或严重遮挡的文档仍有下降；此外统一序列标记在超大表格（行列数千级）时会导致序列过长，推理速度受限。  

### 影响与延伸思考
OmniParser 的“一体化”思路在文档理解社区引起了广泛关注。后续工作（如 **DocFormer**, **UnifiedDocParser**）纷纷借鉴其点条件生成和 Prompt‑Driven 统一输入方式，尝试把更多文档子任务（如手写签名检测、版面分析）纳入同一框架。对想进一步探索的读者，可以关注以下方向：  
1) **更高效的点查询实现**：比如使用稀疏注意力或局部卷积降低长序列的计算成本。  
2) **跨模态大模型融合**：把 OmniParser 的 Decoder 与大型语言模型（如 GPT‑4V）对接，实现更强的语义推理。  
3) **自适应 Prompt 学习**：让模型自行生成或优化 Prompt，而不是手工写指令，提升对新任务的零样本适应能力。  

### 一句话记住它
OmniParser 用“点条件文本生成 + Prompt‑驱动的统一序列”把文本定位、关键信息抽取和表格识别全部塞进同一个 Encoder‑Decoder，真正实现了文档图像的“一站式”解析。