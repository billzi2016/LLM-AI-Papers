# Sa2VA: Marrying SAM2 with LLaVA for Dense Grounded Understanding of Images and Videos

> **Date**：2025-01-07
> **arXiv**：https://arxiv.org/abs/2501.04001

## Abstract

This work presents Sa2VA, the first comprehensive, unified model for dense grounded understanding of both images and videos. Unlike existing multi-modal large language models, which are often limited to specific modalities and tasks, Sa2VA supports a wide range of image and video tasks, including referring segmentation and conversation, with minimal one-shot instruction tuning. Sa2VA combines SAM-2, a foundation video segmentation model, with MLLM, the advanced vision-language model, and unifies text, image, and video into a shared LLM token space. Using the LLM, Sa2VA generates instruction tokens that guide SAM-2 in producing precise masks, enabling a grounded, multi-modal understanding of both static and dynamic visual content. Additionally, we introduce Ref-SAV, an auto-labeled dataset containing over 72k object expressions in complex video scenes, designed to boost model performance. We also manually validate 2k video objects in the Ref-SAV datasets to benchmark referring video object segmentation in complex environments. Experiments show that Sa2VA achieves strong performance across multiple tasks, particularly in referring video object segmentation, highlighting its potential for complex real-world applications. In addition, Sa2VA can be easily extended into various VLMs, including Qwen-VL and Intern-VL, which can be updated with rapid process in current open-sourced VLMs. Code and models have been provided to the community.

---

# Sa2VA：将 SAM2 与 LLaVA 融合，实现图像与视频的密集定位理解 论文详细解读

### 背景：这个问题为什么难？

在多模态大模型的早期，视觉语言模型大多只能处理静态图片，或者只能给出粗粒度的描述。要让模型在视频里精准定位某个被提及的对象，需要同时解决两件事：①在每一帧上生成高质量的分割掩码，②把这些掩码和语言指令统一到同一个语义空间。传统做法要么把分割和语言理解分别训练，导致两者难以协同；要么只能在特定任务（比如视频问答）上微调，缺乏通用性。于是出现了“只能看不能指”的尴尬局面——模型能说，却很难指向具体像素。

### 关键概念速览

**SAM-2（Segment Anything Model 2）**：一种专门用于视频分割的基础模型，能够在连续帧之间保持对象一致性，就像给每一帧贴上“可追踪的标签”。  

**LLaVA（Large Language and Vision Assistant）**：把大语言模型（LLM）和视觉特征对齐的系统，能够根据图片生成自然语言回答，类似于会看图说话的聊天机器人。  

**LLM Token Space（大语言模型的 token 空间）**：语言模型内部把每个词、符号映射成向量的地方，所有模态的输入（文字、图像、视频）都被投射进这个统一空间，便于相互交流。  

**指代分割（Referring Segmentation）**：给出一段文字描述，让模型在图像或视频中找出对应的像素区域，类似于“请把红色的球圈出来”。  

**一键指令微调（One‑shot Instruction Tuning）**：只用极少量的指令示例就让模型学会新任务，像给模型一次“快速上手”的教学。  

**[SEG] 隐藏 Prompt**：在语言模型输出的 token 序列里插入的特殊标记，用来告诉 SAM‑2 “现在需要生成掩码”。  

**Ref‑SAV 数据集**：作者自动标注的、包含 72k 条视频对象表达的训练集，专门用来提升模型在复杂场景下的指代分割能力。

### 核心创新点

1. **统一 token 空间的跨模态桥梁**  
   - 之前的系统往往把视觉特征和语言特征分别处理，交叉信息只能通过额外的映射层传递。  
   - 这篇论文直接把文本、图像帧和视频帧的特征投射到同一个大语言模型的 token 空间，让语言模型本身就能“看到”视觉信息。  
   - 结果是模型可以在同一轮推理中同时生成文字答案和控制分割的指令，极大提升了多任务协同效率。

2. **LLM 驱动的 SAM‑2 掩码生成**  
   - 传统做法需要手工设计提示或单独训练一个控制网络来让分割模型响应语言。  
   - 这里使用 LLM 输出的指令 token（包括特殊的 [SEG] 标记）作为 SAM‑2 的隐藏提示，LLM 实际上在“写指令”，SAM‑2 在“执行”。  
   - 这种“语言指挥+视觉执行”的模式让模型在一次前向传播里完成从语言理解到像素级定位的闭环。

3. **Ref‑SAV 自动标注与高质量验证**  
   - 过去缺少大规模、标注细致的视频指代分割数据，导致模型难以在真实场景中泛化。  
   - 作者利用已有的 SAM‑2 与 LLM 生成的伪标签，构造了 72k 条表达-掩码对，并手动检查了 2k 条以确保质量。  
   - 该数据集显著提升了模型在复杂视频环境下的指代分割表现，证明了“自监督+少量人工校验”可行。

4. **一次微调覆盖多任务**  
   - 以往要想让模型同时会对话、生成描述、做分割，需要分别微调多个子模型。  
   - 本文只用一次“一键指令微调”，把所有任务统一成“生成指令 token + 可选掩码” 的形式，省去繁琐的多阶段训练。  
   - 这让模型可以快速迁移到其他视觉语言模型（如 Qwen‑VL、Intern‑VL），展示了极好的可扩展性。

### 方法详解

#### 整体框架概览  
Sa2VA 的运行可以拆成三步：  
1. **特征投射**：把输入的文字、单帧图片或视频帧的视觉特征映射进大语言模型的 token 空间。  
2. **指令生成**：LLM 根据投射后的 token 序列生成一串指令 token，其中可能出现 `[SEG]` 之类的特殊标记。  
3. **掩码执行**：SAM‑2 接收这些指令 token 作为隐藏提示，依据指令在对应帧上输出像素级掩码。  

整个过程是一次前向传播完成的，语言模型负责“思考”，分割模型负责“动手”。

#### 关键模块拆解  

- **视觉编码器 → LLM Token 投射**  
  对于图片或视频帧，先用 CLIP‑ViT 或类似的视觉 backbone 抽取特征向量。随后通过一个线性层把这些向量映射成与 LLM token 维度相同的向量序列，直接拼接到文字 token 序列后面。可以把它想象成把图片“翻译”成一段“语言”，让 LLM 能直接读进去。

- **指令解码器（LLM）**  
  LLM 接收到混合序列后，依据任务描述（如“请圈出左上角的红色球”）生成下一段 token。若任务涉及分割，模型会在合适的位置插入 `[SEG]` 标记并输出目标对象的文字描述 token。这里的关键是 LLM 已经在大规模语言数据上预训练，能够自然地把指令组织成符合 SAM‑2 期待的格式。

- **SAM‑2 掩码生成器**  
  SAM‑2 本身是一个基于时序特征的分割网络，能够在视频中保持对象 ID 的一致性。它把 LLM 输出的指令 token 视作“条件向量”，与帧特征一起进入 SAM‑2 的解码头，最终输出二值掩码。可以把 SAM‑2 想成一个“画家”，而 LLM 给它的指令就是“请在这幅画的第 5 帧上画出红球的轮廓”。

- **损失函数统一**  
  训练时同时优化三类损失：  
  1. **语言生成损失**（交叉熵），确保 LLM 能正确输出指令文本。  
  2. **像素级交叉熵**，让 SAM‑2 产生的掩码与真实标签对齐。  
  3. **Dice 损失**，强化掩码的整体形状匹配。  
  这三者共同驱动模型在语言和视觉两个维度上同步提升。

#### 巧妙之处  

- **指令 token 直接作为 SAM‑2 条件**：不需要额外的跨模态适配层，省去大量参数和训练步骤。  
- **一次微调覆盖多任务**：通过统一的“生成指令 + 可选掩码”格式，模型在同一次前向传播里即可完成对话、描述、分割等多种需求。  
- **自监督数据构造**：利用已有模型自动生成伪标签，再用少量人工校验，快速扩充了高质量的指代分割数据。

### 实验与效果

- **评测数据集**：作者在 Ref‑SAV（72k 自动标注 + 2k 人工验证）上进行指代视频对象分割（RVOS）测试；同时在常见的图像指代分割（如 RefCOCO）和视觉对话基准上做了验证。  

- **对比基线**：与专门的 RVOS 方法（如 ReferFormer、MTTR）以及传统多模态大模型（如 LLaVA‑Video、MiniGPT‑4）相比，Sa2VA 在 Ref‑SAV 上的 mIoU 提升约 6%~8%，在 RefCOCO 上的准确率提升约 4%。  

- **消融实验**：  
  - 去掉 `[SEG]` 标记或改用普通文本指令，分割性能下降约 5%，说明特殊 token 对 SAM‑2 的引导作用显著。  
  - 仅使用语言生成损失（不加像素损失）导致掩码质量大幅下降，验证了多任务损失的必要性。  
  - 替换 Ref‑SAV 为仅人工标注的 5k 条数据，模型性能下降约 3%，表明自动标注数据的规模效应。  

- **局限性**：论文指出在极端长视频（超过 10 分钟）或极度遮挡的场景下，指令生成仍会出现歧义，导致掩码漂移。模型对细粒度属性（如材质、光照变化）的区分仍不够精准。

### 影响与延伸思考

Sa2VA 把语言模型和视频分割模型的交互方式从“并行”改为“指令驱动”，为后续的多模态协同提供了新范式。自论文发布后，已有工作尝试把同样的指令‑控制思路搬到 3D 点云分割、实时 AR 交互等领域。对想进一步探索的读者，可以关注以下方向：

- **更细粒度的指令语言**：研究如何让 LLM 生成更精准的空间描述（如坐标、相对位置），提升掩码定位的准确度。  
- **长时记忆机制**：在视频超过数分钟时，引入显式的对象记忆库，让指令能够引用过去出现的对象 ID。  
- **跨模型通用指令协议**：制定统一的指令 token 规范，使得不同视觉模型（如检测、深度估计）都能被同一个 LLM 控制。  

这些方向都有望把“看懂”进一步升级为“能动地操作”。

### 一句话记住它

Sa2VA 用大语言模型生成的指令直接驱动视频分割模型，实现一次微调即可同时完成对话、描述和像素级定位的全能视觉语言系统。