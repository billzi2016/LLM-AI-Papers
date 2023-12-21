# V*: Guided Visual Search as a Core Mechanism in Multimodal LLMs

> **Date**：2023-12-21
> **arXiv**：https://arxiv.org/abs/2312.14135

## Abstract

When we look around and perform complex tasks, how we see and selectively process what we see is crucial. However, the lack of this visual search mechanism in current multimodal LLMs (MLLMs) hinders their ability to focus on important visual details, especially when handling high-resolution and visually crowded images. To address this, we introduce V*, an LLM-guided visual search mechanism that employs the world knowledge in LLMs for efficient visual querying. When combined with an MLLM, this mechanism enhances collaborative reasoning, contextual understanding, and precise targeting of specific visual elements. This integration results in a new MLLM meta-architecture, named Show, sEArch, and TelL (SEAL). We further create V*Bench, a benchmark specifically designed to evaluate MLLMs in their ability to process high-resolution images and focus on visual details. Our study highlights the necessity of incorporating visual search capabilities into multimodal systems. The code is available https://github.com/penghao-wu/vstar.

---

# V*: 引导式视觉搜索作为多模态大语言模型的核心机制 论文详细解读

### 背景：这个问题为什么难？
多模态大语言模型（MLLM）已经可以把文字和图片一起处理，但在面对高分辨率、细节密集的图像时，它们往往像“盲人摸象”，只能一次性把整张图喂进去，缺乏挑选关键区域的能力。传统的视觉特征提取器只能提供全局或固定尺度的特征，导致模型在复杂场景下容易被无关信息淹没。缺少类似人类“视觉搜索”的机制，使得模型在需要定位小目标、阅读细小文字或比较局部差异的任务上表现不佳，这正是本文要破解的瓶颈。

### 关键概念速览
- **多模态大语言模型（MLLM）**：能够同时接受文字和图像输入，并在语言模型的框架下生成文字输出的系统。想象成一个会说话的“看图机器人”。  
- **视觉搜索（Visual Search）**：人类在观察场景时会主动把注意力投向感兴趣的区域，再细致分析。这里指模型在大图中主动“挑选”子区域进行深入处理。  
- **自上而下（Top‑down）注意**：由任务目标或语言提示驱动的注意力分配方式，类似于老师告诉学生“找出图中的红色车”。  
- **V\***：本文提出的“LLM‑guided 视觉搜索”模块，利用语言模型的世界知识生成查询，指导视觉特征提取器定位感兴趣区域。  
- **SEAL（Show, sEArch, and TelL）**：把 V\* 嵌入到 MLLM 中的整体架构，分别对应展示图像、搜索细节、输出答案三个阶段。  
- **V*Bench**：专门为评估模型在高分辨率图像上进行细粒度视觉搜索能力而设计的基准套件。  
- **视觉查询（Visual Query）**：由语言模型生成的文字描述或关键词，用来在视觉特征空间中检索对应的区域。  

### 核心创新点
1. **从全局特征到查询驱动的局部特征**  
   - 之前的 MLLM 大多一次性把整张图的特征喂进语言模型，缺乏针对性。  
   - V\* 让语言模型先产生一段“视觉查询”，再把这段查询交给视觉编码器去检索对应的局部特征。  
   - 这样模型可以在需要时聚焦细小目标，显著提升了在高分辨率、细节密集图像上的表现。  

2. **自上而下的注意机制与语言知识的深度融合**  
   - 传统视觉注意往往是自下而上（基于图像本身的显著性）或固定的软注意。  
   - V\* 把语言模型的世界知识直接转化为注意指令，使得注意力分配更符合任务需求，例如“找出图中左上角的红色标志”。  
   - 这种融合让模型在跨模态推理时能够更自然地把语言线索映射到视觉空间。  

3. **SEAL 元架构的三阶段协同**  
   - “Show”阶段负责把原始图像送入视觉编码器得到粗略特征；  
   - “sEArch”阶段调用 V\* 生成并执行视觉查询，得到细粒度特征；  
   - “TelL”阶段把两阶段特征拼接进语言模型，完成答案生成。  
   - 通过明确的阶段划分，系统在推理过程中可以循环使用搜索结果，提升了解释性和可调试性。  

4. **专用评估基准 V*Bench**  
   - 现有的多模态评测多聚焦在问答或描述，忽略了对高分辨率细节的考察。  
   - V*Bench 引入了需要定位、计数、阅读小字等任务，专门检验模型的视觉搜索能力。  
   - 该基准为后续研究提供了统一的衡量标准。  

### 方法详解
整体思路可以看作一次“看‑想‑说”的循环。首先把整张图送入标准的视觉编码器（如 ViT），得到一组全局特征向量；随后语言模型根据任务描述生成一段查询文本，这段文本会被转化为向量并与全局特征进行匹配，检索出最相关的局部特征块；最后把全局特征、局部特征以及原始文本一起喂入语言模型，完成答案输出。

**步骤拆解**  
1. **全局特征提取（Show）**  
   - 输入图像 → ViT（或其他卷积/注意力网络） → 产生 N 个 patch 特征。  
   - 这些特征相当于“粗粒度的视野”。  

2. **语言驱动的查询生成（sEArch‑Query）**  
   - 语言模型接收任务指令（如“图中有几只猫？”）并在内部生成一段“视觉查询”。  
   - 查询可以是关键词（“猫”）也可以是更具体的属性描述（“左上角的黑白相间的猫”）。  

3. **查询向量化与注意匹配**  
   - 查询文本经过同一个语言模型的嵌入层，得到查询向量 Q。  
   - 将 Q 与每个 patch 特征做点积相似度，得到注意权重分布。  
   - 取权重最高的 K 个 patch（K 通常为 1~5），拼接成局部特征集合。  

4. **细粒度特征融合（sEArch‑Refine）**  
   - 对选中的局部 patch 再次通过更高分辨率的视觉编码器（可选）提取细节特征。  
   - 这一步相当于“放大镜”，让模型在感兴趣区域获得更丰富的信息。  

5. **答案生成（TelL）**  
   - 将全局特征、细粒度局部特征以及原始文本指令一起拼接成语言模型的输入序列。  
   - 语言模型在生成答案时能够直接访问到与查询对应的视觉信息，从而给出更精准的回答。  

**巧妙之处**  
- 查询生成完全依赖语言模型的内部知识，无需额外的标注或检索数据库，实现了“零样本”视觉搜索。  
- 注意匹配采用的是软注意的形式，但只保留最高分的少数 patch，既保留了可微分性，又避免了信息稀释。  
- SEAL 的三阶段设计让每一步都可以单独评估和调试，提升了系统的可解释性。  

### 实验与效果
- **评测平台**：作者构建的 V*Bench 包含高分辨率图片、细粒度定位、计数、阅读小字等子任务。  
- **对比基线**：常见的 MLLM（如 Flamingo、BLIP‑2）以及直接使用全局特征的变体。  
- **结果概述**：论文声称在 V*Bench 上相较于基线取得了显著提升，尤其在需要定位小目标和阅读细小文字的子任务中，成功突破了原有模型的瓶颈。  
- **消融实验**：作者分别去掉查询生成、局部特征细化以及 SEAL 中的某一阶段，发现查询生成和局部细化对性能贡献最大，验证了自上而下注意的核心作用。  
- **局限性**：V\* 依赖语言模型生成的查询质量，若指令模糊或语言模型知识不足，搜索效果会下降；此外，额外的局部特征提取会带来一定的计算开销。  

### 影响与延伸思考
这篇工作首次把“视觉搜索”以语言驱动的方式引入多模态大语言模型，打开了让模型主动聚焦图像局部的思路。随后的研究开始探索更高效的查询生成（如使用小型专用检索网络）或把视觉搜索与工具使用（如 OCR、目标检测）结合，形成更强的“看‑做‑说”循环。想进一步了解，可以关注以下方向：  
- **跨模态检索**：如何让语言模型生成更精准的视觉查询。  
- **层次化注意**：在多尺度特征上实现自上而下与自下而上的协同。  
- **可解释多模态推理**：利用搜索路径作为模型解释的依据。  

### 一句话记住它
V\* 把语言模型的世界知识变成视觉查询，让多模态大语言模型像人一样主动搜索图像细节，从而在高分辨率场景中实现精准推理。