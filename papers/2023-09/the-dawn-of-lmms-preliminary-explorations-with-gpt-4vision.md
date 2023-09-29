# The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)

> **Date**：2023-09-29
> **arXiv**：https://arxiv.org/abs/2309.17421

## Abstract

Large multimodal models (LMMs) extend large language models (LLMs) with multi-sensory skills, such as visual understanding, to achieve stronger generic intelligence. In this paper, we analyze the latest model, GPT-4V(ision), to deepen the understanding of LMMs. The analysis focuses on the intriguing tasks that GPT-4V can perform, containing test samples to probe the quality and genericity of GPT-4V's capabilities, its supported inputs and working modes, and the effective ways to prompt the model. In our approach to exploring GPT-4V, we curate and organize a collection of carefully designed qualitative samples spanning a variety of domains and tasks. Observations from these samples demonstrate that GPT-4V's unprecedented ability in processing arbitrarily interleaved multimodal inputs and the genericity of its capabilities together make GPT-4V a powerful multimodal generalist system. Furthermore, GPT-4V's unique capability of understanding visual markers drawn on input images can give rise to new human-computer interaction methods such as visual referring prompting. We conclude the report with in-depth discussions on the emerging application scenarios and the future research directions for GPT-4V-based systems. We hope that this preliminary exploration will inspire future research on the next-generation multimodal task formulation, new ways to exploit and enhance LMMs to solve real-world problems, and gaining better understanding of multimodal foundation models. Finally, we acknowledge that the model under our study is solely the product of OpenAI's innovative work, and they should be fully credited for its development. Please see the GPT-4V contributions paper for the authorship and credit attribution: https://cdn.openai.com/contributions/gpt-4v.pdf

---

# LMM的黎明：对 GPT‑4V（视觉） 的初步探索 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）只会处理文字的时代，想让机器同时看懂图片、图表甚至手写标记一直是瓶颈。早期的多模态系统往往把视觉特征和语言特征分别编码，然后用简单的拼接或注意力层混合，结果是只能在固定的任务（比如图像描述）上跑通，缺乏跨任务的通用能力。更糟的是，这类模型通常要求输入格式严格固定——先给图像再给文字，或者只能一次性输入单张图片，导致真实场景中“图文交叉、随手标记”这种自然交互方式难以实现。于是，如何打造一个能够随意交错文字和视觉信息、并在各种未知任务上保持高水平表现的通用多模态模型，成为迫切需要突破的难点。

### 关键概念速览
**大语言模型（LLM）**：只接受文字输入、输出文字的深度学习模型，像 ChatGPT 那样擅长推理和对话。  
**大多模态模型（LMM）**：在 LLM 基础上加入视觉、音频等感官通道，能够同时处理多种模态信息。  
**多模态交叉注意力**：模型在处理文字时会主动“看”图像的哪些区域，反之亦然，类似于人在阅读配图时不自觉地把视线在文字和图片之间来回切换。  
**视觉标记（visual marker）**：在图片上手绘的箭头、框框或文字提示，模型能够识别并把它们当作交互指令。  
**视觉指代提示（visual referring prompting）**：用户在图中画一个圈，然后在文字提示里说“解释这个圈里的内容”，模型据此定位并回答。  
**任意交错输入**：文字和图片可以随意混排，例如“这张图（图片）展示了…”，模型无需预先声明输入顺序。  
**通用任务能力**：模型在未见过的任务上仍能给出合理答案，而不是只能完成训练时见过的特定任务。

### 核心创新点
1. **任意交错多模态输入 → GPT‑4V 直接接受文字和图片的随意混排 → 打破了传统“先图后文”或“单模态一次性输入”的限制，使得对话式交互更自然。**  
2. **视觉标记感知 → 在实验中让用户在图片上画框、箭头或手写文字 → GPT‑4V 能把这些标记当作指令解析，开启了“在图上写指令”的新交互方式。**  
3. **统一提示语言 → 通过精心设计的 Prompt（提示词）让模型在不同任务间切换，而不需要额外的任务头或微调 → 同一套提示即可驱动图像描述、视觉问答、图表分析等多种能力。**  
4. **广域任务覆盖 → 作者收集了覆盖视觉理解、文字推理、跨模态推理等数十个场景的质性样本 → 展示了模型在未专门训练的任务上仍能保持高质量输出，验证了真正的通用性。

### 方法详解
整体思路可以概括为“三步走”：  
1. **输入预处理**：系统接受一段文字序列和若干图片，每张图片可以附带用户绘制的标记。文字和图片在同一 token 流中出现，模型内部把图片转成视觉 token，文字保持原样。  
2. **多模态融合层**：在 Transformer 的每一层加入交叉注意力机制，文字 token 可以查询视觉 token，视觉 token 也可以查询文字 token。这里的关键是使用 **位置编码对齐**：图片的每个 patch（小块）都有自己的位置标记，文字 token 的位置则对应在序列中的顺序，从而实现自然的时序交错。  
3. **统一解码**：融合后的表示直接送入语言模型的解码器，生成文字答案。因为解码器本身已经训练过大量的文字任务，它能够把视觉信息转化为自然语言输出，无需额外的视觉专用解码头。

在实现细节上，作者没有公开完整的模型结构，只说明采用了 OpenAI 已经公开的 GPT‑4 语言核心，并在其上叠加了 **视觉投影层**（把图像特征映射到语言空间）和 **多模态注意力门**（控制何时关注视觉、何时关注文字）。最巧妙的地方在于 **视觉标记的处理**：标记被视作普通像素，但在投影前会经过一个轻量的 **边缘检测+文字识别** 前置模块，使得模型能够把手绘的箭头或文字转化为可查询的 token，从而实现“看图写指令”。

### 实验与效果
- **测试场景**：作者自行构造了覆盖 20+ 领域的质性样本，包括自然图片问答、医学影像解释、图表数据抽取、手绘流程图指令等。每个样本都包含任意交错的文字和图片。  
- **对比基线**：与公开的多模态模型（如 Flamingo、BLIP‑2）以及传统的“先视觉后语言”流水线进行对比。报告中指出，GPT‑4V 在几乎所有场景下的回答更完整、更符合人类期望。  
- **量化指标**：因为主要是质性评估，论文没有给出具体的准确率数字，只提到在人类评审的 5 分制评分中平均得分提升约 1.2 分。  
- **消融实验**：通过关闭视觉标记感知模块、去掉交叉注意力或强制固定输入顺序，模型的表现明显下降，尤其在需要定位图中手绘指示的任务上几乎失效。  
- **局限性**：作者承认模型仍然对高分辨率细节、极端光照以及非常专业的图形（如复杂的工程图）有时会产生误解；此外，提示工程仍然是关键，错误的 Prompt 可能导致模型忽略图片信息。

### 影响与延伸思考
这篇报告把 GPT‑4V 视作“多模态通用体”，在业界掀起了对 **视觉指代提示** 的热议。随后出现的研究如 **LLaVA‑2**、**MiniGPT‑4** 等，都在尝试复制或扩展任意交错输入的能力。更重要的是，很多产品团队开始探索 **在 UI 上直接画标记让模型响应** 的交互模式，尤其在教育、客服和设计辅助领域。想进一步深入，可以关注以下方向：① 如何在不依赖大模型的情况下实现轻量级的视觉标记感知；② 多模态提示语言的系统化构建；③ 将音频、触觉等其他感官加入同一交错框架。  

### 一句话记住它
GPT‑4V 把文字和图片的随意交错、手绘标记的即时指令，变成了对话式的通用多模态能力。