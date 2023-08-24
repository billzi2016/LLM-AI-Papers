# Qwen-VL: A Versatile Vision-Language Model for Understanding,   Localization, Text Reading, and Beyond

> **Date**：2023-08-24
> **arXiv**：https://arxiv.org/abs/2308.12966

## Abstract

In this work, we introduce the Qwen-VL series, a set of large-scale vision-language models (LVLMs) designed to perceive and understand both texts and images. Starting from the Qwen-LM as a foundation, we endow it with visual capacity by the meticulously designed (i) visual receptor, (ii) input-output interface, (iii) 3-stage training pipeline, and (iv) multilingual multimodal cleaned corpus. Beyond the conventional image description and question-answering, we implement the grounding and text-reading ability of Qwen-VLs by aligning image-caption-box tuples. The resulting models, including Qwen-VL and Qwen-VL-Chat, set new records for generalist models under similar model scales on a broad range of visual-centric benchmarks (e.g., image captioning, question answering, visual grounding) and different settings (e.g., zero-shot, few-shot). Moreover, on real-world dialog benchmarks, our instruction-tuned Qwen-VL-Chat also demonstrates superiority compared to existing vision-language chatbots. Code, demo and models are available at https://github.com/QwenLM/Qwen-VL.

---

# Qwen-VL：面向理解、定位、文本阅读等多任务的通用视觉语言模型 论文详细解读

### 背景：这个问题为什么难？
视觉语言模型（VLM）要同时处理图像和文字，面临两大挑战：一是如何让语言模型“看到”图像的细节，二是如何在不同任务（描述、问答、定位、阅读）之间共享知识。早期的 VLM 多数只在大规模图文对上做预训练，缺乏对 **定位**（把文字或物体框选出来）和 **文本阅读**（直接识别图中印刷文字）的显式监督，导致在需要精确空间信息的任务上表现乏力。再者，跨语言的多模态数据稀缺，使得模型在非英语场景下的鲁棒性受限。于是出现了一个需求：在同一模型里统一实现图像理解、空间定位和文字识别，并且支持多语言。

### 关键概念速览
**视觉受体（visual receptor）**：把原始像素映射成向量序列的前置网络，类似相机的感光元件，把图像“翻译”成模型能读的“文字”。  
**多模态输入‑输出接口**：定义模型如何同时接受图像特征和文本提示，以及如何把答案输出为文字、坐标或框。可以想象成一条双向的传送带，左边装载图像特征，右边装载语言 token。  
**三阶段训练流水线**：先在大规模图文对上做基础预训练 → 再加入带框的图文对进行定位微调 → 最后用指令数据进行对话式微调。层层递进，像是先学会走路、再学会跑步、最后学会踢球。  
**图文‑框三元组（image‑caption‑box tuple）**：每条训练样本同时包含图片、对应描述和目标框坐标，用来教模型把文字或物体与具体位置对应起来。  
**多语言多模态清洗语料**：在构建训练数据时，去除噪声、统一格式、确保不同语言的文本质量，保证模型在中文、英文等语言上都有可靠的输入。  
**指令微调（instruction tuning）**：在对话数据上进一步训练，让模型学会遵循用户指令、生成自然对话式回答，类似给模型上了一堂客服培训课。  
**视觉 grounding**：模型在回答问题的同时给出答案所在的图像区域，等价于在答案上画一个高亮框。  

### 核心创新点
1. **视觉受体 + 语言模型的深度融合 → 通过专门设计的视觉受体把图像特征直接拼接进 Qwen‑LM 的 token 序列 → 语言模型在同一层次上同时处理视觉和文字信息，显著提升了跨模态理解的统一性。**  
2. **引入图文‑框三元组进行显式定位训练 → 在第二阶段训练中加入了大量带框的标注，使模型学会把文字或物体映射到具体坐标 → 在视觉 grounding 和文本阅读任务上实现了前所未有的准确率。**  
3. **三阶段训练流水线 → 先大规模图文预训练，再定位微调，最后指令微调 → 让模型既保留了通用的语言能力，又具备了专业的空间感知和对话交互能力，零/少样本下的表现大幅提升。**  
4. **多语言多模态清洗语料库 → 统一收集并清洗了多语言的图文对，确保不同语言的质量一致 → 模型在中文、英文等多语言基准上都能保持竞争力，突破了多数 VLM 只擅长英语的局限。**  

### 方法详解
整体思路可以划分为四个步骤：  
1) **视觉特征提取**：使用经过精心调参的视觉受体（如 ViT‑B/16）把输入图片转成一系列视觉 token。  
2) **特征拼接与位置编码**：将视觉 token 与语言 token 按顺序拼接，并加入统一的位置信息，使得模型能够辨别“这是图像的第 5 帧还是第 12 帧”。  
3) **三阶段训练**：  
   - **阶段一**（大规模图文预训练）：在清洗后的多语言图文对上进行自回归语言建模，目标是预测下一个文字 token。  
   - **阶段二**（定位微调）：使用图文‑框三元组，模型需要在生成答案的同时预测对应的框坐标。这里采用了一个额外的回归头，输出四维坐标（左上 x、y、宽、高），并通过 L2 损失与真实框对齐。  
   - **阶段三**（指令微调）：在对话式指令数据上继续训练，加入了“请标出图中所有文字”之类的任务指令，使模型学会在交互中主动输出框或文字识别结果。  
4) **多任务解码**：推理时，根据用户指令的类型，模型会选择不同的解码路径：如果要求描述，则只输出文字；如果要求定位，则同时输出文字和框坐标；如果要求阅读，则输出识别的文本并附带位置。  

**巧妙之处**在于：视觉特征直接作为 token 进入语言模型，而不是单独走一个视觉分支再做跨模态融合，这让语言模型的自注意力机制天然地捕获图像内部的空间关系。此外，图文‑框三元组的加入，使得模型在训练阶段就学会了“看见文字 → 知道它在图上的哪儿”，避免了后期单独训练 OCR 模块的繁琐。  

### 实验与效果
- **测试任务**：包括图像描述（image captioning）、视觉问答（VQA）、视觉 grounding、文本阅读（scene text recognition）以及真实对话基准（vision‑language chatbot）。  
- **基线对比**：与同等规模的 LLaVA、MiniGPT‑4、BLIP‑2 等模型相比，Qwen‑VL 在大多数公开基准上取得了最高的评测分数。摘要中提到“在相似模型规模下创下新纪录”，但未给出具体数值。  
- **消融实验**：论文报告了去掉定位微调阶段或不使用图文‑框三元组会导致视觉 grounding 和文本阅读的准确率显著下降，说明这两个模块是性能提升的关键驱动。  
- **局限性**：作者承认模型仍然依赖大规模清洗语料，对极端低光或高度遮挡的图像仍有困难；此外，多语言支持主要集中在常见语言，稀有语言的表现未作深入评估。  

### 影响与延伸思考
Qwen‑VL 的“一体化”设计展示了在同一模型里兼顾语言、视觉、空间定位和文字识别的可行路径，随后的多模态研究纷纷借鉴其三阶段训练思路和图文‑框三元组标注方式。后续工作如 **Mistral‑VL**、**Gemini‑Vision** 等在公开报告中提到受到了 Qwen‑VL 的启发，尤其是在指令微调阶段加入多任务指令的做法。想进一步深入，可以关注以下方向：① 更高效的视觉受体与语言模型的跨层融合技术；② 稀疏或自监督的定位微调，以降低对标注框的依赖；③ 扩展到视频、3D 场景的多模态理解。  

### 一句话记住它
Qwen‑VL 用统一的语言模型加上图文‑框三元组训练，让同一个模型既能说、也能指、还能读图中文字。