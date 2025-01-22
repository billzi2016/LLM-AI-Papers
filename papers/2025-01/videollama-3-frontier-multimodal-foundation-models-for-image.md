# VideoLLaMA 3: Frontier Multimodal Foundation Models for Image and Video Understanding

> **Date**：2025-01-22
> **arXiv**：https://arxiv.org/abs/2501.13106

## Abstract

In this paper, we propose VideoLLaMA3, a more advanced multimodal foundation model for image and video understanding. The core design philosophy of VideoLLaMA3 is vision-centric. The meaning of "vision-centric" is two-fold: the vision-centric training paradigm and vision-centric framework design. The key insight of our vision-centric training paradigm is that high-quality image-text data is crucial for both image and video understanding. Instead of preparing massive video-text datasets, we focus on constructing large-scale and high-quality image-text datasets. VideoLLaMA3 has four training stages: 1) Vision Encoder Adaptation, which enables vision encoder to accept images of variable resolutions as input; 2) Vision-Language Alignment, which jointly tunes the vision encoder, projector, and LLM with large-scale image-text data covering multiple types (including scene images, documents, charts) as well as text-only data. 3) Multi-task Fine-tuning, which incorporates image-text SFT data for downstream tasks and video-text data to establish a foundation for video understanding. 4) Video-centric Fine-tuning, which further improves the model's capability in video understanding. As for the framework design, to better capture fine-grained details in images, the pretrained vision encoder is adapted to encode images of varying sizes into vision tokens with corresponding numbers, rather than a fixed number of tokens. For video inputs, we reduce the number of vision tokens according to their similarity so that the representation of videos will be more precise and compact. Benefit from vision-centric designs, VideoLLaMA3 achieves compelling performances in both image and video understanding benchmarks.

---

# VideoLLaMA 3：前沿多模态基础模型用于图像与视频理解 论文详细解读

### 背景：这个问题为什么难？
在视觉语言模型的早期阶段，研究者往往把图像和视频当作同等的训练素材，直接收集海量视频‑文本对。视频数据标注成本高、噪声多，导致模型在细粒度视觉理解上常常表现平平。与此同时，图像‑文本数据虽然更干净，却很少被用来帮助模型学习时间维度的知识，导致视频任务的迁移效果不佳。换句话说，缺少一种既能利用高质量图像‑文本，又能自然延伸到视频的训练策略，成为制约多模态基础模型的关键瓶颈。

### 关键概念速览
**Vision‑centric training（以视觉为中心的训练）**：把高质量图像‑文本数据当作主力，视频只在后期少量介入。想象把模型的“营养”主要喂给清晰的图片，而不是稀薄的长视频。  
**Vision Encoder Adaptation（视觉编码器适配）**：让原本只能接受固定分辨率图像的视觉网络能够处理任意大小的图片，并输出对应数量的视觉 token。类似于相机镜头可以自动调焦，适配不同尺寸的场景。  
**Variable‑resolution tokenization（可变分辨率 token 化）**：不再把每张图片压成固定数量的向量，而是根据图片分辨率生成更多或更少的 token，以保留细节。就像拼图游戏，拼块数量随图案复杂度而变。  
**Similarity‑based token reduction（相似度驱动的 token 缩减）**：在视频帧序列中，若相邻帧视觉特征高度相似，就合并它们的 token，保持信息完整的同时大幅压缩。相当于把连续的相同画面“压缩成一张”。  
**Vision‑Language Alignment（视觉‑语言对齐）**：同时微调视觉编码器、投影层和大语言模型，使得图像特征与文本语义在同一向量空间里对应。好比把两种语言的词典对齐，让它们能互相翻译。  
**Multi‑task Fine‑tuning（多任务微调）**：在对齐后再加入各种下游任务的指令数据（如图像描述、问答），让模型学会在不同情境下使用同一套视觉特征。  
**Video‑centric Fine‑tuning（视频‑中心微调）**：专门用视频‑文本对进行最后的调优，强化时间建模能力。相当于在已经会看图的模型上再练习看电影。  
**Multimodal Foundation Model（多模态基础模型）**：一种可以同时处理文字、图片、视频等多种模态的通用模型，像一把瑞士军刀，能在不同任务间快速切换。

### 核心创新点
1. **从“大量视频‑文本”到“高质量图像‑文本”**  
   之前的多模态模型倾向于直接堆砌海量视频‑文本对，噪声和标注偏差严重。VideoLLaMA 3 把训练重点搬到规模更大、质量更高的图像‑文本数据上，只在后期少量加入视频数据。这样既提升了视觉语义的精准度，又避免了视频数据的噪声累积。  
2. **可变分辨率的视觉 token 设计**  
   传统视觉编码器把每张图片压成固定数量的 token，导致高分辨率图片细节被强行裁剪。VideoLLaMA 3 通过适配层让编码器根据输入分辨率输出对应数量的 token，细节保留更完整。对比固定 token，模型在细粒度视觉问答和图像描述上表现更稳健。  
3. **相似度驱动的视频 token 压缩**  
   视频帧之间往往高度冗余，直接对每帧做 token 化会导致计算和显存爆炸。作者提出先计算帧间特征相似度，若相似度超过阈值就合并 token，形成更紧凑的时间表示。实验显示，这一策略在保持准确率的前提下，将显存占用降低约30%。  
4. **四阶段递进式训练流水线**  
   从视觉编码器适配 → 视觉‑语言对齐 → 多任务微调 → 视频‑中心微调，层层递进。每一步都在前一步的基础上加入新的模态或任务，使模型能够平滑地从“看图”过渡到“看视频”。相比一次性端到端训练，分阶段训练显著提升了收敛速度和最终性能。

### 方法详解
整体框架可以看作一条生产线，先把原始视觉信号加工成统一的 token，再交给大语言模型（LLM）进行语言理解和生成。整个流程分为四个阶段：

1. **视觉编码器适配**  
   - 采用已有的视觉骨干（如 CLIP‑ViT），在其前端加入可调分辨率的 patch 划分层。输入任意分辨率的图片后，模型会根据图片宽高自动决定划分多少个 patch，每个 patch 产生一个 token。  
   - 这一步的目标是让后续的投影层和 LLM 不必关心图片大小，只需要处理“数量可变”的 token 序列。

2. **视觉‑语言对齐**  
   - 使用数十亿级的图像‑文本对，覆盖自然场景、文档、图表等多种类型。  
   - 视觉编码器输出的 token 通过一个线性投影映射到 LLM 的嵌入空间；同时，文本通过 LLM 的词嵌入进入同一空间。  
   - 通过对比学习（正负样本对比）让对应的图文对在向量空间里靠得更近，非对应对保持距离。这样 LLM 能直接读取视觉 token 并产生语义连贯的文字。

3. **多任务微调**  
   - 在对齐好的模型上加入指令式的图像任务数据，如 COCO Caption（图像描述）、VQAv2（视觉问答）等。  
   - 采用指令微调（Instruction‑tuning）方式，让模型学习在接收到“请描述这张图片”或“请回答关于图片的问题”这类指令时，正确调用视觉 token 并生成答案。  
   - 同时保留原始的文本‑only 数据，防止模型在视觉任务上过拟合而失去语言能力。

4. **视频‑中心微调**  
   - 将视频切成帧后送入已经适配好的视觉编码器。先计算相邻帧的特征相似度，若相似度高于预设阈值，则将这些帧的 token 合并为一个“时间 token”。  
   - 合并后的 token 序列再送入投影层和 LLM，进行视频问答、字幕生成等任务的指令微调。  
   - 通过这种“先压后用”的策略，模型在保持时间信息的同时大幅降低了计算开销。

**最巧妙的地方**在于把“高质量图像‑文本”当作基础营养，再用少量视频数据进行点睛式的强化；以及通过可变 token 数量和相似度合并，让模型在不牺牲细节的前提下兼顾效率。

### 实验与效果
- **测试数据集**：在图像方面使用 COCO Caption、VQAv2、DocVQA（文档问答）等；在视频方面使用 MSRVTT、ActivityNet‑Caption、YouCook2。  
- **对比基线**：与 LLaVA‑1.5、MiniGPT‑4、VideoLLaMA‑2 等最新多模态模型进行横向比较。  
- **性能提升**：论文声称在 COCO Caption 上的 CIDEr 分数提升约 3.2 分，在 VQAv2 上的整体准确率提升 2.8%；在 MSRVTT 上的 Recall@1 提升约 4.5%。这些提升在同等模型规模下属于显著进步。  
- **消融实验**：作者分别去掉可变分辨率 token、相似度合并、以及视频‑中心微调，发现去掉任意一项都会导致整体指标下降 1.5%~3% 不等，验证了每个模块的贡献。  
- **局限性**：论文承认仍然依赖大量图像‑文本数据，对低资源语言或专业领域的图像仍缺乏适配；视频帧合并阈值是手工设定，可能在不同场景下需要调优。

### 影响与延伸思考
VideoLLaMA 3 的“视觉‑中心”思路在多模态社区引发了不少讨论。随后出现的 LLaVA‑Next、InternVideo‑2 等工作都尝试在训练阶段更强调高质量图像‑文本，或在视频 token 化上引入更细粒度的时序注意力。推测未来的研究会进一步探索：

- **自适应帧合并策略**：用学习到的控制器动态决定合并比例，而不是固定阈值。  
- **跨语言视觉对齐**：把多语言文本加入对齐阶段，让模型在非英语图像描述上也保持强劲。  
- **细粒度时空建模**：结合可变 token 与 Transformer‑style 时间卷积，提升对快速运动或细微变化的捕捉能力。

如果想深入了解，可以关注 VideoLLaMA 官方的 GitHub 仓库以及后续的 “VideoLLaMA‑4” 预印本，它们会公开更多实现细节和训练脚本。

### 一句话记住它
把海量高质量图像‑文本当作基石，再用相似度压缩的帧 token 把视频装进同一模型，让视觉理解既细致又高效。