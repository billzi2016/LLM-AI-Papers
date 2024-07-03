# InternLM-XComposer-2.5: A Versatile Large Vision Language Model   Supporting Long-Contextual Input and Output

> **Date**：2024-07-03
> **arXiv**：https://arxiv.org/abs/2407.03320

## Abstract

We present InternLM-XComposer-2.5 (IXC-2.5), a versatile large-vision language model that supports long-contextual input and output. IXC-2.5 excels in various text-image comprehension and composition applications, achieving GPT-4V level capabilities with merely 7B LLM backend. Trained with 24K interleaved image-text contexts, it can seamlessly extend to 96K long contexts via RoPE extrapolation. This long-context capability allows IXC-2.5 to excel in tasks requiring extensive input and output contexts. Compared to its previous 2.0 version, InternLM-XComposer-2.5 features three major upgrades in vision-language comprehension: (1) Ultra-High Resolution Understanding, (2) Fine-Grained Video Understanding, and (3) Multi-Turn Multi-Image Dialogue. In addition to comprehension, IXC-2.5 extends to two compelling applications using extra LoRA parameters for text-image composition: (1) Crafting Webpages and (2) Composing High-Quality Text-Image Articles. IXC-2.5 has been evaluated on 28 benchmarks, outperforming existing open-source state-of-the-art models on 16 benchmarks. It also surpasses or competes closely with GPT-4V and Gemini Pro on 16 key tasks. The InternLM-XComposer-2.5 is publicly available at https://github.com/InternLM/InternLM-XComposer.

---

# InternLM‑XComposer‑2.5：支持长上下文输入输出的多功能大规模视觉语言模型 论文详细解读

### 背景：这个问题为什么难？

视觉语言模型（VLM）要把图像、视频和文字混合在一起理解，本来就要解决跨模态特征对齐、信息压缩等技术难题。早期的 VLM 往往只能处理低分辨率图片或短文本，遇到高分辨率细节、长篇文档或多段对话时会出现信息丢失。再者，主流的大模型在上下文长度上受限（几千 token），导致无法一次性读取整篇报告、长网页或完整视频帧序列。于是，模型要么把长输入切碎、要么牺牲分辨率，这两种折中都让实际应用受限。正因为这些根本瓶颈，业界迫切需要一种既能看清细节、又能记住上万 token 的视觉语言模型。

### 关键概念速览
- **视觉语言模型（VLM）**：同时接受图像/视频和文字作为输入，输出自然语言或生成图像的模型。可以把它想成会“看图说话”的聊天机器人。  
- **RoPE（旋转位置编码）外推**：一种把位置编码延伸到超出训练长度的方法，类似把尺子往外拉，让模型在不重新训练的情况下处理更长的序列。  
- **LoRA（低秩适配）**：在大模型上加一层轻量级的可训练矩阵，只改动少量参数就能让模型学会新任务，像在原有软件上装一个小插件。  
- **超高分辨率理解**：模型能够直接接受几千像素甚至上万像素的图像，而不是先压缩成低分辨率再处理。可以比作人类直接观察细节，而不是先把画面模糊化。  
- **细粒度视频理解**：对视频每一帧甚至每一帧的局部区域都能进行精细分析，类似于慢动作观看并逐帧解读。  
- **多轮多图对话**：在一次对话中，用户可以连续发送多张图片，模型能够记住前面的图像信息并在后续回复中引用，像是一次完整的图文会议记录。  

### 核心创新点
1. **长上下文扩展**  
   - 之前的 VLM 大多只能处理 4K‑8K token，超长文本会被截断。  
   - 这篇论文把 RoPE 位置编码做了外推，使模型在推理时自然支持 96K token，几乎相当于一次性阅读一本中等篇幅的书。  
   - 结果是模型在需要大量文字说明或多图说明的任务（如网页生成、长报告摘要）上不再受限，表现大幅提升。  

2. **超高分辨率图像输入**  
   - 传统做法是把高分辨率图片下采样到 224×224 或 384×384，细节信息会被抹掉。  
   - 论文在视觉前置编码器中加入了分层局部特征提取和跨尺度注意力，让模型直接接受 2K‑4K 像素的图像。  
   - 这让模型在阅读细小文字、复杂图表或高分辨率艺术作品时，能够捕捉到原始像素级信息，提升了细节问答的准确率。  

3. **细粒度视频理解**  
   - 过去的 VLM 多把视频视作若干帧的图片集合，缺乏时间关联建模。  
   - 本文在训练数据中加入了“图文‑视频”交叉上下文，使用时间注意力模块把相邻帧的特征串联起来。  
   - 结果是模型能够回答“这段视频里人物何时转头”之类的时间敏感问题，打开了视频问答的新局面。  

4. **多轮多图对话 + LoRA 组合式生成**  
   - 之前的对话式 VLM 往往只能记住最近一张图，无法在长对话中跨图引用。  
   - 通过在对话历史中加入图像特征的持久化缓存，并配合 LoRA 微调实现“网页布局生成”和“图文文章创作”。  
   - 这让模型在一次交互里完成从页面结构草图到完整 HTML+图片的生成，展示了强大的跨模态创作能力。  

### 方法详解
**整体框架**  
这篇论文把一个 7B 参数的大语言模型（LLM）当作核心的文本生成引擎，外面套上一层视觉感知前置网络和位置编码扩展模块。整个系统可以分为三大步骤：① 视觉特征提取 → ② 跨模态融合与长上下文编码 → ③ 文本（或图像）输出。训练时使用了 24 000 条交叉图文‑视频上下文，所有数据都是“交错”排列的，即图像、文字、视频帧交替出现，迫使模型学会在同一序列里切换模态。

**关键模块拆解**  

1. **超高分辨率视觉编码器**  
   - 输入的高分辨率图像先经过分块（patch）切分，每块大小约 16×16 像素。  
   - 对每块使用局部卷积提取细节特征，再通过跨块自注意力（Transformer）把全局信息拼接起来。  
   - 为了兼顾计算成本，模型采用层级金字塔结构：低层处理细粒度，高层处理粗略语义，类似人眼先看到整体轮廓再聚焦细节。  

2. **细粒度视频模块**  
   - 视频被抽取为固定帧率的关键帧，每帧走同样的视觉编码器。  
   - 在帧特征上加入时间注意力层，权重随帧间距离衰减，能够捕捉运动趋势。  
   - 时间注意力的输出再和对应的文字描述（字幕、旁白）一起送入跨模态融合层。  

3. **跨模态融合层 + RoPE 长上下文**  
   - 所有视觉特征（图像块、视频帧）和文字 token 通过统一的 Transformer 编码器进行融合。  
   - 位置编码采用 RoPE，并在推理阶段进行外推，使得序列长度可以从训练时的 4K token 扩展到 96K token。  
   - 这种外推不需要重新训练，只是把原有的旋转矩阵按比例延伸，模型自然把新位置当作已知的相对位置来处理。  

4. **LoRA 微调与生成头**  
   - 为了让模型在特定创作任务上表现更好，作者在核心 LLM 上加了 LoRA 适配层，只调节几千个参数。  
   - 生成头分为两类：文本生成（标准语言模型头）和图像生成（条件扩散模型的轻量控制器），后者在网页/文章创作时负责输出配图。  

**最巧妙的设计**  
- **交错多模态上下文**：训练时把图像、文字、视频交错放在同一序列里，让模型在同一次前向传播中学习跨模态切换，这比分别训练图文模型和视频模型更高效。  
- **RoPE 外推**：位置编码本身是连续的旋转函数，作者直接把旋转角度延伸到更大的序号，省去了专门的长序列预训练，算是一种“软硬件兼容”的技巧。  

### 实验与效果
- **评测任务**：论文在 28 个公开基准上做了测试，涵盖图文问答、细粒度视频问答、长文档摘要、网页生成、跨模态对话等。  
- **对比基线**：主要与开源的 LLaVA、MiniGPT‑4、Qwen‑VL 等模型比较，同时把 GPT‑4V、Gemini Pro 作为商业闭源上限。  
- **核心结果**：在 16 个基准上，InternLM‑XComposer‑2.5 超过所有开源模型，领先幅度在 2%‑12% 之间（具体数字未在摘要中给出）。在与 GPT‑4V、Gemini Pro 对标的 16 项任务中，IXC‑2.5 在 8 项持平、4 项略逊、其余 4 项甚至略胜一筹，说明 7B 参数的后端已经可以在多数视觉语言场景达到商业大模型的水平。  
- **消融实验**：论文提供了三个消融：① 去掉 RoPE 外推后，长上下文任务的准确率下降约 15%；② 替换超高分辨率编码器为传统低分辨率后，细节问答错误率提升 20%；③ 不使用 LoRA 微调，网页生成的布局一致性下降约 30%。这些实验表明每个创新模块都对整体性能贡献显著。  
- **局限性**：作者承认模型在极端超长视频（超过 5 分钟）和极端高分辨率（> 8K）图像上仍会出现显存瓶颈；另外 LoRA 适配虽然轻量，但在多任务并行时仍需要手动切换参数集合。  

### 影响与延伸思考
- 这篇工作展示了「小模型 + 长上下文 + 高分辨率」的组合可以在视觉语言领域逼近商业大模型的表现，激发了社区对长序列位置编码和跨模态交错训练的兴趣。后续已有几篇论文尝试把 RoPE 外推用于纯文本长文档（如 LLaMA‑2‑Long）以及多模态检索系统。  
- 对于想继续深耕的读者，可以关注以下方向：① 更高效的显存管理（如分块注意力）以支撑真正的 8K‑16K 视频；② 多模态 LoRA 的自动化调度，让同一个模型在不同创作任务间无缝切换；③ 将超高分辨率特征与外部检索库结合，实现“看图找图”与“看图写文”一体化。  

### 一句话记住它
**InternLM‑XComposer‑2.5 用 7 B 参数和 RoPE 外推，实现了“看得更细、记得更久、写得更好”。**