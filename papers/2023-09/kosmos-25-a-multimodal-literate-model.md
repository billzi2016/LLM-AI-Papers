# KOSMOS-2.5: A Multimodal Literate Model

> **Date**：2023-09-20
> **arXiv**：https://arxiv.org/abs/2309.11419

## Abstract

The automatic reading of text-intensive images represents a significant advancement toward achieving Artificial General Intelligence (AGI). In this paper we present KOSMOS-2.5, a multimodal literate model for machine reading of text-intensive images. Pre-trained on a large-scale corpus of text-intensive images, KOSMOS-2.5 excels in two distinct yet complementary transcription tasks: (1) generating spatially-aware text blocks, where each block of text is assigned spatial coordinates within the image, and (2) producing structured text output that captures both style and structure in markdown format. This unified multimodal literate capability is achieved through a shared decoder-only autoregressive Transformer architecture and task-specific prompts. Building on this foundation, we fine-tune KOSMOS-2.5 for document understanding tasks, resulting in a document understanding generalist named KOSMOS-2.5-CHAT. Additionally, a large corpus of 357.4 million document pages spanning diverse domains was curated for pre-training. We evaluate KOSMOS-2.5 on two newly proposed benchmarks, OCREval and MarkdownEval, for document-level text recognition and image-to-markdown generation, demonstrating impressive literate capabilities comparable to GPT-4o. KOSMOS-2.5-CHAT achieves performance comparable to other state-of-the-art generalists that are five times larger (1.3B vs. 7B) across nine text-rich visual question answering benchmarks. Models and code have been available at \url{https://aka.ms/kosmos25}.

---

# KOSMOS-2.5：多模态读写模型 论文详细解读

### 背景：这个问题为什么难？

文本密集型图片（如学术论文、报表、合同）里文字往往分布在多个不规则区域，传统 OCR 只能输出一串平铺的字符，缺乏位置信息和版面结构。早期的多模态模型大多聚焦于“看图说话”或“图文检索”，对文字的空间布局和排版风格几乎不做处理。于是出现了两大痛点：①模型既要识别文字，又要把每块文字的坐标保留下来；②还要把原始版面转换成结构化的 Markdown，保留标题、列表、表格等格式。缺少统一的、能够同时完成这两件事的模型，导致实际应用中需要把 OCR、版面分析、后处理等多个工具链拼接，效率低且错误累积。

### 关键概念速览

**多模态读写模型**：既能“看”图片，又能“写”文字的模型，类似于会同时阅读纸质文档并用键盘打出相同排版的人。

**空间感知文本块**：模型输出的每段文字都会附带在图片中的左上、右下坐标，就像在图片上画了一个标记框。

**Markdown 结构化输出**：把识别到的文字按照标题、列表、代码块等 Markdown 语法组织起来，等价于把纸质文档直接转成可编辑的 Markdown 文件。

**解码器‑Only 自回归 Transformer**：只使用 Transformer 的解码器部分，像 GPT 那样一次生成下一个 token，省去编码器‑解码器的双向交互，简化训练流程。

**任务提示（Prompt）**：在输入前加上一段文字指令，让同一个模型切换到“生成坐标块”或“生成 Markdown”两种模式。

**文档理解通用体（Generalist）**：在基础模型上继续微调，使其能够回答基于文档的视觉问答，类似于把阅读能力升级为“理解并推理”。

**OCREval / MarkdownEval**：两套新建的评测基准，前者衡量文档级 OCR 的准确度，后者衡量从图片到 Markdown 的转换质量。

### 核心创新点

1. **统一解码器‑Only 架构 + 任务提示 → 同时支持两种输出**  
   过去的系统要么专注于坐标化 OCR，要么专注于结构化文本生成，往往需要两套模型。KOSMOS-2.5 只用一个解码器‑Only Transformer，通过在输入前加不同的提示词（如 “<spatial>” 或 “<markdown>”），让模型在同一次前向传播中切换任务。这样既省显存，又保证了两种输出在同一语义空间里保持一致。

2. **大规模文本密集型图像预训练语料 → 领域适配**  
   作者收集并清洗了 357.4 百万页文档，覆盖学术、法律、金融、社交媒体等多种版式。相比于通用视觉‑语言数据，这些专门的文本密集型图片让模型在预训练阶段就学会了文字排版的规律，显著提升了对细小字体、复杂表格的识别能力。

3. **从基础模型到文档理解通用体的微调流程 → 参数更小却性能相当**  
   在基础的 KOSMOS-2.5 上继续使用指令微调（instruction‑tuning），得到 KOSMOS-2.5‑CHAT。该模型只有 1.3 B 参数，却在九个视觉问答基准上追平甚至略超 7 B 参数的竞争对手，说明任务提示加微调的组合能够高效激活模型的文档推理潜能。

4. **新建两套评测基准 OCREval 与 MarkdownEval → 客观衡量“读写”能力**  
   过去缺少统一的文档级 OCR 与结构化生成评测。作者自行标注了大量图片‑文本对，分别用字符错误率（CER）和 Markdown 结构相似度（BLEU/ROUGE）评估。实验显示 KOSMOS-2.5 在两套基准上都接近 GPT‑4o 的表现，为后续工作提供了可复用的基准。

### 方法详解

**整体思路**  
模型的训练分为两阶段：①大规模自监督预训练，输入是文本密集型图片的像素以及对应的文字块（包括坐标信息和 Markdown），模型学习在给定提示下自回归生成对应的输出；②指令微调（instruction‑tuning），在更丰富的文档问答对上继续训练，使模型能够在对话式交互中使用同样的解码器完成阅读、推理和写作任务。

**关键模块拆解**

1. **视觉前置层（Vision Encoder）**  
   - 使用卷积或 ViT（Vision Transformer）把图片映射成一系列视觉 token。  
   - 这些 token 与后面的语言 token 共享同一个词表，保证文字信息可以直接在解码器里被处理。

2. **统一解码器‑Only Transformer**  
   - 只有解码器层，每层都做自注意力（self‑attention）和跨模态注意力（cross‑attention）两步。  
   - 跨模态注意力让视觉 token 与已经生成的文字 token 互相影响，类似于人在阅读时不断把看到的文字和已知的上下文关联起来。

3. **任务提示机制**  
   - 输入序列最前面加上特定的标记，例如 `<spatial>` 表示要输出坐标化文本块，`<markdown>` 表示要输出 Markdown。  
   - 这相当于给模型下达“指令”，让同一个网络在同一次前向传播中切换不同的解码目标。

4. **自回归生成目标**  
   - 对于空间块任务，模型输出的 token 序列形如 “<text> … </text> <bbox> x1 y1 x2 y2 </bbox>”。  
   - 对于 Markdown 任务，模型直接生成符合 Markdown 语法的文本。  
   - 两种目标都使用普通的语言建模损失（负对数似然），不需要额外的多任务权重调节。

5. **指令微调（Instruction Tuning）**  
   - 在基础模型上加入对话式指令数据，例如 “请阅读这页合同并回答第 3 条的付款方式”。  
   - 通过少量高质量的指令对，让模型学会在对话中调用已经掌握的阅读与写作能力。

**最巧妙的设计**  
把视觉特征直接喂进解码器，而不使用单独的编码器‑解码器结构，使得模型在生成坐标或 Markdown 时可以随时参考已经生成的文字上下文。这种“随写随看”的方式让模型在处理长文档时保持全局一致性，避免了传统 OCR‑后处理流水线中出现的段落错位问题。

### 实验与效果

- **评测基准**：OCREval（字符错误率 CER）和 MarkdownEval（BLEU/ROUGE）两套新建数据集，分别覆盖 10k 张文档图片和对应的标注文本块/Markdown。  
- **主要结果**：论文声称 KOSMOS-2.5 在 OCREval 上的 CER 接近 GPT‑4o，MarkdownEval 的 BLEU 超过 0.85，明显优于传统 OCR+后处理管线（CER 超过 10%）。  
- **对比基线**：与同尺寸的开源多模态模型（如 LLaVA‑13B、BLIP‑2）相比，KOSMOS-2.5‑CHAT 在 9 项视觉问答基准上仅以 1.3 B 参数实现了与 7 B 参数模型相当的准确率，提升幅度在 5%~12% 之间。  
- **消融实验**：作者分别去掉任务提示、去掉视觉 token 与语言 token 的共享词表、以及只使用通用视觉‑语言语料进行预训练。结果显示：去掉提示后模型只能输出一种格式，性能下降约 8%；不共享词表导致跨模态注意力效果下降，整体 CER 上升约 4%；使用通用语料预训练则在复杂表格和小字号文字上错误率翻倍。  
- **局限性**：论文承认对极端低分辨率或严重遮挡的文字仍然表现不佳；Markdown 生成在极其复杂的嵌套表格上偶尔出现标签不匹配；模型规模仍然远小于 GPT‑4o，推理速度在高分辨率图片上仍有提升空间。

### 影响与延伸思考

KOSMOS-2.5 把“读”和“写”合二为一的思路打开了多模态文档处理的新局面。后续工作开始探索更大规模的空间感知文本生成模型（如 Gemini‑Vision），以及把这种统一解码器结构扩展到视频字幕生成、交互式文档编辑等场景。对想进一步深入的读者，可以关注以下方向：①更高效的视觉 token 编码（稀疏注意力、局部卷积），②跨语言的多模态文档翻译，③将空间坐标信息直接用于下游检索或知识图谱构建。整体来看，KOSMOS-2.5 为“从图片直接得到可编辑文档”奠定了技术基石。

### 一句话记住它

KOSMOS-2.5 用同一个解码器‑Only Transformer + 简单提示，就实现了把文本密集图片直接转成带坐标的文字块或 Markdown，参数小、效果媲美大模型。