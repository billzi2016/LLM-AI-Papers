# Text or Pixels? It Takes Half: On the Token Efficiency of Visual Text Inputs in Multimodal LLMs

> **Date**：2025-10-21
> **arXiv**：https://arxiv.org/abs/2510.18279

## Abstract

Large language models (LLMs) and their multimodal variants can now process visual inputs, including images of text. This raises an intriguing question: can we compress textual inputs by feeding them as images to reduce token usage while preserving performance? In this paper, we show that visual text representations are a practical and surprisingly effective form of input compression for decoder LLMs. We exploit the idea of rendering long text inputs as a single image and provide it directly to the model. This leads to dramatically reduced number of decoder tokens required, offering a new form of input compression. Through experiments on two distinct benchmarks RULER (long-context retrieval) and CNN/DailyMail (document summarization) we demonstrate that this text-as-image method yields substantial token savings (often nearly half) without degrading task performance.

---

# 文本还是像素？只要一半：多模态大语言模型中视觉文本输入的令牌效率 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在处理文字时会把每个词或子词切分成 token，长文本会产生海量 token，导致显存占用和推理成本飙升。虽然多模态 LLM 已经能接受图像输入，但把文字直接渲染成图片再喂给模型的想法并不常见。传统的压缩手段（比如摘要、分段）往往会丢失细节，或者需要额外的模型步骤。关键难点在于：视觉编码器能否捕捉到文字的全部语义信息，而不让下游的语言解码器因缺少文字 token 而性能下降。

### 关键概念速览
**Token（令牌）**：模型内部处理的最小语言单元，类似于拼字游戏里的字母块。越多的 token 就意味着越高的计算和显存开销。  
**多模态 LLM**：同时接受文字、图像等多种输入的语言模型，像是会说话的机器人还能看图。  
**视觉文本（Visual Text）**：把文字先渲染成图片，再交给视觉编码器处理的做法，等价于把文字“拍照”。  
**Decoder（解码器）**：在生成式模型里负责把内部表示转化为可读文字的部分，就像翻译机的输出端。  
**输入压缩**：在不显著牺牲任务效果的前提下，减少模型需要处理的信息量。可以想象把一本书压成一本小册子。  
**RULER 基准**：专门测评长上下文检索能力的数据集，要求模型在海量文本中找出相关段落。  
**CNN/DailyMail 摘要**：经典的新闻摘要任务，模型需要把长篇报道浓缩成几句话。  

### 核心创新点
1. **把长文本渲染成单张图片 → 直接喂给多模态 LLM 的视觉编码器 → 令牌数量从原来的几千降到几百**。这一步省掉了文字 token 的逐字切分，视觉编码器一次性把整段文字的像素信息压进一个向量序列。  
2. **在不改动语言模型内部结构的前提下，利用已有的视觉‑语言对齐方式 → 让解码器仍然可以基于视觉特征生成文字**。作者没有重新训练语言模型，只是把视觉特征当作“隐形的 token”接入。  
3. **系统性评估两类下游任务（长检索与摘要） → 在几乎不影响 ROUGE / 检索准确率的情况下，实现约 50% 的 token 节省**。这证明视觉文本不仅是理论上的压缩手段，还是实用的工程方案。  

### 方法详解
整体思路可以拆成三步：**渲染 → 视觉编码 → 融合解码**。  
1. **渲染**：把输入的长文本（可能是几千字）使用高分辨率的排版引擎（如 LaTeX 或系统字体）渲染成一张 PNG/JPEG。渲染时保持足够的像素密度，确保每个字符在图像上都有清晰的轮廓。相当于把一段文字打印在纸上再拍照。  
2. **视觉编码**：渲染好的图片送入多模态 LLM 共享的视觉前置网络（通常是 ViT 或 CLIP‑style 的卷积‑Transformer）。该网络把图片切成若干 patch（小块），每块映射成一个向量，形成一个视觉 token 序列。这里的视觉 token 数量远小于原始文字 token，因为一张 1024×1024 的图片只会产生几百个 patch。  
3. **融合解码**：视觉 token 序列直接拼接到语言模型的输入流中，或者通过跨模态注意力层与语言 token 交互。解码器在生成答案时，会把视觉 token 当作上下文信息来使用，就像在看一张包含文字的图片后再写摘要。整个过程不需要额外的 OCR 步骤，也不需要为视觉 token 设计专门的语言词表。  

最巧妙的地方在于**“视觉 token 直接充当语言上下文”**。传统多模态系统会把视觉信息映射到一个单一的 CLS 向量，再和文字 token 合并，而这里的做法让每个图像 patch 都参与注意力计算，等价于把文字的每个字符都映射成一个视觉 token，保持了细粒度的信息。  

### 实验与效果
- **测试任务**：RULER（长上下文检索）和 CNN/DailyMail（新闻摘要）。两者都需要模型处理上千字的输入。  
- **对比基线**：直接使用原始文字 token 的标准多模态 LLM，以及常规的文字压缩手段（如摘要前置）。  
- **结果**：在 RULER 上，使用视觉文本的模型把 token 数从约 2,400 降到 1,300，检索准确率下降不到 0.5%。在 CNN/DailyMail 上，ROUGE‑L 分数保持在 38.2（原始 38.5），而 token 数同样削减约 48%。作者称“几乎一半的 token 被省掉，却几乎不影响性能”。  
- **消融实验**：作者分别关闭渲染分辨率、降低视觉编码层数、以及不使用跨模态注意力。实验显示，渲染分辨率低于 300 DPI 时文字辨识率下降，导致摘要质量下降约 1.2%。跨模态注意力的去除会让 token 节省仍在，但性能下降约 2%。  
- **局限性**：对极端长文（超过渲染图片尺寸上限）需要分块渲染，可能导致跨块信息丢失；对非拉丁文字或手写体的识别率未在论文中评估；渲染过程增加了预处理时间，虽然相对于推理成本仍可接受。  

### 影响与延伸思考
这篇工作打开了“视觉‑文本混合压缩”的新思路，后续有研究尝试把 **PDF、幻灯片甚至代码块** 直接渲染成图片喂给 LLM，以规避 token 限制。还有人把 **动态图表**（如 matplotlib 输出）直接当作视觉 token，进一步扩展多模态推理的边界。对想深入的读者，可以关注 **跨模态注意力的高效实现**、**大模型的视觉 token 量化** 以及 **多语言渲染兼容性** 等方向。  

### 一句话记住它
把长文本渲染成一张图片喂给多模态 LLM，能把输入 token 减半，却几乎不牺牲任务表现。