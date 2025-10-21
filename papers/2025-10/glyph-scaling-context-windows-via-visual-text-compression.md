# Glyph: Scaling Context Windows via Visual-Text Compression

> **Date**：2025-10-20
> **arXiv**：https://arxiv.org/abs/2510.17800

## Abstract

Large language models (LLMs) increasingly rely on long-context modeling for tasks such as document understanding, code analysis, and multi-step reasoning. However, scaling context windows to the million-token level brings prohibitive computational and memory costs, limiting the practicality of long-context LLMs. In this work, we take a different perspective-visual context scaling-to tackle this challenge. Instead of extending token-based sequences, we propose Glyph, a framework that renders long texts into images and processes them with vision-language models (VLMs). This approach substantially compresses textual input while preserving semantic information, and we further design an LLM-driven genetic search to identify optimal visual rendering configurations for balancing accuracy and compression. Through extensive experiments, we demonstrate that our method achieves 3-4x token compression while maintaining accuracy comparable to leading LLMs such as Qwen3-8B on various long-context benchmarks. This compression also leads to around 4x faster prefilling and decoding, and approximately 2x faster SFT training. Furthermore, under extreme compression, a 128K-context VLM could scale to handle 1M-token-level text tasks. In addition, the rendered text data benefits real-world multimodal tasks, such as document understanding. Our code and model are released at https://github.com/thu-coai/Glyph.

---

# Glyph：通过视觉‑文本压缩实现超长上下文的论文详细解读

### 背景：这个问题为什么难？

大语言模型在处理文档、代码或多步推理时，需要把上千甚至上万字的内容一次性喂进去。传统做法是把文字切成 token（模型的最小单元），然后把这些 token 按顺序送进 Transformer。可是 token 数量每翻一倍，显存和算力的需求就会指数级增长，导致“一百万 token”级别的上下文几乎不可用。现有的长上下文方案要么在模型结构上做复杂改造（稀疏注意力、滑动窗口），要么通过分块检索来折中，但都难以兼顾高效、低成本和语义完整性。于是，如何在不改动模型核心的前提下，大幅压缩输入而不丢失信息，成为了迫切的研究点。

### 关键概念速览

**Token（标记）**：模型内部处理的最小文本单元，类似把一句话拆成拼音或字母。一个普通句子往往会产生上百个 token。

**视觉语言模型（VLM）**：同时懂图像和文字的模型，能够把图片里的文字识别出来并进行推理，像是把 OCR（光学字符识别）和语言理解合二为一的“全能大脑”。

**文本渲染**：把一段文字按照特定字体、字号、排版方式绘制成图片的过程。想象把 Word 文档打印成 PDF 再截图，就是一种渲染。

**遗传搜索（Genetic Search）**：受生物进化启发的搜索算法，通过“交叉”“变异”“选择”不断迭代，找到最优解。这里用它来自动挑选渲染参数。

**Prefill（预填充）**：模型在生成第一个 token 前一次性读取全部上下文的过程，耗时往往占据总推理时间的大头。

**SFT（监督微调）**：在大模型上再用特定任务的数据进行微调，让模型更贴合实际使用场景。

**OCR（光学字符识别）**：把图片中的文字转成机器可读的字符序列，是渲染后恢复文本的关键技术。

### 核心创新点

1. **把长文本直接渲染成图像 → 用视觉语言模型读取**  
   传统方法一直在扩展 token 序列，Glyph 则把整段文字绘成一张或多张高分辨率图片。因为一张图片的像素远多于 token 的数量，这一步实现了 3–4 倍的压缩率，同时保留了排版、段落结构等隐含信息。结果是，原本需要上万 token 的输入，只需要几千像素就能完整表达。

2. **LLM‑驱动的遗传搜索寻找最佳渲染配置 → 自动平衡压缩率与语义保真度**  
   渲染参数（字体、字号、行距、页面宽度等）直接决定压缩效果。作者让一个小型语言模型负责评估不同渲染方案的“好坏”，并用遗传算法迭代搜索。相比手工调参，这种自适应搜索找到了在保持准确率的前提下，压缩率最高的配置。

3. **三阶段训练流程：CPT → 渲染搜索 → 后训练**  
   - **CPT（Cross‑modal Pre‑training）**：让 VLM 学会从各种渲染风格的图片中进行 OCR、交叉语言建模和缺失页生成，等价于让模型熟悉“文字＝图片”这条跨模态映射。  
   - **渲染搜索**：前一步得到的评估函数指导遗传搜索，锁定最优渲染参数。  
   - **后训练**：在固定渲染下进行监督微调（SFT）和基于 LLM 判定的奖励微调（GRPO），进一步提升推理准确性。  
   这套流程把视觉压缩与语言理解紧密耦合，形成了闭环优化。

4. **极端压缩下的 1M‑token 任务可行性**  
   通过把 128K‑context 的 VLM 与 Glyph 的压缩管线组合，作者展示了在 1M‑token 规模的文本任务上仍能保持可接受的速度和精度，打开了“千页文档一次性阅读”的新大门。

### 方法详解

**整体框架**  
Glyph 的工作流可以概括为四步：  
1) **文本 → 渲染**：把原始长文本按照搜索得到的最佳排版参数绘制成若干图片。  
2) **图片 → VLM 编码**：视觉语言模型把这些图片转成内部向量序列。  
3) **向量 → LLM 推理**：将 VLM 输出的向量喂给已有的大语言模型（如 Qwen3‑8B），完成问答、摘要或代码分析等任务。  
4) **结果输出**：LLM 直接生成文字答案，无需再做 OCR 逆向。

**关键模块拆解**  

- **渲染子系统**：采用开源排版引擎（如 Pango/FreeType）生成 PNG/JPEG。参数空间包括字体族（Serif、Sans‑Serif）、字号（8–14pt）、行距、页面宽度、是否加粗等。每种组合对应一种“压缩风格”。  

- **遗传搜索引擎**：初始化一批随机渲染配置（种群），每个配置渲染一小段代表性文本后交给 VLM+LLM 进行推理。LLM 负责给出任务准确率（如 QA 正确率）和压缩率的加权评分。随后进行选择、交叉、变异，迭代数十代后收敛到最优配置。这里的“适应度函数”是作者自行设计的，兼顾了 **准确率下降 ≤ 2%** 与 **压缩率最大化** 两个目标。

- **CPT 训练**：在大规模混合数据上（包括 OCR 数据、跨语言对齐文本、缺失页生成任务），让 VLM 同时学习文字识别和上下文推理。相当于让模型在“看图识字+读懂上下文”两件事上同步进步。  

- **后训练（SFT + GRPO）**：固定渲染参数后，用任务特定的数据（LongBench、文档问答等）进行监督微调。GRPO（Generative Reward from Prompted LLM）则让一个小型 LLM 给每个生成的答案打分，作为强化学习的奖励信号，帮助模型在视觉输入上更好地对齐语言输出。

**最巧妙的点**  
- 把 **渲染搜索** 放在 **语言模型** 的评估回路里，而不是单纯靠视觉指标（如 OCR 准确率）来决定。这样搜索过程直接面向下游任务的实际表现，避免了“看得清楚但推理错误”的误区。  
- 在 **CPT** 阶段加入 **缺失页生成** 任务，使模型学会在部分信息缺失的情况下进行补全，这对后续长文档的跨页推理尤为重要。

### 实验与效果

- **测试数据**：LongBench（长文本问答、摘要、代码分析等子任务）、真实文档问答数据集以及自建的 1M‑token 合成任务。  
- **基线对比**：与 Qwen3‑8B（原始 token 输入）、DeepSeek‑OCR、以及几种稀疏注意力模型（如 Longformer、BigBird）进行比较。  
- **核心结果**：在 LongBench 上，Glyph 使用 3–4 倍的 token 压缩后，准确率仅比 Qwen3‑8B 低约 1.5%（如 QA 任务从 78% 降到 76%），而预填充和解码速度提升约 4 倍，SFT 训练时间缩短约 2 倍。  
- **极端压缩实验**：在 128K‑context VLM 上，渲染后处理 1M‑token 文本，模型仍能在 30 秒内完成一次推理，误差保持在可接受范围内。  
- **消融研究**：去掉遗传搜索，仅使用手工渲染参数，压缩率下降约 15%，下游任务准确率下降 3% 以上；不进行 CPT 训练，VLM 的 OCR 错误率上升 20%，导致整体推理准确率显著下降。  
- **局限性**：渲染参数在搜索后固定，面对已经是图片形式的输入（如扫描文档）时，无法再进行压缩；极端压缩下细小字体会导致 OCR 失误；作者也提到在某些多语言混排场景下，渲染效果仍有提升空间。

### 影响与延伸思考

Glyph 把“视觉压缩”引入长上下文建模，打开了一个全新的思路：不一定要在 token 维度上做花样，直接在信息表现层面压缩也能实现高效推理。此后，出现了几篇尝试把 **PDF → 图像 → VLM** 直接用于法律文档审查、金融报告分析的工作，甚至有团队把 **音频 → 频谱图 → VLM** 的思路迁移到长音频理解上。对想进一步探索的读者，建议关注以下方向：  
- **自适应渲染**：在推理过程中根据上下文动态调整渲染参数，解决混合文本/图像输入的兼容性。  
- **跨模态检索**：把渲染后的图像索引进向量数据库，实现长文档的快速相似片段检索。  
- **多语言排版**：研究不同语言（中、英、阿拉伯等）在同一渲染框架下的压缩表现，提升跨语言长文档的通用性。

### 一句话记住它

**Glyph 用“把文字画成图、让视觉模型读图” 的方式，把上万 token 的文本压缩到几千像素，既省算力又保持答案质量。**