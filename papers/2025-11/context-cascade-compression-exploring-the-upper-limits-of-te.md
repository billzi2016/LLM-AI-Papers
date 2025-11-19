# Context Cascade Compression: Exploring the Upper Limits of Text Compression

> **Date**：2025-11-19
> **arXiv**：https://arxiv.org/abs/2511.15244

## Abstract

Million-level token inputs in long-context tasks pose significant computational and memory challenges for Large Language Models (LLMs). Recently, DeepSeek-OCR conducted research into the feasibility of Contexts Optical Compression and achieved preliminary results. Inspired by this, we introduce Context Cascade Compression C3 to explore the upper limits of text compression. Our method cascades two LLMs of different sizes to handle the compression and decoding tasks. Specifically, a small LLM, acting as the first stage, performs text compression by condensing a long context into a set of latent tokens (e.g., 32 or 64 in length), achieving a high ratio of text tokens to latent tokens. A large LLM, as the second stage, then executes the decoding task on this compressed context. Experiments show that at a 20x compression ratio (where the number of text tokens is 20 times the number of latent tokens), our model achieves 98% decoding accuracy, compared to approximately 60% for DeepSeek-OCR. When we further increase the compression ratio to 40x, the accuracy is maintained at around 93%. This indicates that in the domain of context compression, C3 Compression demonstrates superior performance and feasibility over optical character compression. C3 uses a simpler, pure-text pipeline that ignores factors like layout, color, and information loss from a visual encoder. This also suggests a potential upper bound for compression ratios in future work on optical character compression, OCR, and related fields. Codes and model weights are publicly accessible at https://github.com/liufanfanlff/C3-Context-Cascade-Compression

---

# 上下文级联压缩：探索文本压缩的上限 论文详细解读

### 背景：这个问题为什么难？

在长文本任务中，模型往往需要一次性处理上百万个 token，计算量和显存需求呈指数增长，普通的 LLM 很快就会“卡死”。已有的解决思路大多依赖视觉压缩（比如 OCR 把文字转成图像再压缩）或硬件层面的分块，但这些方法要么信息损失大，要么实现复杂，难以在纯文本环境下直接使用。于是，如何在保持几乎完整语义的前提下，把超长上下文压缩到几百甚至几十个 token，成为了制约大模型长上下文能力的核心瓶颈。

### 关键概念速览
- **latent token（潜在 token）**：压缩后模型内部使用的代替原始文字的“隐形”符号，数量远少于原始 token，类似于把一本书浓缩成几页要点。
- **compression ratio（压缩比）**：原始文字 token 数除以潜在 token 数的比例，20× 表示 20 个文字 token 被压成 1 个潜在 token。
- **cascade architecture（级联架构）**：把两个模型串起来使用，前一个负责压缩，后一个负责解压，像流水线生产一样分工合作。
- **soft compression（软压缩）**：不依赖硬件或图像编码，而是让语言模型自己学习一种高效的内部表征方式。
- **query‑aware compression（查询感知压缩）**：压缩器在生成潜在 token 时会把用户的查询一起喂进去，确保压缩结果对当前任务最有用。
- **reconstruction loss（重构损失）**：训练时比较解码后文本与原始文本的差异，用来驱动压缩器学会不丢信息。
- **natural‑language prompt（自然语言提示）**：解码器的输入采用可读的文字指令，而不是固定的软提示，灵活度更高。

### 核心创新点
1. **双模型级联 → 小模型压缩 + 大模型解码 → 让压缩比例突破 20×、40× 仍保持 93%+ 的解码准确率**  
   传统方法要么只用单一模型压缩，要么依赖视觉编码，压缩比受限。C3 把一个体积小、参数少的 LLM 训练成“压缩器”，只输出几十个潜在 token；随后把这些 token 交给一个参数多、能力强的大模型进行恢复。两者的能力差异被巧妙利用，使得压缩比大幅提升而不牺牲信息完整性。

2. **查询感知压缩 → 把用户 query 直接喂入压缩器 → 生成的潜在 token 更贴合下游任务**  
   以前的软压缩往往只看原始上下文，压缩后通用性强但对特定问题不够精准。C3 在压缩阶段加入查询向量，让压缩器在“浓缩要点”时已经考虑到用户真正想要的答案方向，从而在解码时更容易恢复出正确答案。

3. **自然语言提示的解码器 → 用可读的 prompt 替代固定软提示 → 提升灵活性和可解释性**  
   许多软压缩系统在解码时使用固定的向量作为“提示”，难以调试。C3 让大模型接受一段自然语言指令（例如 “请根据以下压缩信息回答问题”），既保持了模型的语言理解能力，又让人类可以直接查看和修改提示。

4. **纯文本管线 → 完全抛弃布局、颜色等视觉信息 → 为光学字符压缩设定上限**  
   DeepSeek‑OCR 通过图像压缩实现上下文压缩，但受限于视觉噪声和信息丢失。C3 证明，仅靠文字本身就能达到更高的压缩比，这为后续研究提供了一个“上限基准”，提醒光学压缩还有提升空间。

### 方法详解
**整体思路**  
C3 的工作流程可以划分为三步：① 接收原始长文本和用户查询；② 小模型把它们一起压缩成固定长度的潜在 token 序列；③ 大模型把潜在 token 结合自然语言提示解码回完整文本或直接生成答案。整个过程全程在文字空间里完成，没有任何图像或外部编码介入。

**步骤拆解**  

1. **输入准备**  
   - 原始上下文被切分成普通的 token 序列 `C = [c1, c2, …, cN]`（N 可能是上百万）。  
   - 用户查询 `Q` 也被 token 化。  
   - 两者在拼接后形成 `I = [Q, C]`，作为压缩器的输入。

2. **小模型压缩器（Stage‑1）**  
   - 采用参数较少的 LLM（如 1.3B）进行自回归编码。  
   - 目标是生成长度为 `L`（如 32 或 64）的潜在 token 序列 `Z = [z1, …, zL]`。  
   - 训练时使用 **重构损失**：把 `Z` 再喂给解码器（Stage‑2），得到恢复文本 `Ĉ`，计算 `Ĉ` 与原始 `C` 的交叉熵或 BLEU 等相似度指标，梯度回传到压缩器。  
   - 为了让压缩器关注查询，`Q` 在输入时被标记为特殊 token，模型在生成 `Z` 时会自适应地把查询信息编码进去。

3. **大模型解码器（Stage‑2）**  
   - 选用参数更大的 LLM（如 13B 或 70B），其优势在于强大的上下文理解和生成能力。  
   - 解码时，模型的 **prompt** 采用自然语言形式，例如：“下面是一段压缩信息，请根据它回答问题：{Q}”。随后把潜在 token `Z` 直接拼接到 prompt 后面。  
   - 大模型在自回归生成时，先把 `Z` 解释为内部语义，然后继续生成完整的原始文本或直接输出答案。因为 `Z` 已经是高度浓缩的语义向量，模型只需要“展开”即可。

4. **推理流程**  
   - 实际使用时，用户只需要提供查询 `Q` 和长文本 `C`。系统先跑 Stage‑1，得到 `Z`（几毫秒到几秒，取决于小模型大小），再把 `Z` 和 prompt 交给 Stage‑2，得到最终答案。  
   - 由于 `Z` 的长度固定，Stage‑2 的显存占用与原始文本长度无关，实现了对上百万 token 的“看不见”处理。

**关键技巧**  
- **查询感知压缩**：把查询提前进入压缩器，使得潜在 token 不仅是上下文的摘要，还携带了任务导向的信息。  
- **潜在 token 长度的离散选择**：实验发现 32 与 64 两个长度在 20×、40× 压缩比下表现最佳，过短会导致信息不可逆，过长则浪费压缩优势。  
- **自然语言 Prompt**：相比固定向量提示，文字提示让大模型能够利用已有的指令理解能力，提升了恢复质量并降低了调参难度。  
- **纯文本管线**：省去 OCR、图像特征提取等步骤，避免了视觉噪声的累积，使得压缩上限更接近信息论意义上的极限。

### 实验与效果
- **测试场景**：作者在公开的长文本基准上（如数百万 token 的文档集合）进行评估，重点考察在不同压缩比下的解码准确率。  
- **对比基线**：主要与 DeepSeek‑OCR（光学字符压缩）进行比较。DeepSeek‑OCR 在 20× 压缩比时的解码准确率约为 60%。  
- **核心结果**：  
  - 在 **20×** 压缩比（原始 token 数是潜在 token 的 20 倍）时，C3 达到 **98%** 的解码准确率。  
  - 将压缩比提升到 **40×**，准确率仍保持在 **约 93%**，远高于同等压缩下的视觉方法。  
- **消融实验**：论文中提供了两项关键消融：① 移除查询感知输入，压缩比相同情况下准确率下降约 5%；② 用固定软提示替代自然语言 Prompt，解码质量下降约 3%。这些实验表明查询感知和自然语言 Prompt 对性能提升贡献显著。  
- **局限性**：作者指出，纯文本管线忽略了布局、颜色等信息，在处理包含表格、图形或多模态混排的文档时可能失效；此外，压缩器的训练成本仍然不低，需要大规模长文本数据进行自监督预训练。

### 影响与延伸思考
C3 的出现让研究者重新审视“软压缩”在长上下文中的潜力，后续工作开始探索更深层次的级联结构（如三阶段压缩）或把压缩器与检索模块结合，实现“检索+压缩+生成”的统一框架。还有团队尝试把查询感知压缩迁移到多模态场景，加入图像特征作为额外输入，期待在保持高压缩比的同时兼顾视觉信息。对想进一步了解的读者，可以关注以下方向：① 大模型的“记忆”机制（如 KV‑cache 优化）；② 基于自监督的长文本预训练（如 Long-LLM、Transformer‑XL 的改进）；③ 软提示与自然语言 Prompt 的混合使用策略。整体来看，C3 为长文本处理提供了一个“纯文字版的压缩上限”，为后续的光学字符压缩、文档检索和大模型推理优化提供了重要参考。

### 一句话记住它
**C3 用小模型把上百万字压成几十个潜在 token，再让大模型用自然语言提示把它们展开，轻松实现 20×‑40× 超高压缩且保持 93%+ 的解码准确率。**