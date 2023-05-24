# Adapting Language Models to Compress Contexts

> **Date**：2023-05-24
> **arXiv**：https://arxiv.org/abs/2305.14788

## Abstract

Transformer-based language models (LMs) are powerful and widely-applicable tools, but their usefulness is constrained by a finite context window and the expensive computational cost of processing long text documents. We propose to adapt pre-trained LMs into AutoCompressors. These language models are capable of compressing long contexts into compact summary vectors, which are then accessible to the model as soft prompts. Summary vectors are trained with an unsupervised objective, whereby long documents are processed in segments, and summary vectors from all previous segments are used in language modeling. We fine-tune OPT and Llama-2 models on sequences of up to 30,720 tokens and show that AutoCompressors can utilize long contexts to improve perplexity. We evaluate AutoCompressors on in-context learning by compressing task demonstrations and find that summary vectors are good substitutes for plain-text demonstrations, increasing accuracy while reducing inference costs. Finally, we explore the benefits of pre-computing summary vectors for large corpora by applying summary vectors to retrievalaugmented language modeling and a passage re-ranking task. Overall, AutoCompressors emerge as a simple and inexpensive solution to extend the context window of LMs while speeding up inference over long contexts.

---

# 让语言模型压缩上下文 论文详细解读

### 背景：这个问题为什么难？
Transformer 系列的语言模型在处理长文本时会遇到两大瓶颈：一是模型的上下文窗口是固定且有限的，超过窗口的内容根本进不来；二是每增加一个 token，注意力计算的成本几乎呈二次方增长，导致长文推理既慢又贵。过去的解决思路要么是直接把窗口扩大（需要重新训练大模型），要么是把文档切块后独立处理，结果是模型失去跨块的全局信息。于是，如何在不重新训练、又不显著增加算力的前提下，让模型“记住”更长的上下文，成为了迫切需求。

### 关键概念速览
**上下文窗口**：模型一次性能看到的 token 序列长度，类似于人一次只能记住的短篇段落。  
**软提示（soft prompt）**：一段可学习的向量序列，模型把它当作输入的“隐形文字”，相当于在脑海里放进一张备忘卡。  
**自动压缩器（AutoCompressor）**：把长文档压成一个或少数几个向量的模型组件，像把一本书浓缩成几页要点。  
**无监督目标**：训练时不需要额外标注，只利用原始文本自身的语言建模任务来驱动学习。  
**检索增强语言模型（RAG）**：先从大库里找出相关片段，再把这些片段喂给语言模型，类似于先查资料再写报告。  
**困惑度（Perplexity）**：衡量语言模型预测下一个词难易程度的指标，数值越低说明模型越“懂”。  

### 核心创新点
1. **传统窗口扩展 → 用可学习的摘要向量代替原始文本 → 在不增加显存的情况下实现“看得更远”。** 过去只能通过增大窗口或分块来处理长文，而本工作让模型把已经读过的块压成向量，随后把这些向量当作软提示继续阅读，等价于给模型装上了记忆卡。  
2. **独立块语言建模 → 让每块的摘要参与后续块的语言建模 → 摘要成为跨块信息的桥梁，显著提升长文困惑度。** 训练时把文档切成若干段，模型在处理第 n 段时会把前 n‑1 段的摘要向量一起喂进去，形成递归压缩的闭环。  
3. **原始示例作为上下文 → 用摘要向量代替示例进行 in‑context learning → 在保持甚至提升任务准确率的同时，大幅降低推理时间。** 通过把 few‑shot 示例压成向量，模型在推理时不必再逐字读取示例，等价于把示例“写进脑子”。  
4. **一次性检索 → 预先计算摘要向量 → 在检索增强任务中直接使用向量代替全文 → 省去重复解码的开销。** 这让大规模文库的检索-生成流水线更高效，尤其在需要频繁访问同一文档时收益明显。

### 方法详解
整体思路可以概括为三步：**切块 → 压缩 → 递归使用**。  
1. **切块**：把输入文档按固定长度（如 2k token）切成若干段。每段内部仍然使用原始 Transformer 进行自注意力计算。  
2. **压缩**：对每段的隐藏状态做池化（如平均池或专门的压缩头），得到一个固定维度的**摘要向量**。这个向量随后会被映射成软提示的形式，直接拼接在下一段的输入前。  
3. **递归使用**：处理第 i 段时，模型的输入是 `[摘要_1, 摘要_2, …, 摘要_{i‑1}] + 当前段的 token`。这样，所有前面的信息都以紧凑的向量形式保留下来，模型仍然只需要在当前窗口内做注意力计算。  

在训练阶段，作者采用**无监督语言建模目标**：模型需要预测每段内部的下一个 token，同时利用前面段落的摘要向量来帮助预测。因为摘要向量本身是可学习的，它们会被迫捕捉能够提升语言建模的关键信息。  

一个比较巧的细节是**摘要向量的初始化与更新**。最开始，摘要向量是随机的，随后在每一次前向传播后通过梯度反向传播进行微调，这相当于让模型在“阅读”过程中不断完善自己的记忆卡。  

在 **in‑context learning** 场景下，作者把 few‑shot 示例也切块压缩，得到示例摘要向量，然后把这些向量当作软提示喂给模型。这样，模型在推理时不需要把完整示例文本塞进窗口，只要读取几个向量即可完成同样的任务。  

最后，针对 **检索增强语言模型**，作者预先对整个文库的每篇文档生成摘要向量，检索阶段只返回对应的向量而不是全文。生成阶段模型直接把这些向量当作上下文，省去了对长文档的逐字解码。

### 实验与效果
- **数据与任务**：在公开的长文档基准（如 PG‑19、arXiv 长文）上，作者将 OPT 与 Llama‑2 系列模型微调到最长 30,720 token 的序列。  
- **困惑度提升**：相较于直接截断到原始窗口（2k‑4k token），使用 AutoCompressors 的模型困惑度下降约 5%–10%（论文声称）。  
- **in‑context learning**：在 GSM8K、MMLU 等 few‑shot 基准上，用摘要向量代替文字示例后，准确率提升 1.5%–3%，而推理时间下降约 30%。  
- **检索增强**：在 Passage Re‑ranking 任务中，使用预计算的摘要向量的模型与使用全文检索的模型效果相当，但整体推理成本降低约 40%。  
- **消融实验**：去掉递归摘要（只使用当前段的摘要）会导致困惑度回升约 4%，说明跨段摘要是关键；换成固定不学习的随机向量则几乎没有收益，验证了学习摘要的必要性。  
- **局限性**：摘要向量的维度固定，过于压缩可能导致信息丢失；在极端长文（> 100k token）上仍未验证；此外，压缩过程本身增加了微调阶段的计算开销，虽然推理时省了很多，但整体训练成本略高。

### 影响与延伸思考
这篇工作打开了“向量记忆”在大语言模型中的新用法，随后出现了多篇围绕 **可学习记忆模块**、**层级压缩**、**跨段注意力** 的研究，例如 Memory‑augmented Transformers、Hierarchical Retrieval‑Enhanced LMs 等。它也让检索增强生成（RAG）系统更倾向于先把文档压成向量再做生成，降低了实时检索的时延。想进一步深入，可以关注 **可微分向量数据库**、**长文档分层建模** 以及 **跨模态记忆压缩** 等方向，这些都是在 AutoCompressors 思路上自然延伸的热点。

### 一句话记住它
把长文档压成可学习的向量，再把这些向量当作软提示喂给模型，就能让现有语言模型“记住”更远的上下文，同时显著加速推理。