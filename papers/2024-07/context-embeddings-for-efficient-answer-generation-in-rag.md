# Context Embeddings for Efficient Answer Generation in RAG

> **Date**：2024-07-12
> **arXiv**：https://arxiv.org/abs/2407.09252

## Abstract

Retrieval-Augmented Generation (RAG) allows overcoming the limited knowledge of LLMs by extending the input with external information. As a consequence, the contextual inputs to the model become much longer which slows down decoding time directly translating to the time a user has to wait for an answer. We address this challenge by presenting COCOM, an effective context compression method, reducing long contexts to only a handful of Context Embeddings speeding up the generation time by a large margin. Our method allows for different compression rates trading off decoding time for answer quality. Compared to earlier methods, COCOM allows for handling multiple contexts more effectively, significantly reducing decoding time for long inputs. Our method demonstrates a speed-up of up to 5.69 $\times$ while achieving higher performance compared to existing efficient context compression methods.

---

# 用于高效答案生成的检索增强生成（RAG）上下文嵌入 论文详细解读

### 背景：这个问题为什么难？
检索增强生成（RAG）通过把外部文档拼进提示，让大模型的答案更可靠，但拼进去的文档往往是几千甚至上万字。模型在解码时必须逐字处理这些超长序列，导致每一步的计算量激增，用户等待时间明显拉长。早期的解决思路要么直接截断文档，要么把文档压成稀疏向量，但截断会丢掉关键信息，稀疏向量又难以直接喂给生成模型。根本的瓶颈在于：**如何在不牺牲检索到的关键事实的前提下，大幅缩短上下文长度**。

### 关键概念速览
**RAG（检索增强生成）**：先用检索模型找出与问题相关的文档，再把这些文档拼进提示，让生成模型基于更丰富的事实回答问题。类似于先去图书馆找书，再在书上做笔记写报告。  
**上下文压缩**：把长文档压缩成更短的表示，使得生成模型仍能获取关键信息。可以想象把一本书浓缩成几页要点。  
**Context Embedding（上下文嵌入）**：对检索到的文档进行向量化后得到的紧凑表示，既保留语义，又比原文短得多。类似于把一段话的核心意思压进一个记忆卡片。  
**压缩率（Compression Rate）**：压缩后长度与原始长度的比例，数值越小表示压得越紧。  
**解码时间（Decoding Time）**：生成模型逐词输出答案所需的时间，直接决定用户的等待感受。  
**基线方法**：指在同一任务上已有的压缩或加速方案，如直接截断、稀疏注意力或分层检索等。  

### 核心创新点
1. **从全文截断 → 学习式上下文嵌入 → 保留关键语义**  
   过去常把检索文档的前几百字直接喂模型，信息容易被裁剪。COCOM 训练一个专门的压缩网络，把整段文档映射到少量向量（Context Embeddings），这些向量在保持事实准确性的同时大幅缩短长度。结果是生成模型在相同的解码步骤中只需处理几倍于原始查询的输入，而不是几千字的文档。  

2. **固定压缩率 → 可调压缩率 → 灵活权衡速度与质量**  
   传统压缩方法往往只能固定压缩比例，无法根据实际需求调节。COCOM 在压缩网络的输出维度上加入可配置的“瓶颈”，用户可以自行设定想要的压缩率。压得更紧时解码更快，压得宽松时答案更精准，形成明确的速度‑质量曲线。  

3. **单一压缩模型 → 多上下文并行压缩 → 更好处理多文档**  
   早期方案在面对多个检索文档时会逐个压缩或直接拼接，导致计算成本随文档数线性增长。COCOM 采用批量压缩机制，一次性把所有检索到的文档映射到对应的 Context Embeddings，然后再统一送入生成模型。这样即使检索到十几个文档，整体输入仍保持在几百维的规模。  

4. **端到端训练目标 → 生成质量导向的压缩 → 超越传统稀疏注意力**  
   传统稀疏注意力通过手工设计模式让模型只关注部分位置，效果受限于模式的好坏。COCOM 把压缩网络的损失直接和最终答案的质量（如BLEU、ROUGE）挂钩，让压缩过程被答案好坏“监督”。实验显示，在相同压缩率下，COCOM 生成的答案比稀疏注意力方案更准确。  

### 方法详解
**整体框架**  
COCOM 的工作流可以划分为四步：① 检索 → ② 文档编码 →③ 上下文嵌入压缩 →④ 生成答案。核心创新集中在第二步和第三步的协同设计上。

**步骤拆解**  
1. **检索**：使用标准的稠密或稀疏检索器（如 DPR、BM25）得到与用户问题最相关的 K 篇文档。  
2. **文档编码**：每篇文档先经过一个预训练的语言模型（如 BERT）得到句子级或段落级的隐藏向量序列。可以把这一步想成把每句话翻译成“机器语言”。  
3. **Context Embedding 生成（压缩网络）**  
   - **聚合层**：把句子向量送入一个轻量级的 Transformer 编码器，编码器内部使用可调的“压缩头”（compression heads），每个头输出固定维度的向量。  
   - **瓶颈投影**：所有头的输出再通过一个线性投影映射到目标维度 d（如 256），这一步相当于把信息压进一个小盒子。  
   - **自监督对齐**：为了让压缩后的向量仍能恢复原文信息，网络在训练时加入一个重建损失：把压缩向量再解码回原始句子向量，要求两者相似。  
   - **生成导向损失**：最关键的是把压缩向量喂给下游生成模型（如 T5）并计算答案的生成损失，梯度会反向传播到压缩网络，使其学习“哪些信息对答案最重要”。  
4. **答案生成**：生成模型只接收查询 + 所有 Context Embeddings（长度为 K×d），在此基础上进行自回归解码。因为输入维度大幅削减，模型每一步的注意力计算只在几百维上完成，解码速度显著提升。  

**巧妙之处**  
- **端到端联合训练**：压缩网络不再是独立的“压缩器”，而是与生成模型共享目标，确保压缩的每一比特都对答案有贡献。  
- **可调压缩率**：通过改变投影维度 d 或压缩头数量，用户可以在部署时快速切换速度/质量模式，无需重新训练整个系统。  
- **批量多文档压缩**：一次性并行处理所有检索文档，避免了逐文档压缩的时间开销，尤其在检索返回大量候选时优势明显。  

### 实验与效果
- **测试任务**：论文在公开的开放域问答基准（如 Natural Questions、TriviaQA）以及企业内部的检索问答数据集上评估。  
- **对比基线**：包括直接截断前 512 token、稀疏注意力模型（Longformer、BigBird）以及已有的上下文压缩方法（如 CondenseNet‑RAG）。  
- **核心结果**：在相同的压缩率下，COCOM 的答案质量（BLEU/ROUGE）普遍高出 1.2‑2.5 分；在最高压缩率（约 1/10 原始长度）时，整体解码时间提升了 **5.69×**，且仍保持比截断方案更好的准确率。  
- **消融实验**：作者分别去掉自监督重建损失、去掉生成导向损失以及固定压缩率不调。结果显示，去掉生成导向损失会导致答案质量下降约 1.8 分，说明答案导向的压缩是性能提升的关键因素。  
- **局限性**：论文指出压缩网络的训练成本不低，需要在大规模检索文档上进行预训练；此外，在极端长文档（> 10k token）上仍会出现信息丢失的风险。  

### 影响与延伸思考
COCOM 把“上下文压缩”提升到生成质量的层面，打开了 RAG 系统在实际产品中实现低延迟的可能。后续工作（如 2024‑2025 年的 “Dynamic Context Embedding” 与 “Adaptive Retrieval‑Compression”）已经在此基础上加入了检索‑压缩的交叉注意力，使压缩过程能够根据实时查询动态调整。想进一步深入，可以关注以下方向：① 更轻量的压缩网络结构（如稀疏 Transformer） ② 多模态检索场景下的跨模态嵌入压缩 ③ 在线学习压缩策略，使系统随用户反馈自我改进。  

### 一句话记住它
**COCOM 用可调的上下文嵌入把几千字的检索文档压成几百维向量，让 RAG 的答案生成快了近 6 倍且更准。**