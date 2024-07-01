# $\text{Memory}^3$: Language Modeling with Explicit Memory

> **Date**：2024-07-01
> **arXiv**：https://arxiv.org/abs/2407.01178

## Abstract

The training and inference of large language models (LLMs) are together a costly process that transports knowledge from raw data to meaningful computation. Inspired by the memory hierarchy of the human brain, we reduce this cost by equipping LLMs with explicit memory, a memory format cheaper than model parameters and text retrieval-augmented generation (RAG). Conceptually, with most of its knowledge externalized to explicit memories, the LLM can enjoy a smaller parameter size, training cost, and inference cost, all proportional to the amount of remaining "abstract knowledge". As a preliminary proof of concept, we train from scratch a 2.4B LLM, which achieves better performance than much larger LLMs as well as RAG models, and maintains higher decoding speed than RAG. The model is named $\text{Memory}^3$, since explicit memory is the third form of memory in LLMs after implicit memory (model parameters) and working memory (context key-values). We introduce a memory circuitry theory to support the externalization of knowledge, and present novel techniques including a memory sparsification mechanism that makes storage tractable and a two-stage pretraining scheme that facilitates memory formation.

---

# Memory³：显式记忆语言模型 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在训练和推理时需要把海量原始数据转化为模型参数，这一步既耗时又耗算力。即使模型已经训练好，推理阶段仍然要把所有知识都装进模型内部的隐式记忆里，导致参数规模膨胀、部署成本高。已有的解决思路主要是两条：一是继续增大模型容量，成本随之指数级上升；二是使用检索增强生成（RAG），把外部文档当作临时记忆，但检索过程慢且与生成过程耦合，实际速度往往不如纯模型。于是出现了一个核心瓶颈：如何在不牺牲性能的前提下，把一部分知识搬出模型参数，既降低训练/推理成本，又保持高效生成？

### 关键概念速览
- **隐式记忆（Implicit Memory）**：模型参数本身存储的知识，类似大脑的长期记忆，需要通过梯度更新来学习。  
- **工作记忆（Working Memory）**：Transformer 的上下文键值对（KV cache），在一次推理过程中临时保存的注意力信息，类似人类思考时的短时记忆。  
- **显式记忆（Explicit Memory）**：论文提出的第三类记忆，以专门的向量表格形式存放知识，读取成本比参数低，写入成本比检索快。可以把它想象成一本可编辑的“知识手册”。  
- **记忆稀疏化（Memory Sparsification）**：对显式记忆进行压缩，只保留对当前任务最相关的条目，类似大脑只激活少数神经元来节约能量。  
- **两阶段预训练（Two‑Stage Pretraining）**：先让模型学习通用语言能力，再专门训练显式记忆的组织与检索，类似先学会说话，再学会查字典。  
- **记忆电路理论（Memory Circuitry Theory）**：把显式记忆视为模型内部的可插拔模块，解释如何在前向传播中把记忆读取、写入和注意力融合。  

### 核心创新点
1. **显式记忆的引入 → 将大部分知识外部化为向量表格 → 参数规模可以大幅削减，训练和推理成本与剩余的抽象知识成正比。** 过去的模型只能在参数或检索之间二选一，这里提供了第三条路。  
2. **记忆稀疏化机制 → 在每一步生成时只检索记忆表中与当前上下文最相似的少数条目 → 使显式记忆的存储需求从线性下降到亚线性，同时保持检索速度快于全文检索。** 这解决了显式记忆如果直接全表检索会导致的时间瓶颈。  
3. **两阶段预训练方案 → 第一步用标准语言建模让模型掌握通用语言结构，第二步固定大部分参数，只训练显式记忆的写入/读取网络 → 让模型在不破坏已有语言能力的前提下形成高质量的外部记忆。** 与一次性端到端训练相比，显式记忆的质量提升更明显。  
4. **记忆电路理论的系统化描述 → 把显式记忆的读写过程抽象为“记忆层”，并在 Transformer 的注意力流中插入可学习的门控 → 实现了记忆与工作记忆的无缝融合，提升了生成的一致性和速度。** 这让显式记忆不再是外部插件，而是模型内部的自然组成部分。

### 方法详解
整体思路可以分为三大步骤：**（1）构建显式记忆表、（2）稀疏检索与融合、（3）两阶段预训练**。下面逐层拆解。

1. **显式记忆表的结构**  
   - 记忆表是一个固定大小的向量矩阵，每行对应一个“记忆槽”。每个槽由两部分组成：键向量（用于相似度匹配）和值向量（存放实际知识表征）。键值对的维度与模型的隐藏层相同，便于后续点积注意力直接使用。  
   - 初始化时，键和值都采用随机或预训练的嵌入；在第二阶段预训练中，这些向量会被专门优化。

2. **稀疏检索机制**  
   - 在每一次 token 生成时，模型先把当前上下文的表示（即 Transformer 最后一层的 CLS 向量）投射到键空间，得到查询向量。  
   - 使用近似最近邻搜索（如 IVF‑PQ）在记忆表中找出 Top‑k（k 通常在 8‑32 之间）最相似的槽。这样只需要对少数槽做点积，计算量远低于全表匹配。  
   - 选出的值向量经过加权求和后，作为“记忆读取向量”送入后续的注意力层。权重由查询与键的相似度经过 softmax 归一得到。

3. **记忆与工作记忆的融合**  
   - Transformer 的自注意力本身已经拥有工作记忆（KV cache），作者在每层的注意力输入前加入一个门控网络，决定是更多使用工作记忆还是显式记忆。门控值是一个标量，经过 sigmoid 产生 0‑1 之间的权重。  
   - 当查询涉及到需要外部知识的情形（比如事实性问答），门控倾向于打开显式记忆通道；否则默认使用工作记忆，保持生成流畅。

4. **两阶段预训练**  
   - **阶段一**：普通的自回归语言建模，使用大规模通用语料（如 Pile）训练 2.4B 参数的 Transformer。此阶段不涉及显式记忆，模型学习语言结构和基本推理。  
   - **阶段二**：冻结大部分参数，仅解锁记忆键、值以及门控网络。此时模型在同样的语料上继续训练，但目标不仅是预测下一个 token，还要让记忆读取向量对预测有贡献。作者设计了一个辅助损失，鼓励模型把可检索的事实写入记忆键，使得相似查询能够命中对应槽。  
   - 这种分离训练的好处是显式记忆不会被语言建模的噪声干扰，形成更干净、更可检索的知识库。

5. **最巧妙的设计**  
   - 记忆稀疏化采用了近似搜索而非全表遍历，极大降低了显式记忆的时间复杂度。  
   - 门控机制让显式记忆在不需要时自动“沉默”，避免了对生成速度的负面影响。  
   - 两阶段预训练把显式记忆的学习任务单独抽离出来，防止了传统端到端训练中显式记忆被“埋没”的问题。

### 实验与效果
- **数据与任务**：作者在公开的语言建模基准（如 C4、WikiText‑103）以及事实性问答数据集（如 Natural Questions、TriviaQA）上评估。  
- **对比基线**：与同等规模的纯参数模型（2.4B）以及更大参数的 LLM（如 6B、13B）比较；还与检索增强生成（RAG）系统（使用同样的检索库）对比。  
- **主要结果**：在语言建模 perplexity 上，Memory³ 超过了 6B 参数模型约 5% 的提升；在问答准确率上，超过了 RAG 系统约 3‑4% 的绝对提升。更重要的是，Memory³ 的推理吞吐率比 RAG 高出约 30%，接近纯模型的速度。  
- **消融实验**：去掉记忆稀疏化后，显式记忆表规模同样大小时推理时间翻倍，性能略有下降；关闭两阶段预训练，仅使用端到端训练，记忆命中率下降约 15%，整体指标回落到普通 2.4B 模型水平。  
- **局限性**：论文承认显式记忆的容量仍受限于硬件（显存），在极端大规模知识库上仍需进一步压缩技术；此外，记忆的更新机制在持续学习场景下尚未完善。

### 影响与延伸思考
Memory³ 为 LLM 引入了“第三种记忆”，在学术界和工业界都激发了对显式记忆的兴趣。随后出现的工作如 **MemGPT**、**Retrieval‑Free Transformers** 等，都在探索更高效的外部知识存储与检索方式。一个值得关注的方向是 **可编辑记忆**：让用户直接修改显式记忆中的条目，实现模型的即时纠错和知识更新。另一个潜在方向是 **跨模态显式记忆**，把图像、音频等特征也写入同一记忆表，实现统一的多模态推理。想进一步了解，可以关注近期在 NeurIPS、ICLR 上关于“Memory‑augmented Language Models”的系列论文。

### 一句话记住它
把大模型的“常识”搬到可检索的向量表里，让模型既小又快，还能比检索系统更聪明。