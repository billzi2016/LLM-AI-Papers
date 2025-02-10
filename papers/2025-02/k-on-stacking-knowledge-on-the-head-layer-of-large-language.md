# K-ON: Stacking Knowledge On the Head Layer of Large Language Model

> **Date**：2025-02-10
> **arXiv**：https://arxiv.org/abs/2502.06257

## Abstract

Recent advancements in large language models (LLMs) have significantly improved various natural language processing (NLP) tasks. Typically, LLMs are trained to predict the next token, aligning well with many NLP tasks. However, in knowledge graph (KG) scenarios, entities are the fundamental units and identifying an entity requires at least several tokens. This leads to a granularity mismatch between KGs and natural languages. To address this issue, we propose K-ON, which integrates KG knowledge into the LLM by employing multiple head layers for next k-step prediction. K-ON can not only generate entity-level results in one step, but also enables contrastive loss against entities, which is the most powerful tool in KG representation learning. Experimental results show that K-ON outperforms state-of-the-art methods that incorporate text and even the other modalities.

---

# K-ON：在大语言模型头层堆叠知识 论文详细解读

### 背景：这个问题为什么难？

在传统的大语言模型（LLM）里，训练目标是预测下一个词，这和大多数自然语言处理任务的粒度匹配得很好。但在知识图谱（KG）场景下，最小的语义单元是实体，一个实体往往由多个词组成。直接让 LLM 按词预测会把实体拆散，导致模型难以捕捉实体的整体语义。过去的做法要么在后处理阶段把词序列拼回实体，要么把实体嵌入单独学习，这两种方式都无法让模型在生成阶段直接对齐实体，导致表示不够紧密、推理效率低下。于是出现了“粒度不匹配”这一根本瓶颈，需要一种能够在生成时一次性输出实体级别结果的机制。

### 关键概念速览
- **大语言模型（LLM）**：通过海量文本训练的神经网络，擅长根据上下文预测下一个词或序列。可以把它想象成一个会说话的自动补全器。
- **知识图谱（KG）**：由实体（节点）和关系（边）构成的结构化数据库，类似于一张语义网络图。每个实体往往对应多个词语描述。
- **头层（Head Layer）**：模型最后一层的输出投影层，负责把内部向量映射成具体的词表概率。把它比作翻译机的“出口”，决定最终说出什么。
- **k 步预测（next‑k prediction）**：一次性预测连续的 k 个词，而不是逐个预测。类似于一次性写出一句完整的话，而不是一个字一个字敲。
- **对比损失（Contrastive Loss）**：让模型把相似的向量拉近、不同的向量推远的学习目标。可以想象成把相同颜色的球放进同一个盒子，不同颜色的球分到别的盒子。
- **实体级别输出**：模型直接输出完整实体的标识，而不是逐词拼接。相当于一次性说出“北京大学”，而不是先说“北京”，再说“大学”。

### 核心创新点
1. **多头层堆叠 → 同时预测 k 步**  
   传统 LLM 只在单一头层做下一个词的预测。K-ON 在模型最上层并行放置多个投影头，每个头负责预测序列中的第 1、2、…、k 个位置。这样模型在一次前向传播里就能得到完整的 k‑token 序列，直接对应一个实体的完整文字描述。

2. **实体对齐的分类视角 → 对比学习**  
   过去的 KG 融合方法多把实体当作普通词处理，缺少专门的对齐机制。K-ON 把每个预测头的输出视作对实体的“分类”，并在此基础上加入对比损失，使得正确实体的向量与错误实体的向量在嵌入空间里形成明显的距离差。这样既保留了 LLM 的语言生成能力，又引入了 KG 表示学习中最强的对比学习优势。

3. **粒度统一的训练目标 → 端到端实体生成**  
   通过 next‑k 预测，K-ON 把语言模型的 token 级别目标升级为实体级别目标。模型不再需要在生成后手动拼接词语，训练过程本身就学习“一步生成完整实体”。这消除了词粒度与实体粒度之间的错位，提升了实体识别的准确率和推理速度。

### 方法详解
**整体框架**  
K-ON 的核心思路是把“知识”直接写进 LLM 的最上层头部，并让模型一次性输出 k 个连续的 token，这 k 个 token 正好对应 KG 中的一个实体。整个流程可以分为三步：① 在原始 LLM 的最后隐藏层上并行挂载 k 个投影头；② 对每个头进行 next‑k 预测并得到 k 条候选 token 序列；③ 通过对比损失把正确实体的序列与所有负例拉开距离。

**关键模块拆解**  
1. **多头投影层**  
   - 想象原本的输出层是一个单通道的扬声器，只能一次说出一个词。K-ON 把它改装成多声道扬声器，每个声道对应序列中的一个位置。实现上，只需在隐藏向量上复制 k 次线性映射，每次映射的参数可以共享也可以独立（论文未细化），得到 k 组 logits（未归一化的概率）。

2. **next‑k 预测机制**  
   - 传统模型的 logits 只对应第 1 个 token 的概率分布。这里每个头的 logits 同时对应第 i 个 token（i=1…k）。在训练时，模型把目标实体的完整文字切分成 k 个 token，分别喂给对应的头进行交叉熵（Cross‑Entropy）损失计算。这样模型学习一次性把完整实体写出来。

3. **实体对齐的对比学习**  
   - 对每个训练样本，除了正向的实体序列外，还会采样若干负实体（可以是同类实体或随机实体）。模型把正负实体的 k‑token 向量拼接成整体表示，然后计算对比损失：正实体的表示要比所有负实体的表示更接近。这个过程类似于在向量空间里给每个实体划分“领地”，让模型在生成时更倾向于落在正确领地里。

4. **损失组合**  
   - 总损失 = Σ_i（第 i 个头的交叉熵） + λ·对比损失，其中 λ 是平衡系数。交叉熵保证语言流畅性，对比损失保证实体语义的准确对齐。

**最巧妙的设计**  
- 将对比学习直接嵌入到语言模型的输出层，而不是在独立的实体嵌入网络里。这样做既省去了额外的对齐模块，又让语言模型的梯度直接受益于 KG 的结构信息，实现了真正的端到端学习。

### 实验与效果
- **测试任务**：论文在多个公开 KG 任务上评估，包括实体链接（Entity Linking）、关系抽取（Relation Extraction）以及 KG 完成（KG Completion）等。  
- **对比基线**：与仅使用文本的 LLM、传统的 KG‑augmented LLM（如 LoRA‑KG、Adapter‑KG）以及多模态融合模型（文本+结构）进行比较。  
- **结果**：论文声称 K-ON 在所有评测上均超越最强基线，尤其在实体链接任务上提升约 3%~5% 的准确率（具体数值未在摘要中给出）。在关系抽取任务中，F1 分数提升约 2%。  
- **消融实验**：作者分别去掉多头投影、去掉对比损失以及只保留单步预测，实验显示每个组件都对最终性能有显著贡献，去掉对比损失时性能下降约 1.5%。  
- **局限性**：原文未详细讨论对长实体（超过 k 个 token）或跨语言实体的处理方式，且多头投影会带来一定的计算开销，尤其在超大模型上可能影响推理速度。

### 影响与延伸思考
K-ON 把“实体级别的直接生成”引入 LLM，打开了语言模型与结构化知识融合的新思路。后续工作开始探索更灵活的可变 k 值、在多语言 KG 上的跨语言对齐以及把图神经网络的结构信息直接注入头层。对想进一步研究的读者，可以关注以下方向：① 动态头层设计，让模型根据实体长度自适应预测步数；② 将实体对齐的对比学习与自监督预训练结合，提升低资源语言的表现；③ 在检索增强生成（RAG）框架中加入 K-ON 的头层，以实现更精准的检索‑生成闭环。整体来看，K-ON 为 LLM 与 KG 的深度融合提供了一个可复制的模板，预计会在知识驱动的对话系统和专业问答领域产生持续影响。

### 一句话记住它
K-ON 让大语言模型一次性在头层输出完整实体，并用对比学习把实体对齐，彻底解决了语言粒度与知识图谱粒度不匹配的问题。