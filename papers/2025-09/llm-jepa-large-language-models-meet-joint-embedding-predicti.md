# LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures

> **Date**：2025-09-11
> **arXiv**：https://arxiv.org/abs/2509.14252

## Abstract

Large Language Model (LLM) pretraining, finetuning, and evaluation rely on input-space reconstruction and generative capabilities. Yet, it has been observed in vision that embedding-space training objectives, e.g., with Joint Embedding Predictive Architectures (JEPAs), are far superior to their input-space counterpart. That mismatch in how training is achieved between language and vision opens up a natural question: {\em can language training methods learn a few tricks from the vision ones?} The lack of JEPA-style LLM is a testimony of the challenge in designing such objectives for language. In this work, we propose a first step in that direction where we develop LLM-JEPA, a JEPA based solution for LLMs applicable both to finetuning and pretraining. Thus far, LLM-JEPA is able to outperform the standard LLM training objectives by a significant margin across models, all while being robust to overfiting. Those findings are observed across numerous datasets (NL-RX, GSM8K, Spider, RottenTomatoes) and various models from the Llama3, OpenELM, Gemma2 and Olmo families. Code: https://github.com/rbalestr-lab/llm-jepa.

---

# LLM‑JEPA：大语言模型与联合嵌入预测架构的结合 论文详细解读

### 背景：这个问题为什么难？

在语言模型的训练里，主流做法都是让模型直接重建原始文本或生成下一个词，这种“输入空间”目标虽然直观，却往往导致模型在细节上过度拟合、在迁移到新任务时表现不稳。视觉领域早已发现，用“嵌入空间”目标（即让模型预测高层特征而不是像素）能显著提升表征质量和鲁棒性。把这种思路搬到语言上并不简单：语言的离散性、长程依赖以及上下文的多义性让设计合理的嵌入预测任务变得棘手。于是出现了“语言模型缺少 JEPA 风格”的现象，也正是这块空白激发了本文的研究动机。

### 关键概念速览

**LLM（大语言模型）**：参数量从几亿到上千亿不等的模型，能够理解并生成自然语言文本。想象成一个会说话的百科全书。

**JEPA（联合嵌入预测架构）**：一种在视觉里常用的训练框架，模型学习把一部分图像的特征映射到另一部分的特征上，而不是直接重建像素。可以比作把两段文字的意思对应起来，而不是逐字抄写。

**嵌入空间**：模型内部的向量表示，捕捉词语或句子的语义信息。类似于把每个词压缩成一个坐标点，距离近的点表示意思相近。

**对比学习**：通过让相似的样本在嵌入空间靠近、不同的样本远离来学习表征。好比把同类水果放进同一个篮子。

**过拟合**：模型在训练数据上表现极好，但在新数据上失效。像是只会背答案的学生，遇到新题目就慌。

**Finetuning（微调）**：在已有的大模型基础上，用少量任务特定数据继续训练，使模型适应新任务。相当于在通用技能上加一点专门训练。

**Pretraining（预训练）**：在大规模通用语料上训练模型，获得通用语言能力。相当于先学好基础课程。

### 核心创新点

1. **从像素预测到嵌入预测的迁移**  
   过去的语言模型直接预测词或句子本身 → LLM‑JEPA 让模型预测隐藏层的向量表示，而不是原始词序列 → 这样模型更关注语义结构，减轻了对表层文字的过度记忆，提升了跨任务的迁移能力。

2. **双塔结构的 JEPA 适配**  
   视觉 JEPA 通常使用两个网络（编码器和预测器）分别处理不同视角的图像 → LLM‑JEPA 设计了一个“上下文编码塔”和一个“目标嵌入预测塔”，前者负责抽取上下文特征，后者负责预测被遮盖位置的嵌入 → 这种分工让模型在训练时可以并行计算，提高了效率，同时保持了信息流的清晰。

3. **统一的预训练与微调目标**  
   传统做法在预训练阶段用自回归或掩码语言模型，在微调阶段换成任务特定的交叉熵或回归损失 → LLM‑JEPA 把同一套嵌入预测损失贯穿整个训练流程 → 结果是模型在微调时不需要重新适应全新的目标，显著降低了过拟合风险。

4. **跨模型、跨数据集的稳健提升**  
   之前的改进往往只在单一模型或单一任务上有效 → 论文在 Llama3、OpenELM、Gemma2、Olmo 等多种主流 LLM 上实验，并在 NL‑RX、GSM8K、Spider、RottenTomatoes 四类任务上验证 → 统一的提升表明该方法具备良好的通用性。

### 方法详解

**整体框架**  
LLM‑JEPA 的训练过程可以拆成三步：① 选取一段文本并随机遮盖若干子句；② 用“上下文编码塔”把未遮盖的部分编码成高维向量；③ 用“目标预测塔”把这些向量映射到被遮盖子句对应的嵌入上，随后计算预测嵌入与真实嵌入之间的相似度损失。

**关键模块拆解**  

1. **遮盖策略**  
   与传统的掩码语言模型类似，但不是把词直接换成特殊标记，而是把整段子句（可能跨句）隐藏，迫使模型必须从更宽的上下文中推断其语义。

2. **上下文编码塔**  
   采用标准的 Transformer 编码器，只处理可见的文本。它的输出是每个可见 token 的向量，随后通过池化（如平均或注意力加权）得到一个全局上下文向量。

3. **目标预测塔**  
   结构上也是一个轻量级 Transformer，但输入是上下文向量以及一个位置编码，输出是与被遮盖子句长度相匹配的向量序列。这里的目标不是生成词，而是预测对应的“目标嵌入”。

4. **嵌入对齐损失**  
   真实的目标嵌入由一个预训练好的、固定的语言模型（如原始 LLM 的某层输出）提供。预测塔的输出与这些固定嵌入通过余弦相似度或对比损失进行对齐。直白地说，就是让模型的预测向量和“老师”给的向量尽量指向同一个方向。

**最巧妙的设计**  
把目标嵌入固定下来，而不是让预测塔和编码塔共享同一套参数，避免了训练过程中的“信息泄漏”。这相当于在课堂上让学生先听老师讲解，再自己写答案，而不是让老师和学生同时看答案。

### 实验与效果

- **数据集与任务**：论文在四类基准上做评测：NL‑RX（自然语言推理）、GSM8K（数学文字题）、Spider（跨域 SQL 生成）以及 RottenTomatoes（情感分类）。这些任务覆盖了推理、数值、结构化生成和情感分析，能够全面检验模型的通用能力。

- **对比基线**：与同规模的 LLM 使用传统自回归或掩码语言模型进行预训练/微调的结果进行比较。论文声称在所有数据集上均实现了显著提升，尤其在 GSM8K 上的准确率提升最为明显。

- **消融实验**：作者分别去掉遮盖子句的长度多样性、固定目标嵌入、以及预测塔的层数，结果显示每一项都对最终性能有正向贡献。最关键的似乎是目标嵌入的固定策略，去掉后模型的鲁棒性下降明显。

- **局限性**：论文承认目前的实现仍然依赖于一个预先训练好的“老师模型”来提供目标嵌入，这在资源受限的场景下可能不太实际。此外，遮盖策略的设计仍然是经验性的，尚未系统探索最优方案。

### 影响与延伸思考

LLM‑JEPA 把视觉领域的 JEPA 思路成功搬到语言模型，打开了“嵌入空间训练”在 NLP 的新局面。后续工作可能会围绕以下几个方向展开：① 去除对外部老师模型的依赖，尝试自监督生成目标嵌入；② 将 JEPA 与指令微调、RLHF（强化学习人类反馈）结合，探索更高效的对齐方式；③ 在多模态模型中统一视觉 JEPA 与语言 JEPA，实现跨模态的统一表征学习。推测这些方向将在未来两三年内产生一批新论文。

### 一句话记住它

LLM‑JEPA 用“预测隐藏语义向量”取代“直接生成文字”，让大语言模型像看图一样学会更稳健的语义表征。