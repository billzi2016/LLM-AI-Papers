# Barack's Wife Hillary: Using Knowledge-Graphs for Fact-Aware Language   Modeling

> **Date**：2019-06-17
> **arXiv**：https://arxiv.org/abs/1906.07241

## Abstract

Modeling human language requires the ability to not only generate fluent text but also encode factual knowledge. However, traditional language models are only capable of remembering facts seen at training time, and often have difficulty recalling them. To address this, we introduce the knowledge graph language model (KGLM), a neural language model with mechanisms for selecting and copying facts from a knowledge graph that are relevant to the context. These mechanisms enable the model to render information it has never seen before, as well as generate out-of-vocabulary tokens. We also introduce the Linked WikiText-2 dataset, a corpus of annotated text aligned to the Wikidata knowledge graph whose contents (roughly) match the popular WikiText-2 benchmark. In experiments, we demonstrate that the KGLM achieves significantly better performance than a strong baseline language model. We additionally compare different language model's ability to complete sentences requiring factual knowledge, showing that the KGLM outperforms even very large language models in generating facts.

---

# 巴拉克的妻子希拉里：利用知识图谱进行事实感知语言建模 论文详细解读

### 背景：这个问题为什么难？

语言模型擅长捕捉句法规律，却只能记住训练时出现的事实，遇到未见或随时间变化的知识时往往会“卡壳”。传统的自回归模型只能在词表里挑词，无法直接引用外部结构化信息，导致生成的句子在事实层面常常出错。更糟的是，模型内部的知识是固化在参数里，想要更新某条事实（比如美国总统换了）就得重新训练，这在实际系统里几乎不可行。于是，如何让语言模型在保持流畅性的同时，能够随时查阅、复制并生成最新的事实，成为了迫切需要解决的难题。

### 关键概念速览
- **语言模型（LM）**：预测下一个词的概率分布的模型，像是给文字加上“下一个会说什么”的直觉。  
- **知识图谱（KG）**：由实体（节点）和属性关系（边）组成的结构化数据库，例如 Wikidata 把“巴拉克·奥巴马”与“美国总统”用一条边连起来。  
- **实体链接（Entity Linking）**：把文本中的词或短语对应到 KG 中的具体实体，就像在文章里给每个专有名词加上超链接。  
- **局部知识图（Local KG）**：在生成过程中只挑出与当前上下文最相关的若干实体和它们的邻居，形成一个小型、可查询的子图。  
- **复制机制（Copy / Pointer）**：模型在生成词时可以直接“拷贝” KG 中的实体属性，而不是从固定词表里选，类似写作者直接引用百科条目中的原句。  
- **TransE 嵌入**：把 KG 中的实体和关系映射到向量空间，使得向量运算能够近似表示图结构的语义相似度，像是把概念投射到坐标轴上方便计算。  
- **Linked WikiText-2**：在原始 WikiText-2 文本上做了实体链接并对齐到 Wikidata 的数据集，提供了“文本 ↔ KG”对齐的训练信号。  

### 核心创新点
1. **从纯词预测到“词+事实”双预测**  
   - 传统 LM 只预测下一个词的概率。  
   - KGLM 在每一步先判断当前上下文是否需要引用实体属性，如果需要，就从 KG 中挑选最可能的属性并复制；否则回退到普通词预测。  
   - 这样模型能够在未见过的实体上生成准确事实，显著提升了事实完整性。

2. **动态构建局部知识图**  
   - 以前的模型要么把整个 KG 直接喂进去，要么根本不使用结构化信息，导致计算成本高或信息稀疏。  
   - KGLM 先通过实体链接得到当前句子涉及的实体，然后只把这些实体的 1‑hop（或 2‑hop）邻居拉进一个小图，作为查询空间。  
   - 这种“只取所需”策略大幅降低了检索开销，同时保证了事实的相关性。

3. **基于 TransE 的统一向量空间**  
   - 直接把 KG 的符号信息喂进 LSTM 会导致表示不一致。  
   - 作者先用 TransE 把实体和关系映射到向量，再把这些向量和普通词向量拼接进 LSTM 的输入层。  
   - 统一的向量空间让语言模型能够自然地在词汇和结构化知识之间切换。

4. **Linked WikiText-2 数据集的构建**  
   - 训练需要标注好的实体链接，但公开的语言模型基准几乎没有这类标注。  
   - 论文自行对 WikiText-2 进行实体链接并对齐到 Wikidata，形成了一个规模可比的、同时拥有文本和 KG 信息的训练集。  
   - 该数据集为后续研究提供了可复用的基准。

### 方法详解
**整体思路**  
KGLM 的生成过程可以看作三步循环：① 实体检测 → ② 局部 KG 查询 → ③ 复制或普通词预测。模型的核心是一个基于 LSTM 的解码器，解码器的隐藏状态既接受普通词向量，也接受从局部 KG 中检索到的实体向量。

**1. 实体检测与局部 KG 构建**  
- 在每个时间步，模型先用一个轻量级的实体链接器（比如基于字典的匹配）判断当前已生成的上下文是否已经提到了某个已知实体。  
- 若检测到实体 *e*，模型把 *e* 在 KG 中的邻居（属性关系 *r* 和目标实体 *o*）取出，形成一个小的子图 *G_local*。  
- 为了控制预算，作者只保留概率最高的 *k* 条边，类似于编辑器只打开最相关的参考文献。

**2. 事实选择（Attention over Local KG）**  
- 解码器的隐藏状态 *h_t* 通过注意力机制对 *G_local* 中的每条边打分，得分高的边代表当前上下文最需要的事实。  
- 这里的注意力不是普通的词向量注意力，而是对 **实体向量 + 关系向量** 的组合打分，类似于在百科全书里快速定位最相关的章节。

**3. 复制 vs. 生成**  
- 计算得到的注意力分布被映射成一个 **复制概率** *p_copy*。  
- 若 *p_copy* 大于阈值，模型直接把对应的目标实体 *o*（或属性值）拷贝到输出序列；否则进入普通的词表软最大（softmax）层，像传统 LM 那样生成下一个词。  
- 复制机制的实现方式与指针网络相似，只是拷贝的对象是 KG 中的实体标识，而不是原始文本中的词。

**4. 融合向量表示**  
- 实体和关系的 TransE 向量与普通词向量在同一维度上相加或拼接，送入 LSTM。这样 LSTM 的隐藏状态天然携带了结构化知识的语义信息。  
- 训练时，模型同时最小化两部分损失：普通词预测的交叉熵 + 复制目标的二元交叉熵，确保两条路径都能学到有效的梯度。

**最巧妙的地方**  
- **局部 KG 动态裁剪**：只在需要时才打开“知识库的门”，避免了全图检索的高昂计算成本。  
- **统一向量空间**：通过 TransE 把离散的图结构平滑成连续向量，使得 LSTM 能够无缝处理两种不同来源的信息。  
- **复制机制的双向选择**：模型自行决定是“引用事实”还是“自由发挥”，这让它在生成新实体时仍保持流畅。

### 实验与效果
- **数据集**：作者在新构建的 Linked WikiText-2 上训练和评估，数据规模与原始 WikiText-2 基本持平，只是每篇文章都附带了实体到 Wikidata 的对齐信息。  
- **基线**：与一个强大的 LSTM 语言模型（同等参数量、无 KG）以及若干基于检索的增强模型进行比较。  
- **主要指标**：在困惑度（perplexity）上，KGLM 明显低于基线，论文声称提升幅度在“显著”层面；在事实填空任务（给定句子前半句，要求模型补全包含特定实体属性的后半句）中，KGLM 的准确率超过了同等规模的 Transformer‑XL，甚至赶超了数十亿参数的大模型。  
- **消融实验**：去掉局部 KG、去掉 TransE 嵌入或关闭复制机制都会导致性能回落，尤其是复制机制的缺失使得模型在生成未出现过的实体时几乎失效，验证了该模块的关键作用。  
- **局限**：训练过程需要高质量的实体链接标注，构建 Linked WikiText-2 本身成本不低；此外，模型仍基于 LSTM，扩展到更大规模的 Transformer 仍是未解之题。

### 影响与延伸思考
KGLM 把“语言模型 + 知识图谱”这套组合模式推向了实用化，随后出现了多篇工作尝试在 Transformer、GPT 系列上加入类似的检索或复制机制（如 Retrieval‑Augmented Generation、RAG）。还有研究把更复杂的图神经网络（GNN）替代 TransE，以捕捉多跳关系的细粒度语义。对想进一步探索的读者，可以关注以下方向：  
- **大模型与外部 KG 的高效对接**：如何在数十亿参数的模型中保持低延迟的 KG 查询。  
- **时序知识图谱**：把随时间变化的事实（如政治职位）建模为时间切片，让模型能够自动“追踪历史”。  
- **跨语言 KG 对齐**：把多语言的 Wikidata 与多语言语言模型结合，实现跨语言事实生成。  

### 一句话记住它
KGLM 让语言模型在写句子时可以像查百科一样即时检索并复制知识图谱中的事实，从而生成既流畅又可靠的文本。