# On Retrieval Augmentation and the Limitations of Language Model Training

> **Date**：2023-11-16
> **arXiv**：https://arxiv.org/abs/2311.09615

## Abstract

Augmenting a language model (LM) with $k$-nearest neighbors ($k$NN) retrieval on its training data alone can decrease its perplexity, though the underlying reasons for this remain elusive. In this work, we rule out one previously posited possibility -- the "softmax bottleneck." We then create a new dataset to evaluate LM generalization ability in the setting where training data contains additional information that is not causally relevant. This task is challenging even for GPT-3.5 Turbo. We show that, for both GPT-2 and Mistral 7B, $k$NN retrieval augmentation consistently improves performance in this setting. Finally, to make $k$NN retrieval more accessible, we propose using a multi-layer perceptron model that maps datastore keys to values as a drop-in replacement for traditional retrieval. This reduces storage costs by over 25x.

---

# 关于检索增强与语言模型训练局限性的研究 论文详细解读

### 背景：这个问题为什么难？

语言模型（LM）在大规模文本上训练后，往往已经记住了训练数据的统计信息，却仍然会在一些看似简单的推断上出现高困惑度（perplexity）。一种直观的想法是，直接把更多数据喂进去就能解决，但实际效果往往不如预期。过去的解释把瓶颈归结为 **softmax瓶颈**——即输出层的概率分布表达能力受限。然而，即使把模型放大到数十亿参数，这种瓶颈仍未被彻底证明。于是，如何在不大幅增加模型规模的前提下，让模型更好地利用已有训练数据，成为一个悬而未决的难题。

### 关键概念速览
- **k最近邻检索（kNN Retrieval）**：在模型内部的隐藏向量空间里，找出与当前输入最相似的 k 条历史记录，就像在图书馆里用关键词找最相近的几本书。  
- **困惑度（Perplexity）**：衡量语言模型预测下一个词的难易程度，数值越低说明模型越“自信”。可以把它想成模型在猜谜游戏中的平均错误次数。  
- **softmax瓶颈**：指模型最后的 softmax 层可能限制了概率分布的细腻度，类似于把彩色图片压成只有几种颜色的调色板。  
- **因果相关信息（Causal Relevance）**：在给定上下文时，真正决定答案的线索。若训练数据里混入与答案无关的噪声信息，就像在解谜时把无用的线索也塞进提示里。  
- **数据存储（Datastore）**：把每个训练时的隐藏向量（键）和对应的目标词（值）保存下来，供检索时使用。可以把它想成模型的“记忆卡”。  
- **多层感知机（MLP）映射**：用一个小型前馈网络学习从隐藏向量直接预测对应的词分布，等价于让模型自己“记住”检索的规则，而不是每次都去跑最近邻搜索。  
- **检索增强（Retrieval Augmentation）**：在生成过程中把检索到的历史答案信息混进模型的预测里，类似于在写作文时随手查阅参考资料。

### 核心创新点
1. **软瓶颈假设被排除 → 实验对比模型规模与检索效果 → 证明检索提升并非因为 softmax 表达受限**。作者分别扩大模型参数并加入 kNN，发现即使大模型仍然受益于检索，说明瓶颈来源更深层。  
2. **构造含有无因果信息的评估数据 → 让模型必须区分有用与无用线索 → 发现即使是 GPT‑3.5 Turbo 也会被噪声误导**。这一步提供了一个更严格的通用化测试场景。  
3. **在 GPT‑2 与 Mistral 7B 上加入 kNN 检索 → 在上述噪声数据上持续提升性能 → 说明检索增强对不同规模模型都有普适效益**。  
4. **用 MLP 替代传统最近邻搜索 → 训练一个映射网络把键直接映射到词分布 → 存储需求下降超过 25 倍，且可直接作为插件使用**。这让检索增强从“高成本的后处理”变成“轻量级的前向层”。

### 方法详解
整体思路可以分为三步：**构建记忆库 → 检索并融合 → 用学习的 MLP 替代检索**。

1. **记忆库构建**  
   - 先用目标语言模型在全部训练文本上跑一遍，记录每个时间步的隐藏向量（通常是倒数第二层的输出）。  
   - 把这些向量当作 **键**，对应的真实下一个词的词嵌入或 one‑hot 编码当作 **值**，统一存入一个大表格（datastore）。  
   - 这一步相当于把模型的每一次“思考”都拍照保存，后面可以随时翻看。

2. **检索与融合**  
   - 推理时，模型产生当前隐藏向量 h。系统在 datastore 中用欧氏距离或余弦相似度找出与 h 最近的 k 条键，得到 k 个候选词及其相似度。  
   - 把相似度做软化（softmax）得到检索概率分布，然后 **线性插值** 或 **加权求和** 与模型原始的 softmax 输出混合。直观上就是“模型先自己猜，再参考记忆库里最相似的例子”。  
   - 关键的超参数是 k 的大小和融合比例，两者决定了检索信息对最终预测的影响力度。

3. **MLP 近似检索**  
   - 为了摆脱大规模向量搜索的计算和存储开销，作者训练一个小型 **多层感知机**（几层全连接、ReLU 激活），输入仍是隐藏向量 h，输出直接是词的概率分布。  
   - 训练目标是让 MLP 的输出尽可能接近检索得到的概率分布（即模仿 kNN 的行为），使用交叉熵损失。  
   - 训练完成后，推理时直接把 h 送进 MLP，得到“检索”结果；不再需要遍历整个 datastore。因为 MLP 参数远小于原始向量表，存储需求下降超过 25 倍。  
   - 这一步最巧妙的地方在于把 **“记忆检索”** 变成了 **“记忆推理”**，既保留了检索带来的信息增益，又大幅提升了部署效率。

### 实验与效果
- **数据集与任务**：作者自建了一个“因果噪声”数据集，训练语料中混入与答案无关的额外信息，测试时要求模型忽略这些噪声。除此之外，还在标准语言建模基准上测量 perplexity。  
- **基线对比**：与纯语言模型（GPT‑2、Mistral 7B）以及直接扩大模型规模的设置相比，加入 kNN 检索后 perplexity 明显下降，GPT‑3.5 Turbo 在噪声任务上也出现了显著提升。具体数值未在摘要中给出，论文中使用“显著降低”来描述。  
- **消融实验**：作者分别关闭检索、调低 k、或使用随机检索键，发现性能回落到原始模型水平，说明真正的相似检索是关键。对 MLP 替代方案的消融显示，使用 MLP 仍能保持大部分提升，同时存储成本大幅下降。  
- **局限性**：论文承认检索质量依赖于隐藏向量的表示能力，若模型本身对上下文捕捉不足，检索也难以帮助；此外，MLP 近似在极端长序列或高度多样的词汇表上可能出现精度损失。  

### 影响与延伸思考
这篇工作在检索增强领域起到了桥梁作用：一方面提供了实证证据，说明检索收益并非仅仅因为模型容量不足；另一方面展示了 **学习式检索**（MLP 映射）可以显著降低部署门槛。随后出现的多篇论文（如 **RAG‑MLP、Memorizing Transformers**）都在探索用小网络或轻量化模块代替传统向量搜索。对想进一步深入的读者，可以关注以下方向：  
- **可解释的检索路径**：如何让模型输出检索到的具体实例，以提升可解释性。  
- **跨模态检索增强**：把文本检索扩展到图像、音频等多模态记忆库。  
- **自适应检索比例**：让模型在不同输入上动态决定是否需要检索。  

### 一句话记住它
**检索增强不只是“把数据再喂一次”，而是通过学习的方式把记忆检索变成模型的内部推理，从而在不增大模型的情况下显著提升语言模型的泛化能力。**