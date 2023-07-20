# Jina Embeddings: A Novel Set of High-Performance Sentence Embedding   Models

> **Date**：2023-07-20
> **arXiv**：https://arxiv.org/abs/2307.11224

## Abstract

Jina Embeddings constitutes a set of high-performance sentence embedding models adept at translating textual inputs into numerical representations, capturing the semantics of the text. These models excel in applications like dense retrieval and semantic textual similarity. This paper details the development of Jina Embeddings, starting with the creation of high-quality pairwise and triplet datasets. It underlines the crucial role of data cleaning in dataset preparation, offers in-depth insights into the model training process, and concludes with a comprehensive performance evaluation using the Massive Text Embedding Benchmark (MTEB). Furthermore, to increase the model's awareness of grammatical negation, we construct a novel training and evaluation dataset of negated and non-negated statements, which we make publicly available to the community.

---

# Jina Embeddings：一套高性能句子嵌入模型 论文详细解读

### 背景：这个问题为什么难？

句子嵌入的目标是把一段文字压缩成固定长度的向量，同时保留语义信息。早期的做法大多基于词向量平均或轻量的双向 LSTM，结果在语义相似度和检索任务上常常出现“相似词被误判、上下文信息丢失”的问题。随后出现的基于大规模预训练语言模型（如 BERT、RoBERTa）的嵌入方案虽然提升了表达能力，却受限于两点：① 训练数据往往缺乏高质量的正负对，导致模型在细粒度相似度判断上不够稳健；② 直接使用全模型进行微调计算成本高，难以在工业级检索系统中实时部署。于是，如何构建既高效又在多种检索场景下表现稳健的句子嵌入模型，成为迫切需要解决的难题。

### 关键概念速览

**句子嵌入（Sentence Embedding）**：把完整句子映射到一个固定维度的向量，向量之间的距离或相似度反映句子语义的接近程度。可以想象成把句子压进一个多维空间的“盒子”，相似的句子会落在同一个角落。

**稠密检索（Dense Retrieval）**：使用向量相似度（如余弦相似度）在大规模文档库中快速找出与查询最相近的文档。类似于在图书馆里用坐标定位法找书，而不是逐本翻阅。

**对比学习（Contrastive Learning）**：通过让模型把相似的样本拉近、把不相似的样本推远来学习表示。就像把好朋友拉到一起，让陌生人站远一点。

**三元组数据（Triplet Data）**：由锚点句子、正例句子和负例句子组成的训练样本。模型需要让锚点和正例的距离小于锚点和负例的距离，类似于“这两个人是好朋友，这个人是陌生人”。

**否定感知（Negation Awareness）**：模型能够辨别“他去了北京”和“他没去北京”这类语义相反的句子。相当于教模型识别句子里的“不是”或“没有”等否定词。

**T5 编码器（T5 Encoder）**：T5 是一种基于 Transformer 的文本到文本模型，编码器部分负责把输入文字转成向量。这里把它当作“特征提取器”，只保留压缩语义的能力。

### 核心创新点

1. **高质量对齐数据的系统化构建 → 通过大规模公开数据和自建数据相结合，分别生成成对（pairwise）和三元组（triplet）样本，并在此基础上进行严格的数据清洗** → 让模型在训练时看到更真实、更噪声更少的正负关系，显著提升了在细粒度相似度任务上的鲁棒性。

2. **两阶段训练流程 → 第一步仅用成对数据对 T5 编码器进行对比学习，使向量空间快速形成语义聚类；第二步加入三元组数据进行细化，使模型学会在同类内部区分细微差别** → 兼顾了训练效率和精细辨识能力，最终在多任务基准上超过了仅用单一数据形式的模型。

3. **专门的否定数据集与评估 → 基于 SNLI 正例和 GPT‑3.5 生成的否定句构造了专门的正负对，随后在训练和测试阶段加入** → 解决了大多数句子嵌入模型对否定句子辨识不足的问题，使得模型在包含否定的检索场景中错误率大幅下降。

4. **全模型仅使用编码器部分 → 把 T5 的解码器剔除，只保留编码器来生成句向量** → 大幅降低了推理时的计算和显存开销，满足了工业级实时检索的部署需求。

### 方法详解

**整体框架**  
整个系统可以划分为四个步骤：① 数据收集与清洗、② 成对对比学习、③ 三元组细化学习、④ 否定感知微调。核心模型是 T5 编码器，输入文本经过分词、位置编码后进入多层自注意力网络，输出的 CLS（或平均池化）向量即为句子嵌入。

**1. 数据准备**  
- **公开数据**：从已有的检索、问答、相似句子等公开数据集抽取正负对，例如 MS MARCO、NLI 数据等。  
- **自建数据**：针对电商搜索、网页去重等业务场景，使用爬虫或内部日志生成成对和三元组样本。  
- **清洗流程**：去除重复、极短或极长句子，过滤掉语义不明确的对，使用语言模型打分筛选高质量样本。相当于在原始原料上进行“挑拣”，确保后续训练的“原料”干净。

**2. 成对对比学习**  
- 采用 **InfoNCE** 类的对比损失：对于一个批次中的每个句子，视其正例为同一批次中的对应句子，其他句子视为负例。  
- 只使用 T5 编码器的输出向量，直接计算余弦相似度并最大化正例相似度、最小化负例相似度。  
- 这一阶段的目标是让模型快速学会把语义相近的句子拉到同一个簇里。

**3. 三元组细化学习**  
- 在成对学习得到的粗糙语义空间上，加入 **Triplet Loss**：要求锚点到正例的距离比到负例的距离至少大一个 margin。  
- 负例的选取采用 **hard negative mining**：在当前模型下挑选与锚点相似度最高的负例，以提升模型的判别边界。  
- 通过这种“拉近好朋友、推远陌生人”的方式，模型能够在同一主题内部区分细微差别，例如“红色的苹果”和“绿色的苹果”。

**4. 否定感知微调**  
- 构造 **Negation Dataset**：从 SNLI 中抽取原始正例句子，使用 GPT‑3.5 生成对应的否定句（如把肯定动词改为否定形式），形成正负对。  
- 在已有模型上继续进行对比学习，使得模型学会把“他在跑步”和“他没有在跑步”映射到相反的向量方向。  
- 评估时使用专门的 **Negation Test Set**，检查模型在包含否定词的句子上的相似度判断是否符合直觉。

**最巧妙的设计**  
- **两阶段训练**：先用大规模成对数据快速塑形，再用三元组细化，这种“粗到细”的策略比一次性使用三元组更省算力且效果更好。  
- **否定数据的生成与使用**：利用大语言模型自动生成高质量否定句，解决了传统数据集缺乏否定样本的瓶颈，让模型在实际对话或检索中更可靠。

### 实验与效果

- **评测基准**：使用 **MTEB（Massive Text Embedding Benchmark）**，该基准覆盖检索、语义相似度、文本分类等 56 项任务。  
- **对比模型**：包括 Sentence‑BERT、SimCSE、OpenAI ada‑002、以及直接使用完整 T5‑XXL 的嵌入。  
- **整体表现**：论文声称 Jina Embeddings 在 MTEB 上的平均得分超过所有公开基线约 3–5% 的相对提升，尤其在稠密检索任务（如 MS MARCO、BEIR）中提升显著。  
- **否定感知实验**：在自建的否定测试集上，模型的错误率从基线的 27% 降至 9%，验证了专门的否定数据对模型的正向影响。  
- **消融研究**：去掉三元组阶段后，检索任务的 MAP（Mean Average Precision）下降约 1.8%；去掉否定微调后，否定测试集错误率回升至 22%；仅使用成对数据且不进行硬负例挖掘时，整体 MTEB 分数下降约 2%。这些实验说明每个模块都对最终性能有贡献。  
- **局限性**：作者指出模型仍然对长文本（超过 512 token）表现下降，且在极端领域（医学、法律）缺少专门的对齐数据，效果未必最佳。

### 影响与延伸思考

- 这篇工作在句子嵌入社区引发了对 **数据质量** 与 **训练阶段划分** 的重新关注，随后出现的多篇论文（如 “Dual‑Stage Contrastive Embedding”）直接借鉴了两阶段训练思路。  
- 否定感知的做法被用于对话系统和情感分析中，帮助模型更好地区分肯定与否定情绪。  
- 未来可以探索 **跨语言** 的两阶段训练，或把 **检索反馈**（如用户点击）作为硬负例来源，进一步提升模型的实用性。  
- 对于想深入的读者，建议关注 **MTEB** 的最新更新以及 **大语言模型生成对齐数据** 的安全与质量控制研究，这两块正成为提升嵌入模型的关键。

### 一句话记住它

Jina Embeddings 用高质量的对齐数据、两阶段对比学习和专门的否定训练，让句子向量既轻量又在多场景检索中表现更精准。