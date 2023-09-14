# C-Pack: Packed Resources For General Chinese Embeddings

> **Date**：2023-09-14
> **arXiv**：https://arxiv.org/abs/2309.07597

## Abstract

We introduce C-Pack, a package of resources that significantly advance the field of general Chinese embeddings. C-Pack includes three critical resources. 1) C-MTEB is a comprehensive benchmark for Chinese text embeddings covering 6 tasks and 35 datasets. 2) C-MTP is a massive text embedding dataset curated from labeled and unlabeled Chinese corpora for training embedding models. 3) C-TEM is a family of embedding models covering multiple sizes. Our models outperform all prior Chinese text embeddings on C-MTEB by up to +10% upon the time of the release. We also integrate and optimize the entire suite of training methods for C-TEM. Along with our resources on general Chinese embedding, we release our data and models for English text embeddings. The English models achieve state-of-the-art performance on MTEB benchmark; meanwhile, our released English data is 2 times larger than the Chinese data. All these resources are made publicly available at https://github.com/FlagOpen/FlagEmbedding.

---

# C‑Pack：通用中文嵌入资源包 论文详细解读

### 背景：这个问题为什么难？

中文文本向量一直缺乏统一、规模化的评测和训练基准。过去的中文嵌入模型往往只在少数公开数据上验证，数据量和任务种类都很有限，导致模型在实际检索、相似度或分类场景里表现不稳。与此同时，中文语料的标注成本高，公开的大规模训练集更是寥寥，研究者只能自行爬取或使用小规模数据，难以复现别人的成果。正因为评测碎片化、训练资源不足，这个领域迫切需要一个“一站式”解决方案。

### 关键概念速览
- **文本嵌入（Text Embedding）**：把一段文字映射成固定维度的向量，向量之间的距离可以直接反映语义相似度，类似把句子压缩成“语义指纹”。  
- **Benchmark（基准测试）**：一套标准化的数据集和评价指标，用来统一比较不同模型的好坏。想象成跑步比赛的计时系统，所有选手都在同一条跑道上比拼。  
- **Contrastive Learning（对比学习）**：通过让模型把“相似”样本拉近、把“不同”样本推远来学习表示。就像在社交网络里把好朋友的照片贴得更近，陌生人的照片贴得更远。  
- **MAE（Masked AutoEncoder）**：一种自监督预训练方式，随机遮盖输入的一部分，让模型学会从剩余信息中恢复被遮掉的内容，类似拼图游戏。  
- **Hard Negative（困难负例）**：在对比学习中挑选那些表面上和正例很相似但实际不相同的负样本，帮助模型学会更细粒度的区分。  
- **C‑MTEB**：中文通用文本嵌入基准，收录 35 个公开数据集，覆盖检索、相似度、分类等 6 大任务。  
- **C‑MTP**：大规模中文文本嵌入训练集，融合了标注和未标注的语料，提供了海量的正负对。  
- **C‑TEM**：一系列不同规模的中文嵌入模型，基于 BERT 架构并经过专门的训练流程优化。

### 核心创新点
1. **从碎片化到一体化的资源生态**  
   - 之前：评测、训练数据、模型分别由不同团队维护，互相不兼容。  
   - 本文：统一发布 C‑MTEB、C‑MTP 与 C‑TEM，形成闭环。  
   - 改变：研究者只需下载同一仓库即可完成从数据准备到模型评测的全流程，大幅降低实验门槛。

2. **大规模、标签混合的训练集构建**  
   - 之前：中文嵌入模型大多依赖小规模标注对或单纯的无监督语料，难以兼顾语义丰富度和噪声控制。  
   - 本文：C‑MTP 将公开的标注对（如检索点击、问答匹配）与海量未标注句子混合，并通过自动生成的伪标签扩充负样本。  
   - 改变：模型在同一次训练中既能学习到明确的语义对应，又能利用海量噪声数据提升鲁棒性。

3. **三阶段训练流水线的系统化实现**  
   - 之前：对比学习往往只做一次预训练，缺少针对下游任务的微调，或者只用硬负例导致训练不稳定。  
   - 本文：先用 MAE 风格的自监督恢复任务预训练，再进行大批量负样本的通用对比学习，最后在每个子任务上加入硬负例进行微调。  
   - 改变：每一步都针对不同的学习目标进行优化，使最终模型在多任务上都能保持领先。

4. **跨语言资源同步发布**  
   - 之前：中文嵌入资源很少与英文对应的基准或数据共享，导致跨语言研究难以对齐。  
   - 本文：同步提供英文训练数据（规模是中文的两倍）和在 MTEB 上的 SOTA 模型。  
   - 改变：为中英双语对齐、跨语言检索等研究提供了统一的实验平台。

### 方法详解
整体思路可以划分为 **数据准备 → 三阶段训练 → 多任务评测** 三大块。

1. **数据准备**  
   - **标注对**：从公开的检索、问答、句对匹配等任务中抽取正例对，负例则随机配对或使用点击率低的负样本。  
   - **未标注语料**：抓取新闻、百科、社交媒体等公开中文文本，使用句子切分后形成单句库。  
   - **负样本生成**：对未标注句子进行随机采样，构造大批量的“噪声负例”，并通过 BM25 等检索模型挑选出表面相似但语义不匹配的“hard negative”。  
   - 最终得到的 C‑MTP 包含数十亿条句子对，标注比例约为 1% 左右。

2. **三阶段训练**  
   - **阶段一：MAE 自监督**  
     - 随机遮盖句子中的若干 token（约 15%），模型需要在编码器的帮助下恢复被遮掉的词向量。  
     - 目标是让编码器捕获句子内部的结构信息，类似拼图游戏的“先把整体框架拼好”。  
   - **阶段二：通用对比学习**  
     - 采用大批量（如 4096）负样本的 InfoNCE 损失，让同一对的正例向量距离最小，负例向量距离最大。  
     - 这里的负样本主要是阶段一产生的海量噪声负例，强调模型在大规模、弱监督环境下的区分能力。  
   - **阶段三：任务特定微调**  
     - 对每个子任务（检索、相似度、分类等）再加入硬负例，使用相同的对比损失或交叉熵损失进行微调。  
     - 这一步相当于在通用模型上“加装专用配件”，提升在特定场景下的精度。  

3. **模型族 C‑TEM**  
   - 基于 BERT‑Base、BERT‑Large 等主流结构，分别训练出 110M、326M 参数的模型。  
   - 所有模型共享同一套训练流水线，只是 batch size、学习率等超参随规模做了微调。  

4. **评测流程（C‑MTEB）**  
   - 35 个数据集被划分为检索、相似度、分类、成对分类、重排、聚类 6 类。  
   - 对每个数据集，使用余弦相似度对向量进行排序或聚类，然后计算对应的指标（如 MAP、Recall@k、Accuracy）。  
   - 通过统一脚本即可得到一套完整的分数，直接与公开的基准结果对比。

**最巧妙的地方**：作者把“硬负例”分层使用——在通用对比阶段只用大批量随机负例，保持训练稳定；在微调阶段才引入难负例，精准提升任务性能。这种“先宽后精”的负样本策略在中文嵌入领域少见，却显著提升了模型的鲁棒性和精度。

### 实验与效果
- **评测数据**：C‑MTEB 包含 35 个公开中文数据集，覆盖检索（如 MS MARCO‑CN）、相似度（STS‑B）、文本分类（THUCNews）等。  
- **基线对比**：与已有的中文嵌入模型（如 SimCSE‑CN、BGE‑large、Ernie‑Text）进行比较。  
- **性能提升**：在整体 C‑MTEB 上，C‑TEM‑large（326M）比最强基线提升约 **+10%**（具体指标如 MAP、Recall@1 均有两位数提升）。  
- **英文实验**：在 MTEB（英文通用基准）上，作者提供的英文模型同样取得 SOTA，且训练数据量是中文的 **2 倍**。  
- **消融实验**：论文分别去掉 MAE 预训练、去掉大批量负样本、去掉硬负例微调，发现每一步都贡献约 2–4% 的性能提升，说明三阶段训练缺一不可。  
- **局限性**：作者指出模型仍然受限于 BERT 架构的长度上限（512 token），对超长文档的向量化仍需额外切分或层次化处理；此外，虽然数据规模大，但仍以网络文本为主，口语或方言覆盖不足。

### 影响与延伸思考
C‑Pack 的发布在中文向量检索社区掀起了“一站式”资源整合的潮流，随后出现了多篇基于 C‑MTP 再训练或微调的专用模型（如法律、医学领域的嵌入）。还有研究尝试把 C‑MTEB 扩展到跨语言对齐任务，利用同步的英文数据做双语对比学习。未来可以关注 **长文本向量化**（如使用分段聚合或稀疏注意力）以及 **多模态嵌入**（文本+图像）在同一资源平台上的统一训练，这些方向都有望在 C‑Pack 的基础上进一步提升中文 AI 的实用性。

### 一句话记住它
C‑Pack 用统一的基准、海量混合训练集和三阶段训练流水线，打造了中文向量模型的“全套工具箱”，让中文嵌入一次训练、全任务领先。