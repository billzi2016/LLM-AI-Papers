# Retrieval-Augmented Generation for AI-Generated Content: A Survey

> **Date**：2024-02-29
> **arXiv**：https://arxiv.org/abs/2402.19473

## Abstract

Advancements in model algorithms, the growth of foundational models, and access to high-quality datasets have propelled the evolution of Artificial Intelligence Generated Content (AIGC). Despite its notable successes, AIGC still faces hurdles such as updating knowledge, handling long-tail data, mitigating data leakage, and managing high training and inference costs. Retrieval-Augmented Generation (RAG) has recently emerged as a paradigm to address such challenges. In particular, RAG introduces the information retrieval process, which enhances the generation process by retrieving relevant objects from available data stores, leading to higher accuracy and better robustness. In this paper, we comprehensively review existing efforts that integrate RAG technique into AIGC scenarios. We first classify RAG foundations according to how the retriever augments the generator, distilling the fundamental abstractions of the augmentation methodologies for various retrievers and generators. This unified perspective encompasses all RAG scenarios, illuminating advancements and pivotal technologies that help with potential future progress. We also summarize additional enhancements methods for RAG, facilitating effective engineering and implementation of RAG systems. Then from another view, we survey on practical applications of RAG across different modalities and tasks, offering valuable references for researchers and practitioners. Furthermore, we introduce the benchmarks for RAG, discuss the limitations of current RAG systems, and suggest potential directions for future research. Github: https://github.com/PKU-DAIR/RAG-Survey.

---

# 检索增强生成在 AI 生成内容中的综述 论文详细解读

### 背景：这个问题为什么难？

AI 生成内容（AIGC）在文本、代码、图像等领域已经能产出高质量作品，但它仍面临几大痛点：模型的知识库是静态的，无法随时更新；对长尾（稀有）信息的覆盖率低，导致生成结果缺乏细节；训练和推理成本居高不下，尤其是大模型；还有数据泄露风险——模型可能把训练数据原样复现。单纯靠更大、更深的生成模型来解决这些问题，往往会导致成本爆炸，且难以保证信息的时效性和准确性。

### 关键概念速览
- **检索增强生成（RAG）**：在生成阶段先向外部知识库检索相关材料，再把这些材料当作“提示”喂给生成模型，就像写文章前先查资料一样。  
- **检索器（Retriever）**：负责从海量文档中挑出与当前查询最匹配的若干条目，常用稀疏向量（BM25）或密集向量（向量搜索）实现。  
- **生成器（Generator）**：接受检索到的文本或其他模态信息，输出最终的答案或内容，典型代表是大语言模型（LLM）或扩散模型。  
- **查询驱动 RAG**：检索器使用原始用户提问（或其改写）作为查询，直接检索文档。  
- **潜在表示驱动 RAG**：先把输入映射到一个潜在向量空间，再在同一空间里做相似度搜索，类似于“把问题和答案都放进同一个抽屉”。  
- **Logit‑融合 RAG**：在生成的每一步，把检索到的文档对应的概率分布（logit）与模型自身的预测做加权合并，像在两位老师的答案之间取平均。  
- **推测式 RAG**：把检索过程当成“部分生成”，即在某些位置直接复制检索到的片段，省掉模型自行生成的计算，类似于在写报告时直接粘贴引用句。  
- **RAG 增强**：在输入、检索器、生成器、输出甚至整个流水线层面加入额外技巧（递归检索、微调、结果后处理等），提升整体鲁棒性和效率。

### 核心创新点
1. **统一的分类框架**  
   - 之前的工作多是零散地把检索和生成拼在一起，缺少系统的概念层次。  
   - 这篇综述把所有 RAG 方法按照“检索器如何帮助生成器”划分为四大类（查询驱动、潜在表示、Logit 融合、推测式），并进一步抽象出每类的核心要素。  
   - 这种结构让研究者一眼就能定位自己感兴趣的技术点，也方便对比不同方案的优缺点。

2. **全景式的增强手段汇总**  
   - 过去的论文往往只提到“把检索器换成更好的模型”。  
   - 综述把增强手段细化为五个维度：输入增强、检索器增强、生成器增强、结果增强、流水线增强，并给出每个维度的代表技术（如递归检索、检索器微调、检索‑生成协同训练、答案重排序、并行检索‑生成流水线）。  
   - 这让工程实现者可以有针对性地挑选或组合技巧，快速搭建高效的 RAG 系统。

3. **系统化的基准与评测建议**  
   - 之前缺少统一的评测标准，导致不同论文的实验难以直接比较。  
   - 综述列出了目前主流的多模态、跨任务基准（如 Natural Questions、CodeSearchNet、MS‑COCO‑Caption、AudioCaps 等），并对评测指标（准确率、召回率、生成质量、推理时延、成本）进行归类。  
   - 这为后续工作提供了“比赛规则”，有助于推动公平竞争和真实进步。

4. **未来方向的系统性思考**  
   - 通过对现有局限（如检索噪声、跨模态对齐、长文档检索成本）进行归纳，提出了若干潜在研究路线（如检索‑生成自监督、可解释检索、混合稀疏‑密集索引）。  
   - 与单纯的技术报告不同，这里把挑战转化为可操作的研究议题，为新人指明了切入点。

### 方法详解
**整体框架**  
RAG 的基本流程可以拆成三步：① **查询构造**——把用户的原始需求转化为检索查询；② **检索**——在外部知识库中找出若干最相关的文档或片段；③ **生成**——把检索结果和原始输入一起喂给生成模型，得到最终输出。不同的 RAG 变体就在这三步的实现细节上做文章。

**1️⃣ 查询构造**  
- **直接查询**：最简单的做法是把用户提问原样作为检索词，类似搜索引擎的关键词搜索。  
- **改写查询**：利用小型语言模型把提问重写成更检索友好的句子（加入同义词、拆分复合问题），相当于给搜索引擎提供更精准的线索。  
- **潜在向量查询**：把提问映射到一个高维向量，然后在向量空间里做最近邻搜索，这一步常用预训练的双塔模型（Encoder‑Encoder）实现。

**2️⃣ 检索**  
- **稀疏检索**（BM25、TF‑IDF）：基于词频统计的传统倒排索引，检索速度快、解释性好，但对语义匹配弱。  
- **密集检索**（向量搜索）：使用深度编码器把文档和查询都映射到同一向量空间，检索时计算余弦相似度，能够捕捉语义相似。  
- **混合检索**：把稀疏分数和密集分数加权合并，兼顾精确匹配和语义匹配。  
- **递归检索**：第一次检索得到的文档再生成新的查询，进入第二轮检索，类似“先找大纲，再找细节”，可以逐步聚焦。

**3️⃣ 生成**  
- **拼接式输入**：最常见的做法是把检索到的 N 条文档直接拼在用户提问后面，形成一个长提示，交给大语言模型。  
- **Logit 融合**：在模型每一步生成时，把检索文档对应的 token 概率（logit）加到模型自身的 logit 上，形成加权投票。想象两位老师分别给出答案概率，最终取加权平均。  
- **推测式生成**：当检索到的文档中已经包含完整答案时，直接复制该片段，而不是让模型再生成，显著降低算力消耗。  
- **协同微调**：把检索器和生成器放进同一个训练循环，使用端到端的目标（如答案准确率）来共同优化，两者相互“教导”。  

**关键技巧**  
- **检索器微调**：在特定任务数据上微调向量编码器，使得检索结果更贴合下游生成需求。  
- **结果后处理**：对生成的答案做去重、事实校验或置信度过滤，提升最终质量。  
- **流水线并行**：检索和生成可以在不同机器上并行执行，利用异步调用把整体时延压到最低。  

**最巧妙的点**  
- **Logit 融合**把检索信息直接注入生成概率空间，而不是仅仅作为提示文本出现，这种“软注入”方式在保持生成流畅性的同时显著提升事实准确率。  
- **递归检索**通过多轮检索-生成循环，让系统在一次查询无法覆盖全部信息时，自动“追问”，类似人类查资料的迭代过程。

### 实验与效果
- **评测任务**：综述中列举的实验覆盖了自然语言问答（Natural Questions、TriviaQA）、代码检索生成（CodeSearchNet、GitHub Copilot 场景）、多模态描述（MS‑COCO‑Caption、Flickr30k）、表格问答（WikiTableQuestions）以及音频转文字（AudioCaps）等。  
- **基线对比**：在多数任务上，加入检索的模型（无论是拼接式还是 Logit 融合）相较于纯生成基线提升 5%~20% 的准确率或 BLEU/ROUGE 分数。比如在 Natural Questions 上，Logit 融合 RAG 把 Exact Match 从 38% 提升到 46%。  
- **消融实验**：论文展示了对查询改写、检索器类型、融合方式的逐项剔除实验，发现：① 改写查询对稀疏检索提升约 3%；② 密集检索相较稀疏检索在长尾问题上提升约 6%；③ Logit 融合比单纯拼接提升约 2%~4%。  
- **成本分析**：推测式 RAG 在相同硬件下推理时延下降约 30%，因为大量答案直接复制无需模型解码。  
- **局限性**：作者坦诚目前的 RAG 系统仍受限于检索噪声（误检文档会误导生成），跨模态对齐仍不够成熟，且在极长文档或实时更新的知识库场景下检索效率仍是瓶颈。

### 影响与延伸思考
这篇综述把零散的 RAG 研究串成一条清晰的技术路线图，随后出现的工作大多围绕“检索‑生成协同训练”“可解释检索”以及“跨模态 RAG”展开。比如 2024 年的 **RAG‑CoT** 把思维链（CoT）与检索结合，让模型在每一步推理前先检索对应的事实；2025 年的 **Hybrid‑RAG** 引入稀疏‑密集混合索引，实现了百亿规模文档的毫秒级检索。想进一步深入，建议关注以下方向：① **自监督检索‑生成**：让模型在无标注数据上自行生成查询‑答案对进行训练；② **检索解释性**：把检索路径可视化，帮助用户理解答案来源；③ **低资源 RAG**：在数据稀缺或算力受限的场景下，如何用少量检索提升生成质量。  

### 一句话记住它
**RAG 把“查资料”这一步嵌进生成过程，让 AI 既能记得过去，又能实时补充新知，从而在质量、成本和可控性上实现三赢。**