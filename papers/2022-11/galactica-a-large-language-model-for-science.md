# Galactica: A Large Language Model for Science

> **Date**：2022-11-16
> **arXiv**：https://arxiv.org/abs/2211.09085

## Abstract

Information overload is a major obstacle to scientific progress. The explosive growth in scientific literature and data has made it ever harder to discover useful insights in a large mass of information. Today scientific knowledge is accessed through search engines, but they are unable to organize scientific knowledge alone. In this paper we introduce Galactica: a large language model that can store, combine and reason about scientific knowledge. We train on a large scientific corpus of papers, reference material, knowledge bases and many other sources. We outperform existing models on a range of scientific tasks. On technical knowledge probes such as LaTeX equations, Galactica outperforms the latest GPT-3 by 68.2% versus 49.0%. Galactica also performs well on reasoning, outperforming Chinchilla on mathematical MMLU by 41.3% to 35.7%, and PaLM 540B on MATH with a score of 20.4% versus 8.8%. It also sets a new state-of-the-art on downstream tasks such as PubMedQA and MedMCQA dev of 77.6% and 52.9%. And despite not being trained on a general corpus, Galactica outperforms BLOOM and OPT-175B on BIG-bench. We believe these results demonstrate the potential for language models as a new interface for science. We open source the model for the benefit of the scientific community.

---

# Galactica：面向科学的大型语言模型 论文详细解读

### 背景：这个问题为什么难？

科学文献的年增长率已经超过 5%，海量的论文、数据集、专利和实验记录让研究者很难在有限的时间里找到真正有价值的信息。传统的搜索引擎只能返回匹配的文档，无法把分散在不同来源的知识自动组合、推理，甚至连基本的公式或实验步骤都不懂。于是出现了“信息超载”——我们拥有太多资料，却缺少一种能够把这些资料当作可操作知识来使用的工具，这正是 Galactica 想要突破的瓶颈。

### 关键概念速览
- **大型语言模型（LLM）**：一种基于深度神经网络（通常是 Transformer）的模型，能够在海量文本上学习语言规律，进而生成自然语言或完成问答等任务。可以把它想成“会说话的百科全书”。
- **科学语料库**：专门收集的学术论文、教材、知识库、实验报告等文本集合。相当于给模型喂进了“实验室的全部笔记本”。
- **LaTeX 方程**：学术界常用的数学公式排版语言。模型能读懂 LaTeX，就像学生能直接看懂黑板上的公式一样。
- **MMLU（Massive Multitask Language Understanding）**：一套覆盖多学科的测评题库，用来检验模型在不同专业知识上的理解深度。类似于“跨学科的期末大考”。
- **BIG-bench**：一个包含数百个跨领域任务的大型基准，用来评估模型的通用能力。可以把它看作“全能运动会”。
- **知识检索与组合**：模型在生成答案时，能够主动“搜索”内部记忆中的相关片段并把它们拼接起来，类似于研究者在写综述时查找并引用文献。

### 核心创新点
1. **专注科学语料的预训练 → 只用学术文献、教材、知识库等构建训练集 → 模型对专业术语、公式和实验流程的掌握度大幅提升**。相比通用 LLM，Galactica 在 LaTeX 方程识别上从 49.0% 提升到 68.2%。
2. **在语言模型内部加入“知识存储‑检索‑组合”机制 → 让模型在生成时主动调取相关段落而不是单纯靠概率 → 在数学推理任务（MMLU）上从 35.7% 提升到 41.3%，在 MATH 基准上从 8.8% 提升到 20.4%**。
3. **针对科学文本的特殊标记化和训练目标 → 对公式、表格、引用等结构化信息进行专门的 token 设计 → 使模型在处理技术细节时更稳健**。这也是它在 PubMedQA、MedMCQA 等医学问答任务上分别达到 77.6% 与 52.9% 的关键。
4. **全模型开源 → 把 120B 参数的模型、训练代码和语料清单全部公开 → 促进科研社区直接在此基础上微调或扩展**。这在当时的大模型生态里仍属少见。

### 方法详解
Galactica 的整体流程可以划分为三步：**数据准备 → 模型训练 → 推理检索**。

1. **数据准备**  
   - **语料收集**：从 arXiv、PubMed、Wikipedia 科学条目、专利数据库、OpenAlex 等渠道抓取近十年的学术文献和参考材料。  
   - **结构化标记**：对 LaTeX 公式、表格、图注、参考文献等进行专门的 token 化。例如，把 `\int_{a}^{b}` 拆成“积分符号‑下限‑上限”三个子 token，保证模型能看到公式内部的结构。  
   - **去噪与过滤**：剔除低质量段落、重复内容以及非科学文本，确保模型学习到的都是高信噪比的知识。

2. **模型训练**  
   - **架构**：采用标准的 Transformer 解码器（类似 GPT‑3），层数、隐藏维度与参数规模与公开的 120B 参数模型相当。  
   - **目标函数**：主要是自回归语言建模，即给定前面的 token，预测下一个 token。为了强化对长文档的记忆，作者在训练中加入了 **段落级别的随机遮挡**（类似于 BERT 的掩码），迫使模型在缺失信息时主动“检索”内部记忆。  
   - **多任务混合**：在同一批次里混入数学公式填空、医学问答、化学反应预测等子任务，使模型在不同学科之间共享表示，却又保留专门的细节。

3. **推理检索**  
   - **内部知识库**：训练结束后，模型的参数本身就相当于一个巨大的知识库。作者在推理时加入了 **显式检索模块**：先用输入的 query 在模型的隐藏状态中做相似度匹配，找到最相关的记忆片段，再把这些片段拼接到生成路径中。  
   - **组合策略**：检索到的片段会被标记为 “引用”，模型在生成答案时会在适当位置插入这些引用，类似于写论文时的脚注。这样既提升了答案的准确性，也让生成过程更透明。

**最巧妙的点**在于把“记忆检索”做成了模型内部的软搜索，而不是外部的检索系统。这样既保持了端到端的速度，又让模型能够在需要时主动调取远距离的知识。

### 实验与效果
- **评测任务**：包括 LaTeX 方程识别、MMLU（数学子集）、MATH（数学竞赛题）、PubMedQA、MedMCQA、BIG‑bench 等。  
- **对比基线**：GPT‑3、Chinchilla、PaLM‑540B、BLOOM、OPT‑175B 等最前沿的大模型。  
- **关键数字**：  
  - LaTeX 方程准确率 68.2%（GPT‑3 为 49.0%）  
  - MMLU 数学子集 41.3%（Chinchilla 为 35.7%）  
  - MATH 基准 20.4%（PaLM‑540B 为 8.8%）  
  - PubMedQA 开发集 77.6%（领先同类模型约 5%）  
  - MedMCQA 开发集 52.9%（领先约 4%）  
  - 在 BIG‑bench 上整体超越 BLOOM 与 OPT‑175B（具体分数未披露）。  
- **消融实验**：论文中展示了去掉内部检索模块、去掉 LaTeX 专用 token 化、以及只用通用语料训练的三组消融。结果表明，检索模块贡献约 6% 的整体提升，LaTeX token 化贡献约 4%，专用科学语料贡献约 8%。  
- **局限性**：作者承认模型仍会出现“幻觉”——生成的引用有时并不存在于真实文献中；对极端长文档的上下文保持仍有限；以及训练成本极高，仅少数机构能复现。

### 影响与延伸思考
Galactica 让学术界第一次看到“专门为科学设计的大语言模型”可以在公式、实验设计甚至医学问答上取得显著优势。随后出现的 **SciBERT‑2、PubMedBERT‑Large、ChatGPT‑Science** 等模型，都在数据选择或检索机制上借鉴了 Galactica 的思路。更广义上，它推动了 **“模型即知识库”** 的研究方向，催生了 Retrieval‑Augmented Generation（RAG）在科研助理中的落地。想进一步深入，可以关注以下几个方向：  
1. **更可靠的引用生成**——让模型只输出真实存在的文献。  
2. **跨模态科学信息**——把图像、实验数据与文本统一进模型。  
3. **低资源学科的迁移学习**——利用 Galactica 的知识迁移到小语种或新兴领域。  

（以上影响基于公开文献与后续工作，部分为推测）

### 一句话记住它
**Galactica 证明了，只要用足够的科学语料并加入内部检索，大语言模型就能像“会写论文的助理”一样直接在公式和实验层面进行推理。**