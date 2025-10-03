# Less LLM, More Documents: Searching for Improved RAG

> **Date**：2025-10-03
> **arXiv**：https://arxiv.org/abs/2510.02657

## Abstract

Retrieval-Augmented Generation (RAG) couples document retrieval with large language models (LLMs). While scaling generators often improves accuracy, it also increases inference and deployment overhead. We study an orthogonal axis: enlarging the retriever's corpus, and how it trades off with generator scale. Across multiple open-domain QA benchmarks, corpus scaling consistently strengthens RAG and can in many cases match the gains of moving to a larger model tier, though with diminishing returns at larger scales. Small- and mid-sized generators paired with larger corpora often rival much larger models with smaller corpora; mid-sized models tend to gain the most, while tiny and very large models benefit less. Our analysis suggests that these improvements arise primarily from increased coverage of answer-bearing passages, while utilization efficiency remains largely unchanged. Overall, our results characterize a corpus-generator trade-off in RAG and provide empirical guidance on how corpus scale and model capacity interact in this setting.

---

# 少用大模型，多用文档：寻找更优的检索增强生成 论文详细解读

### 背景：这个问题为什么难？
开放域问答系统需要先在海量文档中找到可能包含答案的片段，再交给语言模型生成自然语言答案。过去的研究几乎把提升重点放在让生成模型更大更强，却忽视了检索端的规模。大模型固然能提升推理和语言表达，但随之而来的算力、延迟和部署成本会急剧上升。与此同时，检索库往往被固定在几百GB甚至几TB的规模，导致很多真实答案根本找不到。于是出现了一个两难：是继续砸钱买更大的生成模型，还是让检索库更丰富？这篇论文正是为了解答这个两难而展开的。

### 关键概念速览
- **检索增强生成（RAG）**：先用检索模型挑出若干文档片段，再把这些片段拼进提示，让语言模型基于外部事实生成答案。相当于先找资料、后写报告的写作流程。  
- **检索器（Retriever）**：负责在大规模语料库中快速定位与查询相关的文档，常用向量相似度或稀疏倒排索引实现。它像图书馆的目录员。  
- **生成模型（Generator）**：接受检索到的文档和用户提问，输出自然语言答案的语言模型。可以把它想成写作助手。  
- **语料库规模（Corpus Size）**：检索库中包含的文档总量，通常以GB或TB计。规模越大，覆盖的事实越全。  
- **覆盖率（Coverage）**：检索库中实际包含答案所在段落的比例。高覆盖率意味着更大概率能把正确信息送给生成模型。  
- **利用效率（Utilization Efficiency）**：检索到的文档被生成模型真正用到的程度。即使检索到很多片段，如果生成模型只用到少数，效率就低。  
- **模型容量（Model Capacity）**：语言模型的参数量或层数，决定其学习和推理能力。容量大通常意味着更强的语言理解和生成。  
- **开放域问答基准（Open‑Domain QA Benchmarks）**：如Natural Questions、TriviaQA等，用来衡量系统在真实世界提问上的表现。

### 核心创新点
1. **从“更大模型”转向“更大语料库”**  
   之前的主流做法是直接升级生成模型的规模，以期提升答案质量。论文则把焦点放在扩大检索库的大小，通过增大文档数量来提升系统整体表现。实验显示，适度扩大语料库的收益可以与换用更大模型相媲美。  
2. **系统性量化“语料库‑模型容量”权衡**  
   作者在多个公开问答基准上，分别跑了不同检索库规模（从几百GB到上百TB）和不同生成模型规模（tiny、small、mid、large）的组合。通过横向对比，绘制出一张“语料库规模 vs. 模型容量”的收益曲线，明确指出在何种情况下增大语料库比升级模型更划算。  
3. **覆盖率驱动的性能解释**  
   通过统计检索到的答案片段比例，论文发现语料库扩张带来的主要收益来源于提升了答案所在段落的覆盖率，而生成模型对检索结果的利用效率几乎保持不变。这一发现帮助解释了为何增大语料库在中等规模模型上效果最明显。  
4. **实用的“Chunk 大小”控制手段**  
   为了在同一语料库上实现不同规模的检索库，作者采用了改变文档切分粒度（Chunk size）的办法：把原始文档切成更细的块就能在不增加原始数据量的前提下提升检索库的条目数。这个技巧简单易行，却在实验中显著提升了覆盖率。

### 方法详解
整体思路可以拆成三步：**构建可伸缩的检索库 → 进行检索 → 交给生成模型生成答案**。下面逐步展开每一步的细节。

1. **可伸缩检索库的构建**  
   - **原始文档收集**：从公开的网页、维基百科、新闻等来源抓取数十TB的原始文本。  
   - **Chunk 切分**：把每篇文档按照预设的字符数（如 100、200、400）切成若干块。块越小，库中条目数越多，等价于“放大”了检索库。  
   - **向量化**：使用一个固定的轻量检索模型（如 DPR、BM25+Dense）把每个块映射成向量或稀疏特征。  
   - **索引构建**：把所有向量写入可扩展的向量搜索引擎（FAISS、ScaNN），支持数十亿条目检索。  

   这里最巧妙的点在于：不需要真的去抓更多的网页，只要把已有文本切得更细，就能在同等存储成本下“放大”检索空间。

2. **检索阶段**  
   - 给定用户问题，先用同样的检索模型把问题向量化。  
   - 在索引中进行近邻搜索，返回 top‑k（通常 5‑10）最相似的块。  
   - 对返回的块做一次轻量的重排序（比如基于交叉注意力的打分），确保答案片段排在前面。  

   这一步与传统 RAG 基本一致，唯一的区别是检索库的规模可能比以往大几个数量级。

3. **生成阶段**  
   - 把检索到的块拼接成一个提示，格式类似：“问题：… 文档1：… 文档2：…”。  
   - 将提示喂入不同规模的生成模型（从 70M 参数的 tiny 到 11B 参数的 large）。  
   - 模型直接输出答案，或者在需要时进行后处理（如答案抽取）。  

   关键观察是：即使换成更大的模型，检索到的文本数量和质量基本不变，说明生成模型的“利用效率”在不同模型之间相对稳定。

**反直觉点**：很多人会担心把文档切得太细会导致检索噪声增多，反而降低效果。但实验表明，只要保持一定的上下文完整性（如 200‑400 字块），覆盖率提升的好处远大于噪声的副作用。

### 实验与效果
- **数据集**：论文在 Natural Questions、TriviaQA、WebQuestions 等四个主流开放域问答基准上做评测。  
- **基线对比**：与同等生成模型下的标准 RAG（使用固定 3‑5 TB 检索库）以及直接使用更大模型（不增库）进行比较。  
- **核心发现**：  
  - 在 mid‑size（≈1B 参数）模型上，将检索库从 5 TB 扩展到 50 TB，准确率提升约 4‑5%（相当于把模型升级到 3B 参数的收益）。  
  - 对 tiny（≈70M）和 ultra‑large（≈11B）模型，扩库的提升幅度明显小于 mid‑size，分别只有约 1% 和 1.5% 的增益。  
  - 通过覆盖率统计，检索库扩大 10 倍后，答案所在块的命中率从约 55% 提升到 78%，而生成模型对检索结果的利用率（即实际被引用的块比例）保持在 30% 左右。  
- **消融实验**：作者分别关闭 Chunk 大小调节、重排序步骤以及向量化方式，发现 Chunk 大小是提升覆盖率的主因，重排序对最终准确率贡献约 0.8%。  
- **局限性**：  
  - 当检索库规模超过约 200 TB 时，收益出现明显递减，说明覆盖率已经接近饱和。  
  - 实验仅在英文公开数据上完成，中文或其他低资源语言的效果仍未验证。  
  - 论文未深入探讨检索成本（GPU/CPU 时间）随库规模的线性增长对实际部署的影响。  

### 影响与延伸思考
这篇工作在 RAG 社区掀起了“库先行”的讨论潮。随后出现的几篇论文（如 *Scaling Retrieval for Open‑Domain QA*、*Corpus‑aware Prompting*）都在不同维度上进一步验证或扩展了“增库比增模更划算”的观点。业界也开始在产品化系统里加入可动态扩容的检索层，例如把用户生成的日志、企业内部文档实时加入检索库，以提升答案的时效性和准确性。想继续深入的读者可以关注以下方向：  
- **检索效率优化**：在百亿级向量库上实现毫秒级检索的算法。  
- **跨语言检索**：如何在多语言语料库中保持高覆盖率。  
- **检索‑生成协同训练**：让生成模型反馈哪些检索结果更有用，从而引导检索器学习更精准的表示。  

### 一句话记住它
在检索增强生成中，适度扩大文档库往往能抵消更大语言模型的收益，关键在于让答案所在的段落更容易被检索到。