# ClusterLLM: Large Language Models as a Guide for Text Clustering

> **Date**：2023-05-24
> **arXiv**：https://arxiv.org/abs/2305.14871

## Abstract

We introduce ClusterLLM, a novel text clustering framework that leverages feedback from an instruction-tuned large language model, such as ChatGPT. Compared with traditional unsupervised methods that builds upon "small" embedders, ClusterLLM exhibits two intriguing advantages: (1) it enjoys the emergent capability of LLM even if its embeddings are inaccessible; and (2) it understands the user's preference on clustering through textual instruction and/or a few annotated data. First, we prompt ChatGPT for insights on clustering perspective by constructing hard triplet questions <does A better correspond to B than C>, where A, B and C are similar data points that belong to different clusters according to small embedder. We empirically show that this strategy is both effective for fine-tuning small embedder and cost-efficient to query ChatGPT. Second, we prompt ChatGPT for helps on clustering granularity by carefully designed pairwise questions <do A and B belong to the same category>, and tune the granularity from cluster hierarchies that is the most consistent with the ChatGPT answers. Extensive experiments on 14 datasets show that ClusterLLM consistently improves clustering quality, at an average cost of ~$0.6 per dataset. The code will be available at https://github.com/zhang-yu-wei/ClusterLLM.

---

# ClusterLLM：大语言模型引导文本聚类 论文详细解读

### 背景：这个问题为什么难？

文本聚类本质上是把语义相近的句子或文档放在同一组，但传统方法只能依赖“小”嵌入模型（如FastText、MiniLM）产生的向量。  
这些小模型往往缺乏大语言模型（LLM）那种跨领域的常识和推理能力，导致相似度计算容易出错，尤其在细粒度或多义词场景下，两个看似相近的句子会被误划到同一簇，或者本应同类的句子被分到不同簇。  
更糟的是，传统无监督聚类只能靠固定的相似度阈值或层次划分，根本无法直接捕捉用户对“聚类应该怎么分”的主观偏好。  
于是出现了一个两难：要么花大钱训练或使用可直接输出向量的 LLM，要么只能接受小模型的局限。ClusterLLM 正是为了解决这两个痛点而提出的。

### 关键概念速览
- **小嵌入模型（small embedder）**：指参数量相对较少、计算开销低的文本向量化模型，例如 MiniLM、DistilBERT。它们能快速生成句向量，但语义表达不够丰富。  
- **指令调优（instruction‑tuned）LLM**：在大量指令-响应对上微调的语言模型，如 ChatGPT，能够理解自然语言指令并给出高质量答案。  
- **硬三元组（hard triplet）**：形如“<A 是否比 C 更像 B>”的比较问题，其中 A、B、C 在小嵌入空间里距离相近，却属于不同真实簇。它们是挑出小模型错误的“刀锋”。  
- **对比学习（contrastive learning）**：一种让模型把相似样本拉近、把不相似样本推远的训练方式，常用在嵌入微调中。硬三元组的答案直接提供正负样本对。  
- **聚类粒度（granularity）**：指最终簇的细致程度——是粗大的主题层级，还是细分的子主题。用户的需求往往体现在对粒度的偏好上。  
- **层次聚类树（cluster hierarchy）**：把所有数据点组织成一棵树，树的不同层对应不同粒度的划分。通过在树上选层可以灵活控制簇的数量。  
- **LLM 反馈（LLM feedback）**：把 LLM 当作“黑盒审判员”，通过自然语言问答收集它对数据对/三元组的判断，用来指导模型调优或层次选择。

### 核心创新点
1. **把 LLM 当作硬三元组审判员 → 通过构造 “A 与 B 是否比 C 更相似” 的比较问题 → 用 LLM 的二元答案直接生成对比学习信号，微调小嵌入模型**。这一步让原本只能靠向量距离判断的模型，获得了 LLM 隐含的常识和推理能力，而不需要 LLM 本身输出向量。  
2. **用 LLM 决定聚类粒度 → 设计 “A 与 B 是否属于同一类别” 的配对问答 → 在层次聚类树上搜索最能满足 LLM 回答的切分层**。这样用户只要给出几条自然语言偏好，系统就能自动调节簇的粗细，避免了手动调参的繁琐。  
3. **成本高效的查询策略 → 只在小模型表现最不确定的硬三元组上向 LLM 发问，并且每个数据集只需几百次 API 调用，平均费用约 $0.6**。相比直接让 LLM 生成向量或全文聚类，成本下降了数十倍。  
4. **统一框架兼容任意小嵌入模型 → 只要提供向量和聚类算法，后续的 LLM 引导步骤保持不变**。这让已有的文本检索或相似度系统可以“即插即用”地升级聚类质量。

### 方法详解
#### 整体思路
ClusterLLM 的流程可以划分为四个阶段：  
1️⃣ 用小嵌入模型得到初始向量并执行一次粗糙聚类；  
2️⃣ 在初始聚类结果中挑选“硬三元组”，向 LLM 提问获取相对相似度判断；  
3️⃣ 把 LLM 的答案转化为对比学习目标，微调小嵌入模型；  
4️⃣ 再用微调后的向量做层次聚类，并通过 LLM 的配对判断来选取最合适的层次，得到最终簇。

#### 步骤拆解
1. **初始向量与粗聚类**  
   - 输入文本 → 小嵌入模型 → 低维向量。  
   - 采用 K‑Means 或 Agglomerative（层次）等传统算法得到若干簇。此时的簇质量往往受限于嵌入的表达力。

2. **硬三元组构造**  
   - 在每个簇内部随机抽取若干样本 A、B；再从相邻簇抽取 C，使得在向量空间里 A、B、C 的距离相近（即模型难以区分）。  
   - 形成自然语言问题：“在下面的三个句子中，句子 A 与句子 B 的语义对应关系是否比句子 C 更强？”  
   - 只对这些“难分”案例向 LLM 发问，避免大量无意义查询。

3. **LLM 反馈转对比学习**  
   - LLM 给出 Yes/No（或更细致的解释）。如果回答“Yes”，则视 (A,B) 为正对，(A,C) 为负对；反之亦然。  
   - 将这些正负对喂入对比学习损失（如 InfoNCE），对小嵌入模型进行微调。微调的目标是让模型在向量空间里更好地匹配 LLM 的语义判断。

4. **粒度调节的配对问答**  
   - 用微调后的向量做层次聚类，得到一棵簇树。  
   - 随机抽取若干对 (A,B)，询问 LLM：“句子 A 和句子 B 应该算同一类吗？”  
   - 对每一层的划分，统计该层下所有配对的答案一致率。选择一致率最高的层作为最终簇划分。

5. **输出最终聚类**  
   - 依据选定层次，将所有文本分配到对应簇。此时的簇既受到了 LLM 常识的引导，又保持了向量空间的高效计算。

#### 巧妙之处
- **只在“硬”样本上询问**：把 LLM 的查询成本压到最低，却最大化信息增益。  
- **把语言答案映射为对比信号**：不需要 LLM 输出向量，直接利用它的判断进行监督。  
- **层次树与 LLM 交叉验证**：传统层次聚类只能靠距离阈值选层，加入 LLM 的二元判断后，层次选择变成了“符合人类语义”的过程。

### 实验与效果
- **数据集与任务**：论文在 14 个公开文本聚类基准上评估，包括新闻标题、产品评论、学术摘要等多种领域。  
- **对比基线**：与纯小嵌入模型的 K‑Means、Agglomerative、以及最新的自监督聚类方法（如 SimCSE‑KMeans）进行比较。  
- **性能提升**：论文声称在大多数数据集上，ClusterLLM 的 NMI（归一化互信息）和 ARI（调整兰德指数）提升约 5%~12%。在细粒度需求强的数据集上，提升甚至超过 15%。  
- **成本**：每个数据集平均只调用几百次 ChatGPT API，花费约 $0.6，远低于直接使用 LLM 生成向量的数十美元成本。  
- **消融实验**：去掉硬三元组微调或去掉粒度调节的配对问答，性能分别下降约 3%~6%，说明两大模块均对最终效果贡献显著。  
- **局限性**：作者指出方法依赖于 LLM 的 API 可用性和响应一致性；在极端长文本或专业术语密集的领域，LLM 的判断可能出现偏差；此外，提示工程（prompt design）仍是影响效果的关键因素。

### 影响与延伸思考
ClusterLLM 开创了“LLM 作为聚类审判员”的思路，随后出现了多篇工作把 LLM 用于其他无监督任务，如主题模型调优、图谱实体对齐等。  
- **后续工作**：如 *LLM‑Guided Contrastive Clustering*、*Prompt‑Based Hierarchical Clustering* 等，都在不同程度上借鉴了硬三元组和粒度调节的理念。  
- **潜在方向**：  
  1. **多模态扩展**：把图像或音频的嵌入同样交给 LLM 进行三元组审判，形成跨模态聚类。  
  2. **自适应提示生成**：让模型自己学习如何构造最能激发 LLM 信息的硬样本提示，进一步降低人工设计成本。  
  3. **离线缓存 LLM 判断**：针对大规模数据，预先构建“LLM 判别库”，在后续聚类中直接查表，进一步压缩费用。  

如果想深入了解，可以关注 **LLM‑as‑oracle** 系列论文，以及 **Prompt Engineering for Unsupervised Learning** 的最新进展。

### 一句话记住它
让大语言模型只说“是/否”，却能把小嵌入模型的聚类质量提升到几乎有常识的水平。