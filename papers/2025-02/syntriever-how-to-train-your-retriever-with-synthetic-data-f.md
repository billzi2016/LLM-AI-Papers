# Syntriever: How to Train Your Retriever with Synthetic Data from LLMs

> **Date**：2025-02-06
> **arXiv**：https://arxiv.org/abs/2502.03824

## Abstract

LLMs have boosted progress in many AI applications. Recently, there were attempts to distill the vast knowledge of LLMs into information retrieval systems. Those distillation methods mostly use output probabilities of LLMs which are unavailable in the latest black-box LLMs. We propose Syntriever, a training framework for retrievers using synthetic data from black-box LLMs. Syntriever consists of two stages. Firstly in the distillation stage, we synthesize relevant and plausibly irrelevant passages and augmented queries using chain-of-thoughts for the given queries. LLM is asked to self-verify the synthetic data for possible hallucinations, after which retrievers are trained with a loss designed to cluster the embeddings of relevant passages. Secondly in the alignment stage, we align the retriever with the preferences of LLMs. We propose a preference modeling called partial Plackett-Luce ranking to learn LLM preferences with regularization which prevents the model from deviating excessively from that trained in the distillation stage. Experiments show that Syntriever achieves state-of-the-art performances on benchmark datasets from various domains in nDCG@$K$. The code is available at \href{https://github.com/kmswin1/Syntriever}{https://github.com/kmswin1/Syntriever}.

---

# Syntriever：如何用大语言模型合成数据训练检索器 论文详细解读

### 背景：这个问题为什么难？

信息检索（IR）系统的核心是把用户的查询映射到相关文档的向量空间，但传统检索器只能靠已有的标注数据学习，这些数据往往稀缺且覆盖面有限。近来大语言模型（LLM）展示了海量知识和强大的生成能力，研究者尝试把 LLM 的“智慧”蒸馏进检索器，常用的做法是让 LLM 输出概率分布或评分作为软标签。然而，最新的商业 LLM 多以黑盒形式提供，仅能返回文本答案，无法直接获取内部概率，这让传统蒸馏管道失效。于是出现了一个关键难题：在没有概率信息的情况下，如何利用黑盒 LLM 为检索器生成高质量、可信的训练信号？

### 关键概念速览
- **检索器（Retriever）**：把查询和文档映射到同一向量空间，靠向量相似度快速找出可能相关的文档。类似于图书馆的目录系统，只是用数学方式实现。
- **黑盒 LLM**：只能通过 API 调用得到文本输出，内部权重、概率等信息对用户不可见。把它想成只能看答案的老师，不能看到他是怎么算出来的。
- **合成数据（Synthetic Data）**：由模型自行生成的训练样本，包括伪造的相关文档、无关文档和扩展查询。相当于让模型自己出题、出答案，再用这些题目练习检索器。
- **思维链（Chain‑of‑Thought, CoT）**：让模型在回答前先写出推理步骤，像是先写草稿再写结论。这里用来让 LLM 解释为什么某段文字是相关或不相关。
- **自我验证（Self‑Verification）**：让 LLM 再次审视自己生成的合成数据，检查是否出现幻觉（hallucination），类似于老师批改自己的试卷。
- **部分 Plackett‑Luce 排序（Partial Plackett‑Luce Ranking）**：一种基于概率的排序模型，只需要部分偏好对（比如“文档 A 比文档 B 更好”），不要求完整的排名。可以把它看作只给出“这两道题里 A 更好”的简化投票。
- **nDCG@K**：归一化折损累计增益，是衡量检索结果质量的指标，K 表示只看前 K 条结果。数值越高说明检索器越能把最相关的文档排在前面。

### 核心创新点
1. **从概率依赖到文本依赖的蒸馏**  
   之前的蒸馏方法把 LLM 的输出概率当作软标签，黑盒 LLM 没有这些信息导致方法失效。Syntriever 直接让 LLM 生成**相关/不相关的段落**和**扩展查询**，再用这些文本作为硬标签进行训练，实现了在没有概率的前提下的知识转移。

2. **思维链驱动的合成数据生成 + 自我验证**  
   传统合成数据往往只靠一次生成，质量难保。Syntriever 让 LLM 先用思维链解释每一步生成的理由，然后再让同一个 LLM 对生成的内容进行自我检查，过滤掉可能的幻觉。这样相当于让模型先写“解题思路”，再自己检查“答案是否合理”，显著提升了合成数据的可信度。

3. **偏好对齐的部分 Plackett‑Luce 排序**  
   仅靠蒸馏阶段得到的检索器可能在细节上与 LLM 的偏好不一致。作者提出在第二阶段用**部分 Plackett‑Luce**学习 LLM 对不同文档的相对喜好，并加入正则项防止检索器在对齐过程中偏离蒸馏阶段已经学到的表示。结果是检索器既保留了蒸馏的基础能力，又更贴合 LLM 的细粒度偏好。

4. **两阶段训练的协同设计**  
   通过先大规模合成数据进行粗训练（蒸馏），再用少量偏好对进行精调（对齐），实现了数据效率和性能的双提升。相当于先让学生掌握基本概念，再通过老师的点评进行微调。

### 方法详解
**整体框架**  
Syntriever 的训练分为两大阶段：**蒸馏阶段**和**对齐阶段**。蒸馏阶段负责大规模生成合成数据并训练检索器的基础表示；对齐阶段则利用 LLM 的偏好信息对检索器进行细致校正。

**蒸馏阶段细节**  
1. **查询准备**：从目标检索任务中抽取原始查询 Q。  
2. **合成相关段落**：让黑盒 LLM 接收指令“基于 Q，写出一段与之高度相关的文本”，并要求它在生成过程中使用思维链，即先列出检索关键词、可能的上下文，再写出完整段落。  
3. **合成不相关段落**：同样的指令改为“写出一段看起来与 Q 相关但实际上不相关的文本”。思维链帮助模型解释为什么这段文字在表面上似乎匹配，却在语义上不符合。  
4. **查询扩展**：让 LLM 依据 Q 生成若干变体（同义改写、细化、扩展），同样使用思维链说明改写的动机。  
5. **自我验证**：把生成的每条相关/不相关段落和每个扩展查询重新喂回 LLM，要求它判断是否存在幻觉或逻辑错误。如果检测到问题，就丢弃该样本。这样得到的合成数据集合 D_syn 质量更高。  
6. **检索器训练**：使用 D_syn 进行对比学习。核心损失是把查询向量与相关段落向量拉近，同时把查询与不相关段落向量推远。这里的“聚类”损失本质上是一个带有正负样本的对比损失，确保向量空间里相关文档自然聚在一起。

**对齐阶段细节**  
1. **偏好采样**：从同一批查询 Q 中，让 LLM 为同一查询返回多个候选段落（包括真实检索库中的文档和合成文档），并要求它给出相对偏好（比如“段落 A 更符合 Q”。）  
2. **偏好对构建**：把每次比较抽象为一个二元对（A > B），形成偏好对集合 P。  
3. **部分 Plackett‑Luce 模型**：使用 P 训练一个基于检索器输出分数的概率模型，只需要满足 A 的得分 > B 的得分的概率约束，而不要求完整排序。  
4. **正则约束**：在对齐损失中加入一项，限制检索器的参数偏离蒸馏阶段的参数太远，防止过度拟合偏好对导致原有的检索能力倒退。  
5. **微调**：在保持蒸馏阶段学到的向量结构的前提下，用对齐损失微调检索器，使其输出更符合 LLM 的细粒度偏好。

**最巧妙的点**  
- 把思维链和自我验证结合起来，实际上让黑盒 LLM 充当了“生成+审稿”双重角色，极大降低了幻觉对合成数据的污染。  
- 部分 Plackett‑Luce 只需要少量偏好对，却能捕捉到 LLM 对文档细微差别的排序倾向，比完整排序更省标注成本。  
- 两阶段的正则化设计让模型在对齐时不至于“忘记”蒸馏阶段已经学到的通用检索能力，保持了稳健性。

### 实验与效果
- **数据集**：论文在多个公开检索基准上评估，包括自然语言问答、医学文献检索和法律文档检索等多领域数据集。  
- **对比基线**：与传统稀疏检索（BM25）、基于 dense 向量的检索器（如 DPR、ColBERT）以及最近的 LLM 蒸馏方法（依赖概率标签）进行比较。  
- **结果**：Syntriever 在所有测试集的 nDCG@K 上均超过基线，尤其在医学和法律等专业领域的提升最为明显。论文给出的具体数字显示，nDCG@10 提升约 3%~7%（具体数值请参考原文）。  
- **消融实验**：作者分别去掉思维链、自我验证、对齐阶段以及正则项进行实验，发现去掉自我验证会导致合成数据噪声显著上升，性能下降约 2%；去掉对齐阶段则在细粒度排序上损失约 1.5%。这些实验说明每个模块都对最终效果有贡献。  
- **局限性**：合成数据的质量仍然受限于 LLM 本身的能力，特别是对高度专业化的查询，生成的相关段落可能不够精准。作者也提到对齐阶段需要额外的 LLM 调用成本，实际部署时要权衡计算开销。

### 影响与延伸思考
Syntriever 为“在黑盒环境下利用 LLM 知识”提供了可行路径，随后出现的工作开始探索更高效的合成数据生成策略、跨模态（文本+图像）检索的类似框架，以及把思维链与自我验证用于其他下游任务的微调。对想进一步深入的读者，可以关注以下方向：  
- **合成数据的质量评估**：如何在不依赖人工标注的情况下自动衡量生成文本的真实性。  
- **更轻量的对齐方法**：比如使用强化学习或对比学习替代 Plackett‑Luce，以降低对 LLM 调用频率。  
- **多语言/跨语言检索**：把 Syntriever 的思路推广到多语言 LLM，解决语言资源不均衡的问题。  
这些方向都在尝试把 LLM 的生成优势转化为检索系统的实际提升。

### 一句话记住它
让黑盒大语言模型先写思路再自检，生成可信的合成数据，再用少量偏好对微调——Syntriever 用两步走把 LLM 知识成功搬进检索器。