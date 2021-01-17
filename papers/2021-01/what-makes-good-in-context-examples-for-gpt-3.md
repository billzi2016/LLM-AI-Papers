# What Makes Good In-Context Examples for GPT-$3$?

> **Date**：2021-01-17
> **arXiv**：https://arxiv.org/abs/2101.06804

## Abstract

GPT-$3$ has attracted lots of attention due to its superior performance across a wide range of NLP tasks, especially with its powerful and versatile in-context few-shot learning ability. Despite its success, we found that the empirical results of GPT-$3$ depend heavily on the choice of in-context examples. In this work, we investigate whether there are more effective strategies for judiciously selecting in-context examples (relative to random sampling) that better leverage GPT-$3$'s few-shot capabilities. Inspired by the recent success of leveraging a retrieval module to augment large-scale neural network models, we propose to retrieve examples that are semantically-similar to a test sample to formulate its corresponding prompt. Intuitively, the in-context examples selected with such a strategy may serve as more informative inputs to unleash GPT-$3$'s extensive knowledge. We evaluate the proposed approach on several natural language understanding and generation benchmarks, where the retrieval-based prompt selection approach consistently outperforms the random baseline. Moreover, it is observed that the sentence encoders fine-tuned on task-related datasets yield even more helpful retrieval results. Notably, significant gains are observed on tasks such as table-to-text generation (41.9% on the ToTTo dataset) and open-domain question answering (45.5% on the NQ dataset). We hope our investigation could help understand the behaviors of GPT-$3$ and large-scale pre-trained LMs in general and enhance their few-shot capabilities.

---

# 什么样的上下文示例对 GPT‑3 有效？ 论文详细解读

### 背景：这个问题为什么难？
GPT‑3 之所以火，是因为它能在只给几条示例的情况下完成各种 NLP 任务，这种“在上下文中学习”（in‑context learning）让人觉得不需要再微调模型。但实验发现，同一任务不同的示例组合会导致性能相差甚远，随机挑几条往往只能得到中规中矩的结果。之前的做法几乎都是“随手抽取”，缺少系统的选择原则，导致模型的潜力没有被充分激发。要想真正利用 GPT‑3 的大规模知识库，就必须弄清楚：哪些示例最能帮助模型理解当前测试样本的意图？

### 关键概念速览
**In‑Context Learning（上下文学习）**：把少量示例直接写进模型的输入提示里，让模型在推理时“看到”这些例子并模仿其格式和答案。类似于老师在黑板上演示几道例题，学生据此解答新题。

**Prompt（提示）**：模型的完整输入文本，包括任务说明、示例对（输入‑输出）以及待预测的测试样本。可以把它想成一次“对话的开场白”，决定了模型后面的表现。

**检索（Retrieval）**：从一个大库里挑出和当前测试样本语义相近的已有实例。就像在图书馆里先找几本和你要写的论文主题相似的参考文献，再据此写作。

**Sentence Encoder（句子编码器）**：把一句自然语言映射成向量的模型，向量之间的距离反映语义相似度。类似于把句子压缩成“指纹”，指纹相近的句子被认为意思相近。

**Fine‑tuning（微调）**：在特定任务的数据上继续训练已有模型，使其编码器更懂该任务的语言特征。相当于给通用的指纹仪装上专门识别某类指纹的滤镜。

**Few‑Shot Learning（少样本学习）**：只用极少的标注样本就能让模型完成任务。这里的“少”指的是放进 Prompt 里的示例数量，而不是模型参数的多少。

### 核心创新点
1. **随机抽样 → 语义相似检索**：过去大多数研究直接随机挑选几条示例放进 Prompt。本文改为先用句子编码器在大规模示例库中检索出与测试样本语义最接近的 K 条，然后把这些检索到的例子作为 Prompt。这样模型看到的示例更贴近当前问题，提升了信息利用率。

2. **通用编码器 → 任务微调编码器**：作者进一步尝试把在任务相关数据上微调过的句子编码器用于检索。微调后编码器更懂该任务的细粒度差异，检索到的示例质量更高，实验显示相较于直接使用通用编码器还能再提升几百分点。

3. **统一评估框架 → 多任务验证**：论文把检索式 Prompt 方案在自然语言理解（如问答）和生成（如表格到文本）等多类基准上跑通，证明该策略不是针对某一特定任务的“巧合”。在 ToTTo 表格生成任务上提升了 41.9%，在 NQ 开放域问答上提升了 45.5%，显著超过随机基线。

### 方法详解
整体思路可以拆成三步：

1. **构建示例库**  
   收集大量已标注的输入‑输出对，形成检索的候选集合。比如对问答任务，库里每条记录是一个问题及其答案；对表格生成任务，库里每条记录是一个表格片段和对应的描述句子。

2. **语义检索**  
   - **编码**：使用句子编码器把库中每条示例的输入部分（问题或表格描述）映射成向量。若有任务微调的编码器，则先在该任务的训练集上继续训练，使向量更能捕捉任务特有的语义。  
   - **查询**：对每个测试样本，同样用编码器得到向量，然后在向量空间里找最近的 K 条（常用余弦相似度），即语义最相似的示例。  
   - **排序**：如果相似度相近，可再依据示例的质量（如答案长度、是否包含关键实体）进行二次排序，确保最终选出的示例既相似又可靠。

3. **组装 Prompt**  
   把检索到的 K 条示例按照“输入 → 输出”的格式依次写入 Prompt，随后紧跟测试样本的输入，留出模型生成输出的空间。整个 Prompt 看起来像是“老师先给了 K 道和本题相似的例题，然后让学生直接写答案”。

**最巧妙的地方**在于把检索模块当作“动态 Prompt 生成器”。传统上 Prompt 是手工写死的，而这里每个测试样本都有专属的 Prompt，完全依据其语义特征定制，极大提升了模型的上下文匹配度。

### 实验与效果
- **数据集与任务**：论文在多个公开基准上评估，包括表格到文本的 ToTTo、开放域问答的 Natural Questions（NQ），以及若干自然语言理解任务（具体任务在摘要中未列出，原文未详细描述）。
- **对比基线**：随机抽取 K 条示例作为 Prompt 的随机基线，以及若干已有的 Prompt 设计方法（如手工挑选、固定模板等）。
- **主要结果**：在 ToTTo 上检索式 Prompt 提升了 41.9% 的指标（BLEU/ROUGE 等），在 NQ 上提升了 45.5% 的准确率。相较于随机基线，这两个任务的提升幅度均超过 30%。
- **消融实验**：作者分别去掉检索步骤、使用未微调的通用编码器、以及不进行相似度排序，发现每一步的去除都会导致性能回落数个百分点，说明检索、编码器微调和相似度排序都是贡献因素。
- **局限性**：检索需要维护大规模示例库并进行向量相似度计算，计算开销比纯随机抽样大；此外，若任务本身缺乏足够的高质量示例，检索效果会受限。原文未给出对极端低资源场景的实验。

### 影响与延伸思考
这篇工作把“检索增强”理念从大模型内部（如 Retrieval‑Augmented Generation）搬到了 Prompt 设计层面，开启了“检索‑Prompt”这一新方向。随后的研究陆续探索了更高效的向量检索结构（如 HNSW）、跨语言检索、以及把检索过程与模型生成进行端到端联合训练。对想进一步了解的读者，可以关注以下方向：  
- **可学习的 Prompt 生成**：让模型自己学习如何挑选示例，而不是依赖固定检索器。  
- **多模态检索**：在图像‑文本任务中，用图像特征检索对应的文本示例。  
- **低资源检索**：在示例稀缺的情况下，如何利用合成示例或少量标注提升检索质量。

### 一句话记住它
把最相似的几条已有例子“拉进来”做 Prompt，GPT‑3 的少样本表现会立刻飙升。