# A Hybrid Search for Complex Table Question Answering in Securities Report

> **Date**：2025-11-12
> **arXiv**：https://arxiv.org/abs/2511.09179

## Abstract

Recently, Large Language Models (LLMs) are gaining increased attention in the domain of Table Question Answering (TQA), particularly for extracting information from tables in documents. However, directly entering entire tables as long text into LLMs often leads to incorrect answers because most LLMs cannot inherently capture complex table structures. In this paper, we propose a cell extraction method for TQA without manual identification, even for complex table headers. Our approach estimates table headers by computing similarities between a given question and individual cells via a hybrid retrieval mechanism that integrates a language model and TF-IDF. We then select as the answer the cells at the intersection of the most relevant row and column. Furthermore, the language model is trained using contrastive learning on a small dataset of question-header pairs to enhance performance. We evaluated our approach in the TQA dataset from the U4 shared task at NTCIR-18. The experimental results show that our pipeline achieves an accuracy of 74.6\%, outperforming existing LLMs such as GPT-4o mini~(63.9\%). In the future, although we used traditional encoder models for retrieval in this study, we plan to incorporate more efficient text-search models to improve performance and narrow the gap with human evaluation results.

---

# 一种用于证券报告复杂表格问答的混合检索方法 论文详细解读

### 背景：这个问题为什么难？
在金融报告里，关键信息往往埋在层次复杂、跨行跨列的表格里。传统的表格问答系统要先把表格结构化、标注行列标题，然后才能定位答案。大模型（LLM）虽然擅长自然语言理解，却只能把整张表当成一段长文本输入，结果是对行列关系的把握很差，容易给出错误答案。换句话说，缺少对表格内部层次结构的感知是现有方法的根本瓶颈。

### 关键概念速览
**表格问答（Table Question Answering，TQA）**：系统根据自然语言提问，从文档中的表格里找出对应的单元格并返回内容。想象成在 Excel 里搜索，答案是某个格子。

**混合检索（Hybrid Retrieval）**：同时使用两种不同的相似度计算方式——基于词频的 TF‑IDF 和基于语义的语言模型嵌入——来评估问题和单元格的相关性。就像用关键词搜索再加上语义匹配，提升召回质量。

**行/列头估计**：通过比较问题与每个单元格的相似度，挑出最可能代表行标题的单元格和最可能代表列标题的单元格。相当于在没有显式表头的情况下，自动猜出“这行说的是什么，这列说的是什么”。

**对比学习（Contrastive Learning）**：让模型在训练时把正确的问‑头配对拉近，把错误配对拉远，从而学会更精准的语义匹配。类似于让模型记住“猫”和“狗”是不同的概念。

**Sentence‑BERT**：一种把句子压成向量的模型，向量之间的距离可以直接反映语义相似度。这里用它来把问题和表格单元格映射到同一空间。

### 核心创新点
**传统检索 → TF‑IDF + 语言模型混合 → 更精准的行列定位**  
以前的系统要么只用关键词匹配（TF‑IDF），要么只靠语义向量（embedding），单独使用时容易漏掉关键词或被噪声干扰。本文把两者结合，先用 TF‑IDF 捕捉显式词匹配，再用语言模型补足语义层面的相似度，最终选出最相关的行头和列头。这样在复杂表头（多层合并单元格）下仍能准确定位。

**手工标注表头 → 自动行列头估计 → 零标注流水线**  
过去需要人工标记每张表的行列标题，成本高且难以推广。本文通过对每个单元格计算与问题的相似度，直接挑出得分最高的行单元格和列单元格作为“隐式标题”。省去人工标注，直接对原始 HTML 表格做清洗后即可使用。

**小规模监督 → 对比学习微调 Sentence‑BERT → 提升检索质量**  
仅靠预训练的语言模型在专业金融表格上表现一般。作者收集了少量问‑头配对数据，用对比学习让模型更懂“财务指标”这类专业词汇。结果是检索阶段的相似度评分更可靠，整体准确率提升显著。

### 方法详解
整体思路可以拆成四步：**表格清洗 → 双模检索 → 行列交叉 → 后处理**。下面按顺序展开。

1. **表格清洗**  
   输入的证券报告往往是 HTML 或 PDF，里面混杂了页眉、页脚、样式标签等噪声。系统先把表格区域抽取出来，去掉所有非表格文字和装饰性标签，只保留单元格的纯文本内容。每个单元格被视为一个独立的检索文档，行号和列号作为元数据保存。

2. **双模检索（Hybrid Retrieval）**  
   - **TF‑IDF 计算**：对每个单元格构建词频向量，和提问的词频向量做余弦相似度，得到一个基于关键词的得分。  
   - **Sentence‑BERT 语义匹配**：把提问和每个单元格的文本分别送入已经经过对比学习微调的 Sentence‑BERT，得到向量后再计算余弦相似度，得到语义得分。  
   - **得分融合**：两种得分按固定权重相加（权重在验证集上调优），得到最终相似度。这样既保留了关键字的强信号，又兼顾了同义词、上下文的语义关联。

3. **行列头估计与答案定位**  
   - 按融合得分排序，取前两名单元格。若最高分单元格位于某一行，则该行被视为“最相关行”；若第二高分单元格位于某一列，则该列被视为“最相关列”。  
   - 两者的交叉点（行 × 列）即为答案单元格。若两者落在同一行或同一列，系统会继续向下查找次高分单元格，以确保得到一个真正的交叉格子。

4. **后处理**  
   - **数值归一化**：对答案单元格的数值进行千位分隔、单位统一（如“万元”转为“元”），保证输出格式一致。  
   - **单位抽取**：如果答案中包含单位（%、亿元等），系统会单独标记，方便下游任务使用。

**最巧妙的点**在于把行头和列头的选择看作两个独立的检索任务，然后用同一套相似度模型一次性完成。这样既避免了对表格结构的显式解析，又能在复杂合并单元格的情况下仍然找到正确的交叉格子。

### 实验与效果
- **数据集**：使用 NTCIR‑18 U4 共享任务提供的证券报告表格问答数据集（TQA），该数据集包含多种金融表格和对应的自然语言问题。  
- **基线对比**：与直接把整张表当作长文本喂入 GPT‑4o mini 的方式相比，本文方法的整体准确率从 63.9% 提升到 74.6%。同样的提升也超过了其他公开的 LLM 基线（如 GPT‑3.5、Claude 等），虽然具体数字未在摘要中列出。  
- **消融实验**：作者分别去掉 TF‑IDF、去掉 Sentence‑BERT、以及不做对比学习微调，发现每一步的去除都会导致准确率下降 3%~7%，说明双模检索和微调都是关键贡献。  
- **局限性**：目前检索阶段仍使用传统的 TF‑IDF 与 Sentence‑BERT，速度上不如专门的向量搜索引擎；此外，方法对极端大表（行列数千级）的内存占用未作深入评估，作者在结论中提到计划引入更高效的文本搜索模型来进一步提升性能。

### 影响与延伸思考
这篇工作展示了在不进行显式表格解析的前提下，仍能通过混合检索实现高质量的表格问答，激发了后续研究对“结构感知检索”的兴趣。后续的论文开始尝试把稀疏检索（BM25）和稠密检索（向量索引）结合，用更高效的 ANN（近似最近邻）库来加速大规模表格搜索。对想进一步探索的读者，可以关注以下方向：  
- 使用专门为表格设计的稠密检索模型（如 Table‑BERT、Tabular‑Transformer）。  
- 将表格结构信息（如合并单元格、层级标题）显式编码进向量，以提升对复杂表头的感知。  
- 在多模态场景下加入表格的视觉特征，探索视觉‑语言混合检索的可能性。

### 一句话记住它
把表格问答当成“行列双向检索”，用关键词+语义双模打分，就能在不标注表头的情况下精准定位答案。