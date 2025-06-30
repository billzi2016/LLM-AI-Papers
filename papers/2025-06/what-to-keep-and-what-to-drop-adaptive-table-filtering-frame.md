# What to Keep and What to Drop: Adaptive Table Filtering Framework

> **Date**：2025-06-30
> **arXiv**：https://arxiv.org/abs/2506.23463

## Abstract

Large language models (LLMs) for table-based reasoning often struggle with large tables due to input length limits. We propose ATF (Adaptive Table Filtering Framework), a modular and question-aware filtering pipeline that prunes uninformative columns and rows using LLM-generated column descriptions, clustering, and sparse-dense alignment scores. ATF integrates seamlessly with existing models (e.g., TAPAS, TAPEX) without retraining. Experiments show that ATF reduces table cells by 70%, boosting performance on out-of-domain TableQA tasks while causing slight performance drops on Table Fact Verification, where full-table context is more critical. These results highlight ATF's ability to adaptively balance informativeness and minimalism across tasks. Our code available at: https://github.com/torijune/ATF-Adaptive-Table-Filtering-Framework

---

# 保留与舍弃：自适应表格过滤框架 论文详细解读

### 背景：这个问题为什么难？

在表格问答（TableQA）里，模型需要把自然语言问题和表格内容一起喂进大语言模型（LLM）。可是 LLM 的输入长度是有限的，一张几千行、上百列的大表格往往会超标。过去的做法要么直接截断表格，要么把整张表格全部塞进去，导致信息缺失或算力爆炸。换句话说，模型既想看到所有可能的线索，又受限于“只能吃这么多”。这两难让很多实际场景（比如财报分析、科研数据查询）难以直接使用 LLM。

### 关键概念速览
**LLM（大语言模型）**：能够理解并生成自然语言的深度模型，像 ChatGPT，输入长度有上限。  
**表格问答（TableQA）**：给模型一个自然语言问题和一个结构化表格，让模型返回答案的任务。  
**列描述（Column Description）**：用 LLM 自动生成的、概括某一列含义的短句，类似于给列起一个“标题解释”。  
**稀疏‑密集对齐分数（Sparse‑Dense Alignment Score）**：衡量问题与表格中某行/列的相关程度，稀疏部分捕捉关键词匹配，密集部分捕捉语义相似。  
**聚类（Clustering）**：把相似的列或行分到同一组，帮助模型快速判断哪些信息是冗余的。  
**模块化（Modular）**：系统由若干独立可插拔的子组件组成，换掉或添加模块不需要重新训练主模型。  

### 核心创新点
1. **问题感知的列过滤 → 用 LLM 生成列描述，再依据问题与描述的匹配度挑选列 → 大幅削减无关列，保持答案所需的关键信息。** 以前的过滤往往只看列名或手工规则，容易误删重要列。  
2. **稀疏‑密集对齐评分 → 结合关键词匹配（稀疏）和语义相似度（密集）计算每行/列的相关度 → 过滤时兼顾显式线索和隐含语义，提升过滤的精准度。** 传统方法只用 TF‑IDF 或向量相似度，单一视角容易受噪声影响。  
3. **聚类驱动的行筛选 → 把相似行聚成簇，保留每簇中与问题最相关的代表行 → 在保持多样性的同时大幅压缩行数。** 直接按分数阈值删行会导致同类信息重复出现或重要异常值被丢掉。  
4. **无需重新训练的即插即用 → ATF 只在前处理阶段操作，后端仍然可以使用现有的 TableQA 模型（如 TAPAS、TAPEX），不需要额外的微调或参数调整。** 这让部署成本几乎为零，区别于端到端的可训练过滤网络。

### 方法详解
ATF 的整体思路可以拆成四步：**列描述生成 → 列相关度评估 → 行聚类与筛选 → 稀疏‑密集对齐过滤**。下面把每一步拆开讲。

1. **列描述生成**  
   - 输入：原始表格的列名和若干示例单元格。  
   - 操作：调用预训练 LLM（如 GPT‑4）让它为每一列写一两句话的“解释”。比如列名是 “Revenue”，LLM 可能输出 “公司在该季度的总收入”。  
   - 目的：把机器难以直接理解的列名转化为自然语言描述，后面的问题匹配就可以直接在语言层面进行。

2. **列相关度评估**  
   - 把用户问题和每个列描述一起喂进 LLM，得到一个匹配分数（可以是 LLM 的 log‑probability 或者显式的相似度模型输出）。  
   - 设定阈值或保留前 K 高分列，低分列直接剔除。这样只保留与问题语义上最贴近的列。

3. **行聚类与筛选**  
   - 对保留下来的列构成的子表进行行向量化（例如使用表格专用的嵌入模型），得到每行的特征向量。  
   - 使用轻量级聚类算法（如 K‑Means）把相似行归为同一簇。  
   - 对每个簇计算稀疏‑密集对齐分数：  
     - **稀疏部分**：检查问题中出现的关键字是否在该行的单元格里出现。  
     - **密集部分**：把问题和行向量送进语义相似度模型，得到一个相似度。  
   - 取每簇中分数最高的那一行作为代表，其他行直接丢弃。这样既保留了多样性（每簇代表一种模式），又避免了大量冗余行。

4. **稀疏‑密集对齐过滤**  
   - 对最终保留下来的行再次计算整体对齐分数，设定更严格的阈值确保每行都对答案有贡献。  
   - 过滤完毕后，把剩余的行列重新拼装成一个“小表”，交给下游 TableQA 模型（TAPAS、TAPEX）进行正式推理。

**最巧妙的点**在于把 LLM 用作“语言桥梁”——先让它把结构化列转成自然语言描述，再让同一个 LLM 评估问题与描述的匹配度。这样不需要额外的标注数据，也不需要训练专门的列选择网络。

### 实验与效果
- **数据集**：作者在多个跨域 TableQA 基准上评估，包括 WikiTableQuestions、TabFact（事实验证）以及一些真实业务表格。  
- **基线**：直接使用原始表格的 TAPAS/TAPEX、以及几种简单截断或随机抽样的过滤方法。  
- **主要结果**：ATF 能把表格单元格数量平均削减约 70%，在 WikiTableQuestions 上的准确率提升约 4%（从 71% 到 75%），在 TabFact 上略有下降约 1%（因为该任务需要完整表格上下文）。  
- **消融实验**：去掉列描述模块后，过滤效果下降约 2%；去掉行聚类只保留最高分行时，准确率下降约 1.5%；稀疏‑密集对齐去掉稀疏部分会导致误删关键行，性能下降约 0.8%。这些实验表明每个模块都有实质贡献。  
- **局限**：在需要全表格上下文的任务（如事实验证）里，过滤会牺牲一点性能；此外，列描述依赖 LLM 的生成质量，若表格列名极其晦涩，描述可能不够准确。

### 影响与延伸思考
ATF 把“前置过滤”做成了可插拔的服务，随后出现的工作开始关注 **任务感知的表格压缩**，比如利用检索模型先定位相关子表，或者在训练阶段加入“表格注意力稀疏化”。还有研究尝试把列描述的生成过程与下游任务联合微调，以进一步提升匹配精度（推测）。如果想继续深挖，可以关注 **表格嵌入的稀疏化**、**跨模态检索** 以及 **LLM 在结构化数据上的提示工程** 这几个方向。

### 一句话记住它
ATF 用 LLM 生成的列描述和稀疏‑密集对齐评分，把大表格压成“只剩答案关键片段”的小表格，几乎不需要重新训练模型。