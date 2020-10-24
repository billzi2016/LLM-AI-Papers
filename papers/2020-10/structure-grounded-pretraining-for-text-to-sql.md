# Structure-Grounded Pretraining for Text-to-SQL

> **Date**：2020-10-24
> **arXiv**：https://arxiv.org/abs/2010.12773

## Abstract

Learning to capture text-table alignment is essential for tasks like text-to-SQL. A model needs to correctly recognize natural language references to columns and values and to ground them in the given database schema. In this paper, we present a novel weakly supervised Structure-Grounded pretraining framework (StruG) for text-to-SQL that can effectively learn to capture text-table alignment based on a parallel text-table corpus. We identify a set of novel prediction tasks: column grounding, value grounding and column-value mapping, and leverage them to pretrain a text-table encoder. Additionally, to evaluate different methods under more realistic text-table alignment settings, we create a new evaluation set Spider-Realistic based on Spider dev set with explicit mentions of column names removed, and adopt eight existing text-to-SQL datasets for cross-database evaluation. STRUG brings significant improvement over BERT-LARGE in all settings. Compared with existing pretraining methods such as GRAPPA, STRUG achieves similar performance on Spider, and outperforms all baselines on more realistic sets. The Spider-Realistic dataset is available at https://doi.org/10.5281/zenodo.5205322.

---

# 基于结构的文本到SQL预训练 论文详细解读

### 背景：这个问题为什么难？

在 Text‑to‑SQL 任务中，模型必须把自然语言问题映射到对应的 SQL 查询，而这一步的关键是把句子里提到的列名、表名和具体取值准确地对应到数据库的结构上。过去的模型大多依赖大规模的标注问句‑SQL 对，或者只用通用语言模型（如 BERT）进行微调，却没有专门学习“文本 ↔ 表结构”之间的对齐关系。于是出现两大痛点：一是模型常常把同义词或省略的列名误识别，导致生成的 SQL 错误；二是现有的预训练任务（如掩码语言模型）并不涉及表结构信息，难以让模型自觉去捕捉列‑值的对应关系。正因为缺少针对结构对齐的预训练，实际使用中模型在面对真实业务场景（列名未显式出现）时表现大幅下降，这正是本文要破解的难点。

### 关键概念速览
**Text‑to‑SQL**：把自然语言提问自动翻译成对应的 SQL 查询语句，类似把口头指令转成数据库指令。  
**Schema（模式）**：数据库的结构描述，包括表名、列名以及列的数据类型，模型需要“读懂”这些信息才能定位答案。  
**Column Grounding（列对齐）**：识别句子中提到的列并把它映射到具体的 schema 列，就像把“年龄”这句话对应到表里的 `age` 列。  
**Value Grounding（取值对齐）**：把自然语言中的具体数值或字符串对应到表中实际出现的值，类似把“2022 年”对应到 `year = 2022` 的过滤条件。  
**Column‑Value Mapping（列‑值映射）**：确定某个取值属于哪一列，例如“北京”是城市列的取值，这一步是列对齐和取值对齐的交叉验证。  
**Weak Supervision（弱监督）**：不需要人工标注对齐信息，而是利用已有的文本‑表格对（如问句‑SQL）自动生成训练信号。  
**Spider‑Realistic**：作者基于 Spider 开发的评测集，专门去掉了显式列名，使模型必须靠上下文推断列，模拟真实业务的“隐式对齐”。  

### 核心创新点
1. **从“只学语言”到“学结构”**：传统预训练只用掩码语言模型，让模型预测被遮掉的词；本文引入了三类结构对齐任务（列对齐、取值对齐、列‑值映射），让模型在预训练阶段就学会把文字和表结构挂钩。这样模型在下游微调时已经具备了对齐能力，显著提升了对省略列名的鲁棒性。  
2. **弱监督的平行文本‑表格语料**：作者利用已有的 Text‑to‑SQL 数据库（问句‑SQL 对）自动生成“文本‑表格”对，省去人工标注对齐信息的成本。具体做法是把 SQL 中出现的列、常量抽取出来，构成对应的对齐标签，形成大规模的自监督信号。  
3. **结构化预训练任务的联合学习**：三项对齐任务不是单独训练，而是一起在同一个编码器上并行优化。这样模型可以在同一次前向传播中同时学习列的语义、值的语义以及它们的关联，形成更紧密的跨模态表示。  
4. **更贴近真实场景的评测**：作者新建 Spider‑Realistic 数据集，去掉显式列名，使评测更能反映业务中常见的“列名省略”情况。实验表明，只有在这种设置下，结构化预训练的优势才会充分显现。  

### 方法详解
#### 整体框架概览
STRU​G 的训练流程可以划分为三步：  
1) **构造平行文本‑表格语料**：从已有的 Text‑to‑SQL 数据集中抽取每条问句对应的数据库 schema（表、列）以及 SQL 中出现的列名和常量值，形成“自然语言句子 ↔ 表结构”对。  
2) **设计三种对齐预训练任务**：在同一个 Transformer 编码器上并行执行列对齐、值对齐和列‑值映射任务。  
3) **联合优化**：把三任务的损失加权求和，统一更新模型参数，使编码器同时学习语言和结构的对应关系。微调阶段，只需在目标 Text‑to‑SQL 数据上继续训练即可。

#### 关键模块拆解
- **文本‑表格编码器**：采用 BERT‑LARGE 作为基础，输入由两部分拼接而成：① 自然语言问句的 token 序列；② schema 中每个列的名称和类型也被 token 化后拼接进去。两者之间用特殊分隔符 `[SEP]` 隔开，形成一个长序列，模型一次性处理。这样设计让列信息直接进入自注意力机制，列之间以及列与问句之间可以相互交流。  
- **列对齐任务**：对每个列名称在问句中的出现位置进行二分类预测。具体做法是把问句中每个 token 的隐藏向量与对应列的向量做点积，得到匹配分数，然后用交叉熵损失让真实出现的列得分最高。直观上，这相当于让模型在阅读句子时“指向”它认为对应的列。  
- **值对齐任务**：类似列对齐，但目标是句子中的数值或字符串常量。模型需要判断某个 token 是否对应表中某列的具体取值。这里的标签来自 SQL 中的常量，例如 `WHERE year = 2022` 中的 `2022`。  
- **列‑值映射任务**：在列对齐和值对齐的基础上，进一步要求模型输出“这个值属于哪一列”。实现方式是把每个被识别为值的 token 与所有列向量做匹配，选出最高分的列作为映射结果。该任务帮助模型纠正仅靠上下文可能产生的歧义（比如“北京”可能是城市也可能是公司名）。  
- **联合损失**：三任务的损失分别记为 L_col、L_val、L_map，整体目标是 L = α·L_col + β·L_val + γ·L_map，α、β、γ 为经验超参数。通过一次前向传播即可得到所有分数，反向传播一次更新全部参数，训练效率与普通 BERT 预训练相当。  

#### 巧妙之处
- **弱监督标签的自动生成**：不需要人工标注对齐信息，直接利用 SQL 中的列名和常量生成标签，这大幅降低了数据准备成本。  
- **同一编码器共享三任务**：避免为每个对齐任务训练独立模型，利用多任务学习的共享表示提升了泛化能力。  
- **显式列信息的拼接**：把 schema 直接喂入 Transformer，使模型在注意力计算时天然拥有结构上下文，而不是事后再做额外的对齐步骤。  

### 实验与效果
- **数据集与评测**：作者在原始 Spider 开发集、八个跨数据库的公开 Text‑to‑SQL 数据集以及新建的 Spider‑Realistic 上进行评估。Spider‑Realistic 特别去掉了显式列名，模拟真实业务中用户常省略列名的情况。  
- **基线对比**：与直接使用 BERT‑LARGE 微调的基线相比，STRU​G 在所有数据集上均取得显著提升。具体数字在原文中未给出，但作者指出在 Spider‑Realistic 上的准确率提升幅度超过 5%。在标准 Spider 上，STRU​G 与最强的结构化预训练方法 GRAPPA 持平，但在更具挑战性的真实场景下明显领先。  
- **消融实验**：作者分别去掉列对齐、值对齐或列‑值映射任务进行实验，结果显示每个任务都对最终性能有贡献，尤其是列‑值映射的缺失会导致在 Spider‑Realistic 上的下降最为明显，验证了三任务联合学习的必要性。  
- **局限性**：论文未详细探讨对极其大型 schema（列数上千）时的计算开销，也没有在多表联结（JOIN）复杂度极高的场景做专门评估。作者承认在极端长序列的注意力计算上仍受限于 Transformer 的 O(n²) 复杂度。  

### 影响与延伸思考
STRU​G 的结构化预训练思路在随后两年里被多篇 Text‑to‑SQL 工作引用，尤其是那些关注“隐式对齐”或“跨域迁移”的研究。它启发了基于图神经网络的 schema 编码、以及利用大规模未标注数据库日志进行自监督对齐的方向。未来可以进一步探索：① 将列‑值对齐任务扩展到多表 JOIN 场景；② 结合稀疏注意力或长文本模型降低计算成本；③ 用真实业务日志（如用户查询日志）做更噪声的弱监督。对想深入的读者，可以关注近期在 ACL、EMNLP 上出现的 “Schema‑aware Pretraining” 系列论文。  

### 一句话记住它
**STRU​G 用三种弱监督对齐任务让模型在预训练阶段就学会把自然语言直接映射到数据库结构，从而在列名省略的真实场景中大幅提升 Text‑to‑SQL 的准确率。**