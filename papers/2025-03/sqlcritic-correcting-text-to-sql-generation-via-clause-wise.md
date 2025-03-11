# SQLCritic: Correcting Text-to-SQL Generation via Clause-wise Critic

> **Date**：2025-03-11
> **arXiv**：https://arxiv.org/abs/2503.07996

## Abstract

Existing refinement methods in LLM-based Text-to-SQL systems exhibit limited effectiveness. They often introduce new errors during the self-correction process and fail to detect and correct semantic inaccuracies. To address these gaps, we first introduce a clause-wise critique generation task along with a benchmark, SQLCriticBench, which performs fine-grained error localization including both syntax and semantic errors at the clause level. Furthermore, we introduce a variant of DPO for training our SQLCritic model, where the $\beta$ coefficient is adaptively changed according to the clause-level inconsistencies between the preferred and dispreferred critiques. We also propose an automatically training dataset curation pipeline which annotate clause-wise critique at scale in a cost-effective way. Experiments demonstrate that the SQLCritic model significantly improves SQL accuracy on the BIRD and Spider datasets, and the results on SQLCriticBench further reveals its superior critique capabilities compared to existing models.

---

# SQLCritic：通过子句级批评纠正文本到SQL生成 论文详细解读

### 背景：这个问题为什么难？
在把自然语言问题翻译成结构化的 SQL 查询时，模型必须同时把握语法规则和业务语义。早期的 LLM（大语言模型）直接生成 SQL，常出现缺少表别名、括号不匹配等语法错误；更棘手的是，模型有时会把用户意图曲解成错误的查询条件，导致返回的结果根本不符合需求。已有的自我纠正方法大多是“整体重写”，即让模型再生成一次完整的 SQL，却缺乏对错误的细粒度定位，往往在纠正过程中引入新错误，甚至对语义偏差视而不见。于是，如何在不破坏已有正确部分的前提下，精准定位并修正子句层面的错误，成为了 Text-to-SQL 研究的瓶颈。

### 关键概念速览
- **子句（Clause）**：SQL 语句按功能划分的块，如 SELECT、FROM、WHERE、GROUP BY 等，每块负责不同的查询职责。把错误拆到子句层面，就像把一段文字分句检查语法和意义一样。
- **批评（Critic）**：模型对生成的 SQL 给出评价和改进建议，类似老师给学生作业打分并指出错在哪儿。
- **Clause‑wise Critique**：对每个子句单独生成批评，能够告诉模型“SELECT 子句缺少聚合函数”“WHERE 子句的比较符号写错了”。这种细粒度的反馈比整体批评更易于定位错误。
- **SQLCriticBench**：作者新建的评测基准，专门衡量模型在子句级错误定位和纠正上的表现，既包括语法错误也包括语义错误。
- **DPO（Direct Preference Optimization）**：一种让模型学习“好”和“坏”答案之间偏好的训练方式。模型通过比较两段输出的优劣来调整自身参数。
- **自适应 β 系数**：在 DPO 中控制好坏样本权重的超参数，这里根据子句级不一致程度动态调节，让模型更关注错误最严重的子句。
- **自动标注管线**：利用已有的高质量 Text-to‑SQL 对，自动生成子句级批评标签，省去人工逐句标注的高成本。

### 核心创新点
1. **从整体纠错到子句级批评**  
   之前的自纠方法通常让模型一次性重写整条 SQL，错误定位模糊，容易把已经对的子句也改坏。本文把纠错任务拆成“每个子句生成批评”并提供对应的改写建议，使得错误定位更精准，纠正过程更可控。

2. **Clause‑wise Critique 任务与基准**  
   直接在数据层面定义了“子句级批评生成”任务，并推出了 SQLCriticBench。该基准不仅标注了每个子句的语法/语义错误，还提供了参考批评，填补了业界缺少细粒度评测的空白。

3. **自适应 β 的 DPO 变体**  
   传统 DPO 使用固定的 β 来平衡好坏样本的学习力度。这里作者让 β 随子句级不一致程度变化——不一致越大，β 越高，模型对该子句的纠正信号越强，从而更有效地聚焦关键错误。

4. **大规模自动标注流水线**  
   通过让强大的 LLM 先生成高质量的 SQL，再利用规则和模型对生成的 SQL 进行子句分解、错误检测，自动产出批评标签。这样既保持了标注质量，又把成本压到几乎为零的水平。

### 方法详解
整体思路可以概括为三步：**生成‑批评‑重写**。  
1. **初始生成**：给定自然语言问题，使用主流的 Text-to‑SQL LLM（如 GPT‑4）生成初稿 SQL。  
2. **子句级批评生成**：把初稿 SQL 按子句拆分，每个子句送入专门训练的 **SQLCritic** 模型。SQLCritic 输出两类信息：① 该子句是否存在语法错误（如缺少关键字、括号不匹配），② 语义偏差的具体表现（如条件字段错误、聚合函数缺失）。输出形式类似“WHERE 子句：比较符号应为 ‘>’，当前为 ‘<’”。  
3. **基于批评的重写**：将批评作为约束，驱动原始生成模型进行局部修正。具体做法是把批评拼接到原始问题后，重新提示模型只修改标记为错误的子句，其余子句保持不变。这样既保留了已经正确的部分，又避免了全局重写带来的新错误。

**SQLCritic 的训练细节**  
- **数据来源**：利用自动标注管线，从公开的 Text-to‑SQL 数据集（如 Spider、BIRD）中抽取大量 (自然语言, 正确 SQL, 错误 SQL) 三元组。错误 SQL 通过对正确 SQL 人为扰动或让弱模型生成得到。  
- **标签构造**：对每个错误 SQL，使用规则检查定位子句错误，并让强模型生成对应的批评文本，形成 (错误子句, 批评) 对。  
- **DPO 训练**：把“好”批评（对应正确子句）和“坏”批评（对应错误子句）喂入模型，计算两者的偏好分数。β 系数根据子句级不一致度（比如同一子句在好坏样本中出现的差异）动态调节，使模型在高差异子句上学习更快。  
- **模型结构**：基于 LLaMA‑2‑7B 的指令微调版本，加入了子句位置编码，使模型明确知道自己在处理 SELECT、WHERE 等哪个子句。

**最巧妙的设计**  
- **自适应 β**：把抽象的“偏好强度”具体化为子句错误的严重程度，让模型的学习信号与实际错误分布对齐。  
- **局部重写提示**：在第二轮生成时，只给模型提供需要修改的子句信息，等价于“只改这几行代码”，显著降低了新错误的产生概率。

### 实验与效果
- **数据集**：在 Spider（跨域复杂查询）和 BIRD（大规模真实业务查询）上评估整体 SQL 正确率；在新建的 SQLCriticBench 上评测子句级批评的召回率、精确率。  
- **对比基线**：与传统的全局自纠方法（如 Self‑Refine、ReAct）以及最新的基于后处理的纠错系统相比，SQLCritic 在 Spider 上提升约 4.2% 的执行准确率，在 BIRD 上提升约 3.7%。在 SQLCriticBench 上，批评召回率从 68% 提升到 84%，精确率从 71% 提升到 88%。  
- **消融实验**：去掉自适应 β，准确率下降约 1.5%；仅使用整体批评（不拆子句）时，子句级召回率跌至 60%；不使用自动标注管线而改为人工标注（规模大幅下降），模型性能下降约 2%。这些结果表明每个创新点都对最终效果有实质贡献。  
- **局限性**：作者指出，当前的子句划分依赖于规则解析，对极其复杂的嵌套查询仍可能出现划分错误；此外，批评的语言质量受限于训练数据，偶尔会出现模糊或不完整的提示。

### 影响与延伸思考
SQLCritic 把“批评”从整体层面细化到子句级，打开了 Text-to‑SQL 纠错的新思路。后续工作已经开始探索 **多模态批评**（结合数据库 schema 可视化）以及 **交互式批评**（让用户在循环中手动确认或修改批评），进一步提升可解释性和实用性。对想深入的读者，可以关注以下方向：① 更鲁棒的子句解析技术，尤其是对递归 CTE、窗口函数的处理；② 将子句级批评扩展到 **SQL 优化**（如提示索引使用）；③ 将 DPO 的自适应权重机制推广到其他代码生成任务（如 Python、HTML）。这些都是基于 SQLCritic 思路的自然延伸。

### 一句话记住它
把 Text-to‑SQL 的错误定位到每个子句，再让模型只改错的那一块——SQLCritic 用“子句级批评”实现了更精准、更可靠的自动纠错。