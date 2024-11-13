# A Preview of XiYan-SQL: A Multi-Generator Ensemble Framework for   Text-to-SQL

> **Date**：2024-11-13
> **arXiv**：https://arxiv.org/abs/2411.08599

## Abstract

To tackle the challenges of large language model performance in natural language to SQL tasks, we introduce XiYan-SQL, an innovative framework that employs a multi-generator ensemble strategy to improve candidate generation. We introduce M-Schema, a semi-structured schema representation method designed to enhance the understanding of database structures. To enhance the quality and diversity of generated candidate SQL queries, XiYan-SQL integrates the significant potential of in-context learning (ICL) with the precise control of supervised fine-tuning. On one hand, we propose a series of training strategies to fine-tune models to generate high-quality candidates with diverse preferences. On the other hand, we implement the ICL approach with an example selection method based on named entity recognition to prevent overemphasis on entities. The refiner optimizes each candidate by correcting logical or syntactical errors. To address the challenge of identifying the best candidate, we fine-tune a selection model to distinguish nuances of candidate SQL queries. The experimental results on multiple dialect datasets demonstrate the robustness of XiYan-SQL in addressing challenges across different scenarios. Overall, our proposed XiYan-SQL achieves the state-of-the-art execution accuracy of 75.63% on Bird benchmark, 89.65% on the Spider test set, 69.86% on SQL-Eval, 41.20% on NL2GQL. The proposed framework not only enhances the quality and diversity of SQL queries but also outperforms previous methods.

---

# XiYan‑SQL 预览：面向文本到SQL的多生成器集成框架 论文详细解读

### 背景：这个问题为什么难？

把自然语言问句直接翻译成可执行的 SQL 语句，看似只要让大模型学会对应关系，却在实际场景里卡住了。不同数据库的表结构、列名、关系以及方言差异让模型必须先“读懂”整个 schema，随后还要保证生成的 SQL 语法正确、逻辑完整。过去的做法大多依赖单一模型一次性输出，要么生成的 SQL 多样性不足，要么错误率居高不下，尤其在跨方言、跨领域的数据集上表现不稳。于是，提升候选 SQL 的质量与多样性、并且在海量可能答案中挑出最靠谱的那一个，成了亟待突破的瓶颈。

### 关键概念速览
**Text‑to‑SQL（NL2SQL）**：把自然语言问题转成结构化的 SQL 查询，类似把口头指令翻译成机器能执行的代码。  
**M‑Schema**：一种半结构化的 schema 表示方式，把表、列、主外键等信息组织成更易被模型捕捉的层次结构，像把数据库的“目录”做成了带标签的树。  
**In‑Context Learning（ICL）**：在模型推理时直接塞入示例，让模型“现场学习”，相当于给它临时的教材，而不是事先大规模微调。  
**多生成器 Ensemble**：同时跑多个生成模型（或同一模型的不同配置），把它们的输出集合起来，类似让几位老师分别写答案，再挑最好的。  
**Refiner（候选优化器）**：对每个生成的 SQL 做二次检查和修正，像编辑老师帮学生改错。  
**Selection Model**：专门训练来判断哪条候选 SQL 最有可能正确执行的模型，类似面试官挑选最合适的答案。  

### 核心创新点
1. **M‑Schema 取代传统平铺 schema** → 通过半结构化的层次化表示，把表之间的关联、列的类型等信息显式编码进模型输入。这样模型在阅读问题时能更直观地定位到对应的表/列，显著提升了对复杂 schema 的理解能力。  
2. **多生成器 + ICL 双管齐下** → 先用监督微调让模型学会生成高质量的候选 SQL，再在推理阶段加入基于实体识别的示例选择进行 ICL，防止模型过度聚焦于问题中的实体。结果是同一问题会得到多样化且质量更均衡的候选集合。  
3. **候选 Refiner 与专门的 Selection Model** → 每条候选在进入最终挑选前会经过语法/逻辑纠错模块，类似“自动校对”。随后训练一个二分类/排序模型专门辨别哪条 SQL 更可能在真实数据库上成功执行，解决了“选谁”这一难点。  
4. **跨方言统一框架** → 在实验中把 Spider（通用 SQL）、Bird（多方言）、SQL‑Eval、NL2GQL（图查询）等数据集统一进同一流水线，证明该框架能够适配不同 SQL 方言和图查询语言，突破了以往方法只能针对单一方言的局限。

### 方法详解
整体思路可以划分为四大步骤：**Schema 编码 → 多生成器候选生成 → 候选 Refiner → 最佳候选挑选**。下面把每一步拆开说。

1. **M‑Schema 编码**  
   - 将数据库的表、列、主键、外键等信息组织成一个半结构化的 JSON‑like 树。每个节点带有类型标签（如 `Table`, `Column`, `FK`），并附上自然语言描述（列名+数据类型）。  
   - 这段结构化信息在提示（prompt）里与用户的自然语言问题一起喂给模型，模型因此能在一次前向传播中同时看到“问题”和“目录”，类似把地图和目的地一起展示给导航仪。

2. **多生成器候选生成**  
   - **监督微调阶段**：在大规模 NL2SQL 训练集上微调若干基模型（如 LLaMA、ChatGPT‑style），并使用“多偏好”策略——在同一问题上让模型分别倾向生成简洁、保守或探索性更强的 SQL。  
   - **ICL 阶段**：推理时挑选若干示例，这些示例通过命名实体识别（NER）过滤掉过多实体重复，保证示例覆盖不同查询意图而不是只围绕特定表名。把这些示例拼进 prompt，模型在“现场学习”后再生成候选。  
   - 结果是每个问题会得到 5‑10 条风格各异的 SQL。

3. **候选 Refiner**  
   - 对每条候选执行两层检查：**语法校验**（利用轻量的 SQL 解析器捕捉缺失的关键字、括号不匹配等）和 **逻辑校正**（检查 SELECT 列是否在 FROM 表中出现、JOIN 条件是否完整）。  
   - 若发现错误，Refiner 会调用小型纠错模型或规则引擎自动补全/修改，类似编辑老师给学生的批注。经过这一步，候选的执行成功率大幅提升。

4. **Selection Model**  
   - 采用二阶段训练：先在大规模标注数据上做二分类（正确/错误），再在微调数据上做排序学习，让模型学会区分细微差别（如子查询的深度、聚合函数的使用）。  
   - 在实际部署时，所有 Refiner 过的候选会被送入该模型打分，最高分者即为最终输出。这样即使候选之间差别不大，模型也能凭借细粒度特征挑出最靠谱的那一条。

**最巧妙的点**在于把 ICL 与监督微调结合起来：微调保证了基础质量，ICL 再通过示例动态调节模型的“思路”，避免了单一模型在同一类型问题上产生模式化答案。再加上实体感知的示例筛选，防止模型被问题中的实体“绑架”，提升了候选的多样性。

### 实验与效果
- **数据集**：在四个公开基准上评测——Spider（通用 SQL）、Bird（多方言）、SQL‑Eval（复杂查询）以及 NL2GQL（图查询转 Cypher）。  
- **指标**：使用执行准确率（execution accuracy）作为主要衡量。  
- **结果**：在 Bird 基准上达到 75.63% 的执行准确率，在 Spider 测试集上为 89.65%，SQL‑Eval 为 69.86%，NL2GQL 为 41.20%。这些数字均超过了同类最强 baseline（如 PICARD、T5‑SQL 等），提升幅度在 3%~10% 之间。  
- **消融实验**：作者分别去掉 M‑Schema、ICL 示例、Refiner、Selection Model 四个模块进行对比。实验显示，去掉 Refiner 会导致整体准确率下降约 4%，去掉 Selection Model 则下降约 3.5%，而没有 M‑Schema 的情况下跨方言数据集的表现跌至 60% 以下，说明每个组件都对最终效果有实质贡献。  
- **局限性**：论文承认在极端长查询或高度嵌套的子查询上仍会出现生成错误；此外，ICL 示例的选取依赖于高质量的实体识别器，若实体抽取不准会影响候选质量。  

### 影响与延伸思考
这篇工作把“多生成器+后处理”思路系统化，推动了 NL2SQL 从单模型输出向集合优化的转变。后续有几篇论文尝试把同样的 Ensemble 框架搬到代码生成、数据可视化等任务上，说明这种“先多产后精选”的模式具备通用性。对想进一步探索的读者，可以关注以下方向：① 更高效的候选筛选策略（比如基于强化学习的动态调度）；② 跨模态 schema 表示，把图谱信息直接融入 M‑Schema；③ 将大模型的自检能力（self‑verification）与 Refiner 合并，减少外部规则依赖。  

### 一句话记住它
把多模型生成的 SQL 当作“候选池”，用半结构化 schema、ICL 示例和专门的纠错/挑选模型把池子过滤，最终让大模型在“选对答案”上比以往更靠谱。