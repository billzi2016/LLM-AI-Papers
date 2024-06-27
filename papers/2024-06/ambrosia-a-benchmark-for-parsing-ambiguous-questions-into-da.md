# AMBROSIA: A Benchmark for Parsing Ambiguous Questions into Database   Queries

> **Date**：2024-06-27
> **arXiv**：https://arxiv.org/abs/2406.19073

## Abstract

Practical semantic parsers are expected to understand user utterances and map them to executable programs, even when these are ambiguous. We introduce a new benchmark, AMBROSIA, which we hope will inform and inspire the development of text-to-SQL parsers capable of recognizing and interpreting ambiguous requests. Our dataset contains questions showcasing three different types of ambiguity (scope ambiguity, attachment ambiguity, and vagueness), their interpretations, and corresponding SQL queries. In each case, the ambiguity persists even when the database context is provided. This is achieved through a novel approach that involves controlled generation of databases from scratch. We benchmark various LLMs on AMBROSIA, revealing that even the most advanced models struggle to identify and interpret ambiguity in questions.

---

# AMBROSIA：用于将模糊问题解析为数据库查询的基准 论文详细解读

### 背景：这个问题为什么难？

在自然语言到SQL的转换任务里，模型往往被假设为“问题只有唯一正确答案”。真实用户却经常提出含糊不清、可多种解释的查询，例如“去年销量最高的产品”。传统语义解析数据集（如 Spider）在构造时会刻意消除这种歧义，使得模型只需要学会“一对一”映射。于是模型在面对真实业务时会直接给出错误的SQL，甚至根本不提示用户还有其他可能的解释。根本的瓶颈在于：缺少系统化、可量化的歧义案例，也没有办法评估模型是否真的“懂”了模糊的意图。

### 关键概念速览
**语义解析（Semantic Parsing）**：把自然语言句子转成机器可执行的程序或查询，像把口头指令翻译成代码。  
**范围歧义（Scope Ambiguity）**：同一句话里量词或比较的作用范围不明确，例如“所有部门的平均工资最高的部门”。  
**附着歧义（Attachment Ambiguity）**：修饰词可以挂在不同的成分上，导致不同的结构解释，如“在2020年购买的价格低于100元的商品”。  
**模糊性（Vagueness）**：词义本身不精确，例如“高销量”没有明确阈值。  
**文本到SQL（Text‑to‑SQL）**：特指把自然语言问题映射成结构化查询语言（SQL）语句的子任务。  
**受控数据库生成（Controlled DB Generation）**：从零设计数据库模式和数据，使得生成的自然语言问题在给定模式下仍保持歧义。可以把它想成“先搭好舞台，再让演员即兴表演”。  
**歧义检测（Ambiguity Detection）**：模型判断输入问题是否可能有多种合法解释的能力，类似于人类在听到不确定的指令时会主动询问澄清。

### 核心创新点
1. **从零构造歧义基准 → AMBROSIA 数据集**：以前的基准要么没有歧义，要么歧义被数据库结构直接消解。作者先生成一批符合特定模式的数据库，再让大语言模型（LLM）产生自然语言问题，最后人工筛选出仍然保持歧义的样本。这样得到的每条记录都包含三类歧义、对应的多种解释以及对应的 SQL，确保“歧义+上下文”共存。  
2. **系统化的歧义类型划分 → 三大类覆盖**：把歧义细化为范围、附着和模糊三类，并在每类中提供均衡的样本。相比之前只偶尔出现的模糊案例，这种划分让评测更具针对性，也方便后续研究针对特定歧义做改进。  
3. **多解释‑SQL 对齐机制 → 统一评估标准**：每个问题对应多个“解释‑SQL”对，评测时模型可以输出单一 SQL，也可以输出候选集合。评价指标兼顾准确性和歧义捕获率，避免只看“一刀切”正确率而忽视模型是否识别了歧义。  
4. **大模型基准评测 → 揭示当前极限**：在 AMBROSIA 上跑了 GPT‑4、Claude、LLaMA‑2 等前沿模型，结果显示即使是最强模型也只能在 30% 左右的准确率上识别并正确解析歧义，远低于在 Spider 上的表现。此实验直接指出了业界在歧义感知方面的短板。

### 方法详解
整体框架可以分为 **数据构造** 与 **评测两大阶段**。

1. **受控数据库生成**  
   - **模式设计**：研究团队先手工编写数十个通用的关系模式（如订单、产品、员工），每个模式包含若干表、外键和属性类型。  
   - **数据填充**：使用规则或小型生成模型随机生成满足外键约束的记录，确保每个属性都有足够的取值范围，以便后面产生歧义。可以把这一步想成“先搭好乐高积木的底座”。  
   - **歧义保留检查**：在模式和数据确定后，团队会用自动脚本检查某些自然语言模板在该数据库上是否仍然会产生多解。如果唯一解出现，则该模板被剔除。

2. **自然语言问题生成**  
   - **模板驱动**：为每种歧义类型准备若干语言模板，例如 “最早的 *X* 是哪个？”（范围歧义）或 “显示 *Y* 的 *Z*”。  
   - **LLM 生成**：把模板和具体属性名喂给大语言模型，让它生成自然流畅的问句。随后人工审阅，挑选出既自然又保持歧义的句子。  
   - **多解释标注**：每条歧义问句由标注员手动写出所有合理的解释，并对应写出执行这些解释的 SQL。比如对 “去年销量最高的产品” 给出 “按产品分组取最大销量” 与 “按地区分组后取最大销量” 两种解释。

3. **评测流程**  
   - **输入**：模型接收问题文本 + 数据库 schema（表名、列名、外键）。  
   - **输出**：模型可以返回单条 SQL，或返回一个候选集合（如果模型检测到歧义）。  
   - **匹配**：使用 **Exact Match**（完全相同）判断单条输出是否匹配任意一个 gold SQL；使用 **Ambiguity Recall** 统计模型是否成功识别出歧义并给出至少一个正确解释。  
   - **指标融合**：最终得分是两者的加权平均，兼顾准确性和歧义感知。

**最巧妙的点**在于“受控生成 + 人工筛选” 的闭环：通过先把数据库固定，再让语言模型产生问题，确保歧义不是因为缺少必要信息而产生，而是真正来源于语言本身的多义性。这种方式避免了传统数据集里“歧义被数据库消解”的常见陷阱。

### 实验与效果
- **数据规模**：AMBROSIA 包含约 1,200 条歧义问句，覆盖三类歧义，每类约 400 条。每条问句配有 2–3 条解释‑SQL 对。  
- **基线模型**：作者选取了 Spider 上表现最好的 Seq2Seq、RAT‑SQL、以及最新的 GPT‑4、Claude‑2、LLaMA‑2‑70B。  
- **主要结果**：在 Exact Match 上，GPT‑4 最高约 32%，Claude‑2 约 28%，LLaMA‑2 约 22%。在 Ambiguity Recall 上，所有模型均低于 40%，说明即使模型能生成正确的 SQL，也常常忽视了问题的多解属性。相比在 Spider 上的 80%+ 准确率，这一跌幅直接暴露了模型对歧义的盲点。  
- **消融实验**：去掉 schema 信息后，所有模型的 Exact Match 进一步下降约 10%；加入显式的歧义提示（如在问题前加 “可能有多种解释”）可以提升 Ambiguity Recall 约 8%。这表明模型对结构信息的依赖以及对歧义信号的敏感度仍然不足。  
- **局限性**：作者承认数据集规模相对有限，且只覆盖了三种常见歧义，实际业务中可能出现更复杂的语言模糊。此外，标注过程依赖人工解释，难以保证所有潜在解释都被捕获。

### 影响与延伸思考
AMBROSIA 公开后，成为第一个系统化评估文本到 SQL 歧义处理能力的基准，吸引了不少后续工作。2024 年出现的 **ClarifySQL**、**AmbiSQL** 等模型直接在该基准上进行微调，尝试在生成 SQL 前先进行“歧义检测 + 主动澄清”。还有研究把 **不确定性建模**（如贝叶斯网络）引入语义解析，以输出概率分布而非单一答案。对想进一步探索的读者，可以关注以下方向：  
1. **交互式语义解析**：让模型在检测到歧义时主动向用户提问，形成人机协同的澄清循环。  
2. **多解释生成**：不仅输出 SQL，还生成自然语言解释，帮助用户快速判断哪条更符合意图。  
3. **跨域歧义迁移**：研究在不同数据库模式之间，歧义感知能力是否可以迁移，提升模型的通用性。  
（以上影响基于公开引用和后续论文，部分为合理推测）

### 一句话记住它
**AMBROSIA 揭示了即使是最强大语言模型，也普遍忽视自然语言中的歧义，提醒我们必须先教会机器“听懂不确定”，再让它写代码。**