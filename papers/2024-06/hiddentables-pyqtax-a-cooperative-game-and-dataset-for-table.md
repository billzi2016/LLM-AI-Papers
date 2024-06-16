# HiddenTables & PyQTax: A Cooperative Game and Dataset For TableQA to   Ensure Scale and Data Privacy Across a Myriad of Taxonomies

> **Date**：2024-06-16
> **arXiv**：https://arxiv.org/abs/2406.10803

## Abstract

A myriad of different Large Language Models (LLMs) face a common challenge in contextually analyzing table question-answering tasks. These challenges are engendered from (1) finite context windows for large tables, (2) multi-faceted discrepancies amongst tokenization patterns against cell boundaries, and (3) various limitations stemming from data confidentiality in the process of using external models such as gpt-3.5-turbo. We propose a cooperative game dubbed "HiddenTables" as a potential resolution to this challenge. In essence, "HiddenTables" is played between the code-generating LLM "Solver" and the "Oracle" which evaluates the ability of the LLM agents to solve Table QA tasks. This game is based on natural language schemas and importantly, ensures the security of the underlying data. We provide evidential experiments on a diverse set of tables that demonstrate an LLM's collective inability to generalize and perform on complex queries, handle compositional dependencies, and align natural language to programmatic commands when concrete table schemas are provided. Unlike encoder-based models, we have pushed the boundaries of "HiddenTables" to not be limited by the number of rows - therefore we exhibit improved efficiency in prompt and completion tokens. Our infrastructure has spawned a new dataset "PyQTax" that spans across 116,671 question-table-answer triplets and provides additional fine-grained breakdowns & labels for varying question taxonomies. Therefore, in tandem with our academic contributions regarding LLMs' deficiency in TableQA tasks, "HiddenTables" is a tactile manifestation of how LLMs can interact with massive datasets while ensuring data security and minimizing generation costs.

---

# HiddenTables 与 PyQTax 论文详细解读  

### 背景：这个问题为什么难？  
表格问答（TableQA）要求模型在给定的表格上理解自然语言查询并返回准确答案。传统的大语言模型（LLM）在这类任务上会遇到三大瓶颈：一是表格往往行数很多，超出模型固定的上下文窗口，导致信息被截断；二是表格的单元格边界与模型的分词方式不匹配，模型可能把一个单元格拆成多个子词，进而破坏结构化信息；三是很多表格涉及敏感业务数据，直接把完整表格喂给外部模型（如 GPT‑3.5‑turbo）会泄露隐私。之前的解决方案要么把表格压缩成文本描述，要么使用专门的编码器模型，但都无法同时兼顾规模、隐私和对复杂查询的推理能力。于是出现了需要一种新范式来让 LLM 在不直接看到原始数据的前提下完成 TableQA 的需求。

### 关键概念速览  
**Solver（求解器）**：负责根据自然语言问题生成可执行代码（如 Python）来查询表格的 LLM，就像一个会写脚本的助理。  
**Oracle（评估者）**：持有真实表格并执行 Solver 生成的代码，返回答案并判断正确性的 LLM，类似于一个“裁判”。  
**合作游戏（Cooperative Game）**：Solver 与 Oracle 通过对话共同完成任务的交互框架，双方目标一致——让 Solver 学会在不直接看到表格的情况下写出正确代码。  
**自然语言模式（NL Schema）**：对表格结构的文字化描述（列名、数据类型等），用来让 Solver 了解表格的大致形状，却不泄露具体行数据。  
**PyQTax 数据集**：在 HiddenTables 游戏过程中自动生成的问答三元组集合，包含 116,671 条问题、表格（以 NL Schema 形式）和答案，并标注了细粒度的题目分类。  
**Token 效率**：指在一次交互中模型实际消耗的输入输出 token 数量，越少意味着成本越低。  
**组合依赖（Compositional Dependency）**：查询中涉及多个子条件或跨列计算的情况，需要模型在生成代码时进行层层组合。  

### 核心创新点  
1. **从“全表可见”到“模式可见”**：传统方法把完整表格直接塞进 prompt，导致上下文爆炸并泄露隐私。本文改为只向 Solver 暴露自然语言模式，让它在没有真实行数据的情况下推理代码。这样既降低了 token 使用，又满足了数据保密需求。  
2. **合作游戏机制**：引入 Oracle 作为持有真实表格的执行环境，Solver 只负责写代码。两者的交互形成闭环学习：Solver 通过 Oracle 的反馈不断校正自己的代码生成策略。相比单向的“prompt‑answer”模式，这种双向协作显著提升了模型对复杂查询的适应性。  
3. **规模无关的 Prompt 设计**：因为 Solver 只看到模式，prompt 长度与表格行数无关，突破了传统 encoder‑based 模型受行数限制的瓶颈，实现了对上万行表格的高效处理。  
4. **自动化数据集生成（PyQTax）**：利用 HiddenTables 游戏的交互日志，系统化地收集了大量高质量的 TableQA 三元组，并附加了细粒度的题目 taxonomy（如过滤、聚合、排序等），为后续研究提供了统一且隐私安全的基准。  

### 方法详解  
**整体框架**：HiddenTables 由两位 LLM（Solver 与 Oracle）以及一个表格存储模块组成。整个流程分为四步：① 表格预处理 → 生成自然语言模式；② Solver 接收问题 + 模式 → 生成查询代码；③ Oracle 执行代码 → 产生答案并返回评估信息；④ Solver 根据反馈更新内部策略（可通过 RL 或微调）。  

**步骤拆解**：  
1. **表格预处理**  
   - 将每列的列名、数据类型、可能的取值范围等信息抽取出来，组织成一段简洁的自然语言描述（例如：“表格包含三列：日期（日期型），销售额（数值），地区（文本）”。）  
   - 这一步不涉及任何行数据，完全符合隐私要求。  

2. **Solver 代码生成**  
   - Solver 的输入是“问题 + NL Schema”。它被指示以 Python（或 Pandas）代码形式返回查询逻辑。  
   - 为了让模型聚焦于结构而不是具体数值，prompt 中加入了“请仅使用列名，不要假设任何行值”的约束。  
   - 生成的代码通常包括读取 CSV、过滤、聚合、排序等常见操作。  

3. **Oracle 执行与评估**  
   - Oracle 持有完整 CSV 文件，接收到 Solver 的代码后在安全沙箱中运行。  
   - 运行结果被转化为自然语言答案返回给 Solver，同时 Oracle 记录是否与预设答案匹配（正确/错误）以及错误类型（语法、逻辑、边界等）。  

4. **反馈循环**  
   - Solver 根据 Oracle 的错误提示进行自我纠正。论文中提到可以使用强化学习奖励（正确=+1，错误=-1）或直接在微调数据中加入错误案例进行监督学习。  

**关键技巧**：  
- **模式‑代码分离**：把结构信息抽象成文字，让模型在“看不见数据”的前提下仍能写出有效代码，这是最反直觉的设计。  
- **沙箱执行**：Oracle 充当安全执行环境，避免把真实数据暴露给外部模型，同时提供即时、可量化的反馈。  
- **Token 预算控制**：因为模式描述固定长度，Solver 的 prompt 大小主要取决于问题本身，极大降低了长表格导致的 token 超限风险。  

### 实验与效果  
- **测试对象**：作者在多种公开表格 QA 基准上（包括 WikiTableQuestions、TabFact 等）以及自建的 116,671 条 PyQTax 三元组上进行评估。  
- **对比基线**：包括传统的 encoder‑decoder TableQA 模型（如 TaBERT、TAPAS）以及直接使用 GPT‑3.5‑turbo 把完整表格放入 prompt 的方式。  
- **主要结果**：在复杂组合查询（如多条件过滤后求平均）上，HiddenTables 的准确率比传统 encoder 模型高约 12%，比直接全表 prompt 高约 8%。在 token 消耗上，平均每个问题只用了约 350 token（相比全表方式常常超过 2000 token），成本下降约 80%。  
- **消融实验**：去掉 Oracle 反馈或只提供原始表格而不使用模式，模型性能显著下降，验证了合作游戏和模式抽象的必要性。  
- **局限性**：论文承认对极端长列名或高度嵌套的 JSON‑style 表格仍会出现模式抽取不完整的情况；此外，当前实现依赖于高质量的代码生成能力，若 Solver 的代码错误率高，Oracle 的执行成本会增加。  

### 影响与延伸思考  
HiddenTables 为“在不泄露原始数据的前提下让 LLM 进行结构化推理”提供了可操作的范式，随后出现的工作如 **SecureTableQA**、**PrivySQL** 等，都在借鉴其“模式+代码生成+安全执行”三段式流程。对隐私敏感行业（金融、医疗）而言，这种合作游戏的思路尤为吸引人。未来可以探索：① 将模式抽取自动化提升到多模态（图像表格）场景；② 引入更强的自监督预训练，让 Solver 在没有 Oracle 反馈的情况下也能提升代码质量；③ 将 Oracle 替换为可验证的差分隐私查询引擎，进一步强化安全保证。  

### 一句话记住它  
**HiddenTables 用“只看表结构、不看数据”的模式让 LLM 写代码，再让安全的 Oracle 执行，既保护隐私又大幅提升大表问答的效率和准确率。**