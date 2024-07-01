# TCSR-SQL: Towards Table Content-aware Text-to-SQL with Self-retrieval

> **Date**：2024-07-01
> **arXiv**：https://arxiv.org/abs/2407.01183

## Abstract

Large Language Model-based (LLM-based) Text-to-SQL methods have achieved important progress in generating SQL queries for real-world applications. When confronted with table content-aware questions in real-world scenarios, ambiguous data content keywords and nonexistent database schema column names within the question lead to the poor performance of existing methods. To solve this problem, we propose a novel approach towards Table Content-aware Text-to-SQL with Self-Retrieval (TCSR-SQL). It leverages LLM's in-context learning capability to extract data content keywords within the question and infer possible related database schema, which is used to generate Seed SQL to fuzz search databases. The search results are further used to confirm the encoding knowledge with the designed encoding knowledge table, including column names and exact stored content values used in the SQL. The encoding knowledge is sent to obtain the final Precise SQL following multi-rounds of generation-execution-revision process. To validate our approach, we introduce a table-content-aware, question-related benchmark dataset, containing 2115 question-SQL pairs. Comprehensive experiments conducted on this benchmark demonstrate the remarkable performance of TCSR-SQL, achieving an improvement of at least 27.8% in execution accuracy compared to other state-of-the-art methods.

---

# TCSR‑SQL：面向表格内容感知的自检索文本到SQL方法 论文详细解读

### 背景：这个问题为什么难？

在实际业务中，用户常常会问一些依赖具体表格数据的自然语言问题，例如“去年销量最高的产品是哪款”。传统的 Text‑to‑SQL 系统主要把注意力放在数据库的模式（schema）上，假设用户会直接提到列名或使用明确的属性。然而真实提问里常出现模糊的关键词、拼写错误，甚至根本没有出现对应的列名，导致模型在生成 SQL 时找不到匹配的字段，执行结果错误。再加上企业级表格往往非常大，完整的表数据无法一次性塞进大语言模型（LLM）的上下文窗口，信息缺失进一步削弱了模型的推理能力。于是，如何让模型在不完整的上下文中仍然能够准确捕捉到“表格内容”并生成可执行的 SQL，成为了一个亟待突破的难点。

### 关键概念速览
- **Text‑to‑SQL（NL2SQL）**：把自然语言问题自动转化为结构化的 SQL 查询语句，类似把口头指令翻译成数据库指令。
- **大语言模型（LLM）**：像 GPT‑4 这样拥有数十亿参数的生成式模型，能够在少量示例（in‑context learning）下完成复杂任务。
- **表格内容感知**：模型不仅要理解数据库的列名，还要能够识别并利用实际存储的数值或文本信息来回答问题。
- **自检索（Self‑retrieval）**：模型主动发起查询（如生成“种子 SQL”）在数据库中搜索，获取返回结果后再用这些结果来指导后续生成。
- **种子 SQL（Seed SQL）**：一条粗糙、可能不完整的 SQL，用来在数据库里做模糊搜索，帮助模型发现潜在的列或数值。
- **编码知识表（Encoding Knowledge Table）**：把检索到的列名、精确数值等信息组织成结构化的“知识卡片”，供后续生成阶段直接引用。
- **生成‑执行‑修正循环**：模型先生成 SQL，执行后检查结果是否符合预期，若不符合则利用错误信息重新生成，循环多轮直至满意。

### 核心创新点
1. **从关键词抽取到种子 SQL 的自检索桥接**  
   之前的系统大多直接把用户问题喂给 LLM，让模型一次性输出完整 SQL，缺乏对实际数据的验证。TCSR‑SQL 先让 LLM 在上下文中提取潜在的“数据内容关键词”，再根据这些关键词构造一条模糊的种子 SQL，送入数据库进行搜索。这样模型可以“先试探”，把真实的表格信息拉回到上下文里。

2. **编码知识表的显式对齐**  
   检索得到的结果往往是散落的列名或数值，直接使用会导致信息噪声。论文设计了一个编码知识表，把列名、对应的具体值以及它们在种子 SQL 中的位置统一编码，形成结构化的提示。相比于仅靠 LLM 的隐式记忆，这种显式对齐显著提升了最终 SQL 的准确性。

3. **多轮生成‑执行‑修正机制**  
   单次生成的 SQL 常常因为模糊检索不完整而出错。TCSR‑SQL 将执行结果反馈给模型，模型依据错误信息（如“列不存在”或“返回空集”）重新调整编码知识并再次生成。循环几次后，模型能够自行纠正错误，类似人类调试代码的过程。

4. **专门的表格内容感知基准**  
   为了评估上述思路，作者新建了一个包含 2115 条问题‑SQL 对的基准数据集，专注于表格内容模糊、列名缺失的场景。该基准填补了以往公开数据集对真实业务需求的空缺，为后续研究提供了统一的评测平台。

### 方法详解
**整体框架**  
TCSR‑SQL 的工作流可以概括为四步：  
1) **关键词抽取 & 种子 SQL 生成** → 2) **模糊检索** → 3) **编码知识表构建** → 4) **生成‑执行‑修正循环**。整个过程在一次对话中完成，模型在每一步都利用前一步的输出作为上下文。

**1. 关键词抽取与种子 SQL**  
模型先接收用户自然语言问题。利用 LLM 的 in‑context learning 能力，给出几个示例，让模型猜测可能涉及的列名或具体存储值（比如“2022 年”或“北京”）。这些猜测被称为“数据内容关键词”。随后，系统把这些关键词拼装成一条简化的 SQL，形式类似 `SELECT * FROM table WHERE column LIKE '%keyword%'`，这里的 `LIKE` 用来实现模糊匹配。种子 SQL 并不要求完美，只要能把潜在的相关行拉出来即可。

**2. 模糊检索**  
种子 SQL 被直接送到真实数据库执行。返回的结果可能是一小批行记录，里面包含了实际的列名和数值。此时模型获得了“硬数据”，而不是仅凭语言模型的记忆。

**3. 编码知识表构建**  
检索结果被解析成结构化的知识条目。例如，若返回的行中出现了列 `city` 的值 “上海”，系统就在编码知识表里记下 `(column=city, value=上海)`。同时，系统把原始的关键词与检索到的列名进行对齐，形成映射关系。整个表格被序列化为一段提示，放回 LLM 的上下文，确保模型在后续生成时能直接引用这些精确信息。

**4. 生成‑执行‑修正循环**  
模型基于包含编码知识表的提示生成正式的 SQL。生成后立即在数据库中执行，检查两类错误：  
- **语义错误**：如列名不存在、类型不匹配。  
- **结果错误**：如返回空集或结果与预期不符。  
如果检测到错误，系统把错误信息（例如 “列 `price` 未找到”）加入到提示中，更新编码知识表（可能需要重新检索），然后让模型重新生成。通常 2‑3 轮即可收敛到正确的查询。

**巧妙之处**  
- **自检索代替大上下文**：通过种子 SQL 把大表格的关键信息“拉进”模型，而不是一次性塞进上下文，规避了 LLM 的上下文长度限制。  
- **显式知识对齐**：把检索到的内容转化为结构化提示，让模型不必在海量语言知识中自行找答案，显著降低了幻觉（hallucination）风险。  
- **错误驱动的迭代**：把执行错误当作监督信号，形成闭环，使模型在没有人工标注的情况下自行纠错。

### 实验与效果
- **数据集**：作者构建了一个专门的表格内容感知基准，包含 2115 条自然语言问题与对应的标准 SQL，覆盖列名缺失、模糊关键词、数值检索等真实业务场景。  
- **对比基线**：实验中与几种主流 LLM‑based Text‑to‑SQL 方法（如直接提示的 GPT‑4、ChatGPT‑ZeroShot、以及基于 schema‑only 的 PICARD）进行比较。  
- **主要指标**：执行准确率（Execution Accuracy）是核心评估指标，表示生成的 SQL 在数据库上运行后是否得到正确结果。TCSR‑SQL 在该基准上比最强基线提升了 **至少 27.8%**，实现了显著的性能跃迁。  
- **消融实验**：作者分别去掉种子 SQL、编码知识表、以及多轮修正三项，发现每一项的缺失都会导致执行准确率下降 10% 以上，验证了各模块的贡献。  
- **局限性**：论文指出在极端大表（行数上亿）或高度隐私受限的环境下，种子 SQL 的执行成本仍可能成为瓶颈；此外，当前实现依赖于 LLM 的强大推理能力，模型规模下降时效果会受损。

### 影响与延伸思考
TCSR‑SQL 把“检索-生成-修正”闭环引入 Text‑to‑SQL，开启了“自检索驱动的数据库交互”新思路。后续工作（如 2024 年的 Retrieval‑Enhanced NL2SQL、2025 年的 Adaptive Prompting for SQL）纷纷借鉴了种子 SQL 与编码知识表的设计，尝试在多模态（表格+文本）或跨库查询场景中进一步推广。对想深入的读者，可以关注以下方向：  
- **检索成本优化**：如何在保证检索质量的前提下降低种子 SQL 的执行开销。  
- **跨语言/跨库迁移**：把自检索框架扩展到多语言数据库或分布式数据湖。  
- **更细粒度的错误反馈**：利用数据库的查询计划或统计信息提供更丰富的修正信号。  
- **小模型适配**：在资源受限的环境下，探索如何用更小的模型或知识蒸馏保持同等效果。

### 一句话记住它
**TCSR‑SQL 用“先搜索再生成、执行错误再修正”的闭环，让大语言模型在不把整张表塞进上下文的情况下，也能精准写出符合真实数据的 SQL。**