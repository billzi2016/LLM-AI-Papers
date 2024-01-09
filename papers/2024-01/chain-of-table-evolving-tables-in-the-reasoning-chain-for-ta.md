# Chain-of-Table: Evolving Tables in the Reasoning Chain for Table   Understanding

> **Date**：2024-01-09
> **arXiv**：https://arxiv.org/abs/2401.04398

## Abstract

Table-based reasoning with large language models (LLMs) is a promising direction to tackle many table understanding tasks, such as table-based question answering and fact verification. Compared with generic reasoning, table-based reasoning requires the extraction of underlying semantics from both free-form questions and semi-structured tabular data. Chain-of-Thought and its similar approaches incorporate the reasoning chain in the form of textual context, but it is still an open question how to effectively leverage tabular data in the reasoning chain. We propose the Chain-of-Table framework, where tabular data is explicitly used in the reasoning chain as a proxy for intermediate thoughts. Specifically, we guide LLMs using in-context learning to iteratively generate operations and update the table to represent a tabular reasoning chain. LLMs can therefore dynamically plan the next operation based on the results of the previous ones. This continuous evolution of the table forms a chain, showing the reasoning process for a given tabular problem. The chain carries structured information of the intermediate results, enabling more accurate and reliable predictions. Chain-of-Table achieves new state-of-the-art performance on WikiTQ, FeTaQA, and TabFact benchmarks across multiple LLM choices.

---

# 链式表格：在推理链中演化表格用于表格理解 论文详细解读

### 背景：这个问题为什么难？
表格理解任务（如基于表格的问答或事实验证）要求模型同时把握自然语言问题的意图和表格中半结构化信息的含义。传统的大语言模型（LLM）在处理纯文本时可以直接生成思考链，但面对表格时往往只能把表格“平铺”成文字，导致信息丢失或误解。已有的思维链（Chain‑of‑Thought）方法把推理过程写成一段段文字，却没有办法让模型在推理过程中直接操作表格结构。于是，如何让模型在推理时既保留表格的结构化优势，又能像写草稿一样逐步演进推理步骤，成为瓶颈。

### 关键概念速览
**表格推理（Table Reasoning）**：在给定的表格和自然语言问题之间进行信息抽取、计算或比较，以得到答案的过程。想象成在一张电子表格里查找、筛选、计算后得出结论。  
**思维链（Chain‑of‑Thought, CoT）**：让模型在输出最终答案前，先把每一步推理写出来，类似于解数学题时的草稿。  
**表格原子操作（Table Atomic Operation）**：对表格执行的最小单元动作，如筛选行、添加列、聚合数值等，类似于 Excel 中的单个函数调用。  
**表格演化（Table Evolution）**：在推理过程中，模型不断生成新的表格版本，每一次操作都把上一步的结果写进表格，形成一条结构化的推理链。  
**上下文学习（In‑Context Learning）**：把示例和指令直接放进模型的输入，让模型在不调参的情况下学习如何完成任务。  
**中间表格（Intermediate Table）**：推理链中每一步产生的临时表格，保存了当前的计算状态，后续步骤可以直接基于它继续操作。  
**结构化中间表示（Structured Intermediate Representation）**：相较于纯文字的思考链，使用表格本身作为中间表示，使得信息保持原始的行列关系，便于后续的数值或逻辑运算。

### 核心创新点
- **传统思维链 → 用表格做中间表示 → 推理过程从文字草稿转为结构化表格**  
  以前的 CoT 只能把每一步写成一句话，信息密度受限且难以直接进行数值运算。本文让模型把每一步的结果写进表格本身，既保留了行列关系，又能直接进行筛选、求和等操作，提升了对数值密集型问题的准确性。  

- **一次性平铺表格 → 迭代生成表格操作 → 动态规划推理路径**  
  过去的做法往往把整个表格一次性转成文本，模型只能一次性给出答案，缺乏“思考”过程。这里通过在上下文中提示模型“先做筛选，再聚合”，让模型在每一步生成具体的表格操作指令，并根据上一步的输出更新表格，实现了类似编程语言的循环/条件控制。  

- **统一的操作语言 → 多模型兼容 → 跨 LLM 取得 SOTA**  
  作者设计了一套简洁的表格操作语法（如 `FILTER(col>5)`、`ADD(col_sum)`），所有支持自然语言的 LLM 都能通过上下文学习掌握。实验表明，无论是 GPT‑3.5、Claude 还是 LLaMA，都能在同一框架下实现显著提升，说明创新点不依赖特定模型。  

- **结构化链条 → 更可靠的错误追溯 → 提升可信度**  
  由于每一步都留下了完整的表格快照，若最终答案错误，研究者可以直接回溯到哪一步的操作产生了异常，而不是在一长段文字里寻找线索。这种可追溯性在事实验证任务（TabFact）中特别有价值。

### 方法详解
整体思路可以拆成三大阶段：**（1）问题与原始表格的准备，** **（2）迭代生成表格操作，** **（3）从最终表格抽取答案。** 整个过程像是让 LLM 扮演一个会写 Excel 脚本的助理，先读懂问题，再一步步在表格上敲指令，最后把结果读出来。

1. **输入包装**  
   - 将原始表格转成一种易读的文本形式（行列标题 + 前几行示例），并在同一段落里附上自然语言问题。  
   - 在输入的最前面放入若干“示例对”（few‑shot），每个示例展示了从问题到一系列表格操作再到答案的完整链路。这样模型在上下文中学会“先写操作 → 再更新表格 → 最后输出答案”。

2. **表格操作生成器**  
   - 模型每一次调用都只输出一条操作指令，指令遵循预定义的原子操作语法。比如 `SELECT rows WHERE Age > 30`、`GROUP BY Country SUM(Sales)`。  
   - 生成的指令随后交给一个轻量的表格执行引擎（可以是 Python pandas 或者内部实现），该引擎把指令作用在当前表格上，产生**新的中间表格**。  
   - 新表格再被序列化回文本，拼接到模型的下一轮输入中，形成“上一轮的结果 + 下一轮的任务”循环。

3. **动态规划与终止判定**  
   - 为了让模型知道何时停止，提示中加入了“如果已经可以直接回答，请输出 `ANSWER:` 并给出答案”。模型在看到足够的中间信息后，会自行切换到答案模式。  
   - 若模型在若干轮仍未输出 `ANSWER:`，系统会强制截断，防止无限循环。

4. **答案抽取**  
   - 当模型输出 `ANSWER:` 时，后面的文字即为最终答案。因为中间表格已经把所有必要的计算结果浓缩进来，答案往往只需要一次简单的查找或取值。

**最巧妙的地方**在于把“思考链”从纯文字搬到了可执行的表格操作上。模型不再需要在脑海里模拟数值运算，而是把运算交给了真正的表格引擎，这相当于让 LLM 只负责“写代码”，让专门的执行环境负责“跑代码”。这种职责分离让推理更可靠，也让模型的负担更轻。

### 实验与效果
- **测试数据集**：WikiTableQuestions（WikiTQ）、FeTaQA、TabFact，这三个基准分别覆盖了开放域表格问答、事实验证和多步推理。  
- **对比基线**：传统的 LLM+CoT、Table‑Prompt、TaLM、TAPAS 等方法。论文声称在所有三个数据集上均刷新了最高分，尤其在 TabFact 上提升了约 4% 的准确率，在 WikiTQ 和 FeTaQA 上分别超过 3% 和 2.5%。  
- **消融实验**：去掉表格操作语法、只使用文字思维链、或不提供示例对，性能均出现明显下降，说明“表格原子操作”和“few‑shot 示例”是关键驱动因素。  
- **模型兼容性**：在 GPT‑3.5、Claude‑2、LLaMA‑2‑13B 三种不同规模的 LLM 上实验，所有模型均实现了正向提升，验证了方法的模型无关性。  
- **局限性**：论文承认对极大表格（行数上万）会导致上下文长度爆炸，因为每一步都要把表格序列化进提示；此外，当前的操作集合仍然是手工设计的，若遇到更复杂的 SQL‑like 查询可能需要扩展语法。

### 影响与延伸思考
链式表格的思路打开了“结构化中间表示”在 LLM 推理中的新大门，随后出现了几篇工作尝试把图结构、树结构甚至代码抽象同样放进推理链里（如 Graph‑of‑Thought、Code‑of‑Thought）。未来可能的方向包括：① 自动学习表格操作语言，使模型自行发现更高效的原子指令；② 与检索系统结合，让模型在大规模表格库中先检索相关子表再进行链式演化；③ 设计更紧凑的表格序列化方式，缓解上下文长度限制。想深入了解的读者可以关注近期的 “Table‑LLM” 系列论文以及在 ACL/EMNLP 上的后续工作。

### 一句话记住它
让大语言模型把每一步推理写进可执行的表格，形成结构化的思考链，从而在表格问答和事实验证上实现了新一轮的性能突破。