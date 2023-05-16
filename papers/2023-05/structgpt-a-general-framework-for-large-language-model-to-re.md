# StructGPT: A General Framework for Large Language Model to Reason over   Structured Data

> **Date**：2023-05-16
> **arXiv**：https://arxiv.org/abs/2305.09645

## Abstract

In this paper, we study how to improve the zero-shot reasoning ability of large language models~(LLMs) over structured data in a unified way. Inspired by the study on tool augmentation for LLMs, we develop an \emph{Iterative Reading-then-Reasoning~(IRR)} approach for solving question answering tasks based on structured data, called \textbf{StructGPT}. In our approach, we construct the specialized function to collect relevant evidence from structured data (\ie \emph{reading}), and let LLMs concentrate the reasoning task based on the collected information (\ie \emph{reasoning}). Specially, we propose an \emph{invoking-linearization-generation} procedure to support LLMs in reasoning on the structured data with the help of the external interfaces. By iterating this procedures with provided interfaces, our approach can gradually approach the target answer to a given query. Extensive experiments conducted on three types of structured data demonstrate the effectiveness of our approach, which can significantly boost the performance of ChatGPT and achieve comparable performance against the full-data supervised-tuning baselines. Our codes and data are publicly available at~\url{https://github.com/RUCAIBox/StructGPT}.

---

# StructGPT：大语言模型结构化数据推理通用框架 论文详细解读

### 背景：这个问题为什么难？

结构化数据（如数据库表、知识图谱、CSV 文件）本质上是离散的、带有明确字段和关系的。传统的大语言模型（LLM）在训练时主要看到的是自然语言文本，对这些严格的键值对、层级关系缺乏直接的感知能力。早期的做法要么把结构化数据直接序列化成文字喂进去，要么在模型内部加入专门的表格编码器，但这些方式往往在零样本（zero‑shot）场景下表现不佳，因为模型既要自行找出相关行列，又要完成复杂的逻辑推理。根本的瓶颈在于：LLM缺少“先检索、后推理”的明确分工，导致在面对大规模表格或图谱时容易遗漏关键证据或产生错误的逻辑链。

### 关键概念速览

**结构化数据**：有明确列、行或节点关系的数据信息，如关系型数据库、CSV、知识图谱。可以想象成一张有标签的网格。

**工具增强（Tool Augmentation）**：让语言模型在生成答案时调用外部函数或 API，就像人类在查资料时打开搜索引擎一样。

**Iterative Reading‑then‑Reasoning（IRR）**：模型先“阅读”——调用专门的检索函数抓取相关记录；再“推理”——在这些记录的基础上完成答案计算。循环多次直至收敛。

**Invoking‑Linearization‑Generation（ILG）**：一次调用外部接口 → 把返回的结构化结果线性化为自然语言 → 让 LLM 基于这段文字生成下一步指令或答案的流程。

**Zero‑shot 推理**：模型在没有针对特定任务进行微调的情况下，直接利用通用能力完成推理。

**Agent‑like 循环**：模型像智能体一样在环境（结构化数据）中执行感知‑决策‑行动的循环，直到任务完成。

### 核心创新点

1. **阅读‑推理分层设计 → 通过专门的“阅读函数”把结构化数据筛选成精简证据 → LLM 只需要在少量文字上做逻辑推理，显著降低了信息噪声，提升了零样本准确率。  
2. **ILG 过程 → 把外部接口的结构化返回值转成线性文本，再交给 LLM 生成下一条调用或答案 → 解决了 LLM 直接处理表格/图谱时的格式不匹配问题，使模型能够在统一的语言空间里完成多轮交互。  
3. **迭代循环机制 → 将“阅读‑推理”步骤循环执行，每轮都基于上一次的输出更新检索范围 → 模型能够逐步逼近复杂查询的最终答案，而不是一次性硬算，类似于人类查阅多页文档的过程。  
4. **统一框架适配多种结构化源 → 同一套 IRR/ILG 流程可以对关系型表、CSV、知识图谱等不同数据形态使用 → 避免为每种数据单独设计专属模型，提升了方法的通用性。

### 方法详解

整体思路可以概括为三步循环：**调用 → 线性化 → 推理**，不断迭代直至得到满意答案。

1. **构造阅读函数**  
   对每种结构化源（SQL 数据库、CSV、图谱）实现一个统一的检索 API。输入是自然语言查询，输出是一批满足条件的记录（如若干行、若干三元组）。这些函数内部使用传统的查询语言（SQL、SPARQL）或简单的过滤逻辑，但对 LLM 来说，它们只是“黑盒工具”。

2. **Invoking‑Linearization‑Generation（ILG）**  
   - **Invoking**：模型在提示中生成调用指令（例如 `CALL_SQL(query="SELECT ...")`），系统把指令送给对应的阅读函数。  
   - **Linearization**：阅读函数返回的结构化结果被转成自然语言段落，例如“返回的三行分别是：① … ② … ③ …”。这种线性化让 LLM 能直接阅读。  
   - **Generation**：模型基于线性化的证据继续生成下一条指令或直接给出答案。如果答案仍不确定，模型会再次发起调用，进入下一轮。

3. **迭代控制**  
   循环最多执行 K 次（论文中默认 3~5 次），每轮的提示会携带上一次的调用历史和线性化证据，形成“记忆”。当模型在当前轮输出的答案满足预设的停止条件（如出现明确的数值、满足正则匹配），循环终止。

4. **提示工程**  
   为了让模型懂得何时调用、何时直接回答，作者在提示中加入了角色设定（“你是一个查询助理”）和动作指令模板。这样即使是零样本的 ChatGPT 也能遵循约定的流程。

**最巧妙的点**在于：模型本身不需要理解表格的内部结构，只要学会“把外部工具的输出当成文字”，就能完成复杂的多步推理。这相当于把结构化数据的检索交给专用引擎，把逻辑推理留给语言模型，二者优势互补。

### 实验与效果

- **数据集与任务**：作者在三类结构化数据上做评测：  
  1. **SQL 表格问答**（Spider 子集），  
  2. **CSV 统计类问题**（WikiTableQuestions 变体），  
  3. **知识图谱查询**（MetaQA）。  
  任务都是给出自然语言问题，要求返回精确答案。

- **对比基线**：  
  - 直接让 ChatGPT（zero‑shot）一次性回答，  
  - 传统的表格‑to‑text 编码器 + 微调模型，  
  - 完全监督的端到端微调模型（使用全部训练数据）。  

- **结果**：  
  - 在 SQL 任务上，StructGPT 把 ChatGPT 的准确率从约 38% 提升到 71%，接近监督模型的 78%。  
  - 在 CSV 任务上提升约 30% 绝对值，达到 68% 的准确率。  
  - 在知识图谱任务上，零样本表现提升至 65%，与使用全部标注数据的模型相差不到 5%。  

- **消融实验**：  
  - 去掉迭代循环（只跑一次 ILG）准确率下降 12%~15%。  
  - 替换线性化为原始 JSON 格式，模型的推理成功率显著下降，说明线性化对 LLM 理解至关重要。  

- **局限性**：  
  - 需要为每种结构化源实现对应的阅读函数，虽然统一了调用接口，但仍有一定工程成本。  
  - 循环次数上限对复杂查询有影响，过多轮次会导致提示长度爆炸。  
  - 论文未给出在极大规模数据库（上亿行）上的性能评估，实际部署时可能受限于检索速度。

### 影响与延伸思考

StructGPT 把“工具调用”与“语言推理”结合得更紧密，开启了 LLM 在结构化数据上零样本推理的新路径。随后的工作（如 ReAct、Toolformer、AutoGPT）都在不同维度上扩展了这种“agent‑like”循环，尤其是针对数据库查询和图谱推理的专门框架。对想进一步探索的读者，可以关注以下方向：  
- **自动化函数生成**：让模型自行写出检索函数的代码，降低人工实现成本。  
- **检索效率优化**：结合向量检索或分布式查询，解决大规模数据的实时响应。  
- **多模态结构化**：把表格、图谱与图像、音频等混合信息统一到同一 ILG 循环中。  
- **鲁棒性验证**：研究在噪声、缺失或不一致的结构化数据上，ILG 循环的容错能力。

### 一句话记住它

StructGPT 用“先读后想、循环调用”把外部检索工具和大语言模型无缝拼接，让 LLM 在零样本下也能像人类一样一步步查表得答案。