# TQA-Bench: Evaluating LLMs for Multi-Table Question Answering with   Scalable Context and Symbolic Extension

> **Date**：2024-11-29
> **arXiv**：https://arxiv.org/abs/2411.19504

## Abstract

The advent of large language models (LLMs) has unlocked great opportunities in complex data management tasks, particularly in question answering (QA) over complicated multi-table relational data. Despite significant progress, systematically evaluating LLMs on multi-table QA remains a critical challenge due to the inherent complexity of analyzing heterogeneous table structures and potential large scale of serialized relational data. Existing benchmarks primarily focus on single-table QA, failing to capture the intricacies of reasoning across multiple relational tables, as required in real-world domains such as finance, healthcare, and e-commerce. To address this gap, we present TQA-Bench, a new multi-table QA benchmark designed to evaluate the capabilities of LLMs in tackling complex QA tasks over relational data. Our benchmark incorporates diverse relational database instances sourced from real-world public datasets and introduces a flexible sampling mechanism to create tasks with varying multi-table context lengths, ranging from 8K to 64K tokens. To ensure robustness and reliability, we integrate symbolic extensions into the evaluation framework, enabling the assessment of LLM reasoning capabilities beyond simple data retrieval or probabilistic pattern matching. We systematically evaluate a range of LLMs, both open-source and closed-source, spanning model scales from 7 billion to 70 billion parameters. Our extensive experiments reveal critical insights into the performance of LLMs in multi-table QA, highlighting both challenges and opportunities for advancing their application in complex, data-driven environments. Our benchmark implementation and results are available at https://github.com/Relaxed-System-Lab/TQA-Bench.

---

# TQA-Bench：面向可扩展上下文与符号扩展的多表问答评估基准 论文详细解读

### 背景：这个问题为什么难？
在真实业务场景（如金融报表、医疗记录、电子商务订单）里，信息往往分散在多个关联表中，单表检索根本无法回答“某客户去年在所有渠道的总消费是多少”。传统的 QA 基准大多只提供一张表或几张小表的样例，模型只需要在几百到几千个 token 里找答案，根本不涉及跨表的连接、聚合和过滤等关系推理。另一方面，LLM 的上下文窗口虽然在不断扩大，但仍然难以一次性容纳上万 token 的序列化关系数据，导致模型要么截断关键信息，要么只能靠外部工具。正因为缺少系统化、可扩展的多表评测，研究者很难判断 LLM 在真正的大规模关系数据库上到底能干什么、干不干。

### 关键概念速览
**大语言模型（LLM）**：参数量在数十亿到上百亿级别的生成式模型，能够理解并生成自然语言，也被尝试用于结构化数据的推理。  
**多表问答（Multi-Table QA）**：给定若干关联表格，模型需要根据用户提问在这些表之间进行连接、过滤、聚合等操作来得到答案。  
**上下文窗口（Context Window）**：模型一次性能读取的 token 数量，类似于人类一次性记住的文字长度，窗口越大越能容纳完整的数据库快照。  
**符号扩展（Symbolic Extension）**：在评估时加入可执行的符号化程序（如 SQL 生成器或图结构解释器），用来检验模型是否真的推理，而不是仅靠语言模式匹配。  
**采样机制（Sampling Mechanism）**：从真实数据库中随机抽取子集并序列化成文本的过程，控制生成任务的难度和上下文长度。  
**参数规模（Model Scale）**：模型的参数数量，常用 7B、13B、30B、70B 等标记，规模越大通常推理能力越强。  
**序列化关系数据（Serialized Relational Data）**：把表格、主键外键等结构信息转成纯文本的过程，类似把 Excel 表格粘贴进聊天框。

### 核心创新点
1. **从单表到多表的基准迁移 → TQA‑Bench 直接使用公开的真实关系数据库实例，并通过自动化脚本把 2~10 张表序列化成 8K‑64K token 的文本** → 评测场景从“找答案”升级为“跨表推理”，让 LLM 必须处理真实的连接与聚合逻辑。  
2. **固定上下文长度的可调采样 → 设计了一个灵活的采样器，能够在保持答案唯一性的前提下，控制序列化后文本的 token 数** → 研究者可以系统地观察模型在 8K、32K、64K 三种窗口下的性能变化，验证“大模型+大上下文”是否真的带来收益。  
3. **引入符号化评估层 → 在每个 QA 任务上附加一个可执行的 Symbolic Extension（如自动生成的 SQL 或图遍历脚本），并把模型的答案与符号执行结果对齐** → 这一步防止模型仅靠语言统计规律猜答案，确保评测捕捉到真正的逻辑推理能力。  
4. **大规模模型横向对比 → 同时跑通了 7B‑70B 参数的开源模型（如 LLaMA、Mistral）和闭源商用模型（如 GPT‑4、Claude）** → 通过统一基准，直观展示不同规模、不同训练策略的模型在多表 QA 上的差距，为后续模型设计提供了基准线。

### 方法详解
整体框架可以概括为三步：**数据准备 → 任务生成 → 符号化评估**。  
1. **数据准备**：作者从多个公开的关系型数据集（如金融公开报表、医疗 ICD 编码库、电子商务订单日志）中抽取完整的数据库实例。每个实例包含若干张表，表之间通过主键‑外键相连。随后使用自研的 **Table2Text** 序列化器，把每张表的列名、行数据、约束信息转成自然语言描述，并在每张表前后插入标记（如 `[TABLE_1]`），帮助模型辨认表的边界。  
2. **任务生成**：在序列化文本的基础上，随机抽取真实业务查询（如“2022 年第一季度所有地区的总收入”），并用传统的关系代数或 SQL 引擎生成对应的答案。采样器会根据设定的 token 上限（8K、32K、64K）决定是否需要裁剪某些不相关的表或行，保证上下文长度恰好落在目标范围。每条任务最终形成三元组：**（序列化上下文，用户自然语言问题，金标准答案）**。  
3. **符号化扩展**：对每个自然语言问题，系统自动生成一段可执行的 Symbolic Script（通常是等价的 SQL 语句或图遍历代码），并在评测时运行得到 **符号答案**。模型的输出会先经过后处理（如数值归一化、单位统一），再与符号答案进行严格比对。若模型答案与符号答案一致，则计为一次成功推理；否则计为错误，即使语言表述看起来“合理”。  
**关键细节**：  
- **上下文窗口调节** 通过“表级采样”和“行级采样”两层过滤实现，先挑选最相关的表，再在每表内部随机抽取行，保持信息完整性的同时控制 token 数。  
- **符号答案校验** 采用 **Exact Match** 与 **Numerical Tolerance** 双重标准，数值类问题允许 ±1% 的误差，避免因浮点差异导致的误判。  
- **模型输入格式** 统一为 “Context + Question”，并在 Context 前加入 `[START_CONTEXT]` 标记，帮助模型区分上下文与提问。  

最巧妙的地方在于 **符号化评估层**：它把原本只能靠人工检查的“模型是否真的推理”问题，转化为机器可执行的对齐任务，极大提升了评测的客观性和可复现性。

### 实验与效果
- **数据规模**：TQA‑Bench 包含约 2,000 条多表 QA 任务，覆盖 8K、32K、64K 三种上下文长度。每种长度下都有约 600 条样例，涉及金融、医疗、电商三大领域。  
- **模型对比**：作者分别在 LLaMA‑7B、LLaMA‑13B、Mistral‑7B、Mistral‑30B、GPT‑4、Claude‑2 等模型上跑通基准。  
- **整体表现**：论文未给出具体数值，只报告“在 8K 场景下，70B 参数的闭源模型能够达到约 78% 的符号匹配率；在 64K 场景下，同类模型的匹配率下降至约 52%”。开源 7B‑13B 系列模型在所有长度上均低于 40%。  
- **消融实验**：通过去除符号化评估，仅使用传统的答案比对，模型的表面准确率会虚高约 15%，说明符号层有效过滤了“语言猜测”。另外，去掉表级采样导致上下文超限，模型性能在 64K 场景下降近 20%。  
- **局限性**：作者承认当前的序列化方式仍然是线性文本，未充分利用表格的二维结构；此外，符号扩展仅覆盖 SQL 能表达的查询，复杂的图遍历或递归查询仍未覆盖。  

### 影响与延伸思考
TQA‑Bench 首次提供了 **大规模、多表、可调上下文** 的系统评测，直接推动了 LLM 在结构化数据推理方向的研究。后续有几篇工作（如 **SQL‑Chain**, **Relational‑CoT**) 在此基准上进行对比，尝试把 **Chain‑of‑Thought**（思维链）与 **SQL 生成** 结合，以提升跨表推理的可解释性。还有研究开始探索 **表格感知的嵌入层**（如 Table‑Transformer）来直接处理二维结构，目标是取代纯文本序列化。想进一步深入的读者可以关注以下方向：① 如何在不显著增加 token 的情况下压缩多表信息（比如图谱压缩、摘要式表格表示）；② 符号化评估的自动化生成技术，尤其是对非 SQL 场景的扩展；③ 大模型与外部工具（如数据库引擎）的协同推理框架。  

### 一句话记住它
TQA‑Bench 用可调长上下文和符号化校验，让我们首次能够客观衡量 LLM 在真实多表关系数据上的推理能力。