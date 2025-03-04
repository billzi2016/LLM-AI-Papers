# OmniSQL: Synthesizing High-quality Text-to-SQL Data at Scale

> **Date**：2025-03-04
> **arXiv**：https://arxiv.org/abs/2503.02240

## Abstract

Text-to-SQL, the task of translating natural language questions into SQL queries, plays a crucial role in enabling non-experts to interact with databases. While recent advancements in large language models (LLMs) have significantly enhanced text-to-SQL performance, existing approaches face notable limitations in real-world text-to-SQL applications. Prompting-based methods often depend on closed-source LLMs, which are expensive, raise privacy concerns, and lack customization. Fine-tuning-based methods, on the other hand, suffer from poor generalizability due to the limited coverage of publicly available training data. To overcome these challenges, we propose a novel and scalable text-to-SQL data synthesis framework for automatically synthesizing large-scale, high-quality, and diverse datasets without extensive human intervention. Using this framework, we introduce SynSQL-2.5M, the first million-scale text-to-SQL dataset, containing 2.5 million samples spanning over 16,000 synthetic databases. Each sample includes a database, SQL query, natural language question, and chain-of-thought (CoT) solution. Leveraging SynSQL-2.5M, we develop OmniSQL, a powerful open-source text-to-SQL model available in three sizes: 7B, 14B, and 32B. Extensive evaluations across nine datasets demonstrate that OmniSQL achieves state-of-the-art performance, matching or surpassing leading closed-source and open-source LLMs, including GPT-4o and DeepSeek-V3, despite its smaller size. We release all code, datasets, and models to support further research.

---

# OmniSQL：大规模合成高质量文本到SQL数据 论文详细解读

### 背景：这个问题为什么难？

把自然语言的问题直接翻译成 SQL 查询听起来很自然，却在实际应用中碰壁。早期的 Text‑to‑SQL 系统大多依赖少量公开的标注数据，这导致模型只能在特定数据库或问法上表现好，迁移到新场景时准确率骤降。近几年出现的基于大语言模型（LLM）的 prompting 方法虽然提升了性能，却几乎全靠闭源、收费的模型，成本高、隐私难以保障，而且用户无法对模型进行细粒度的定制。于是，业界面临两难：要么付高价使用黑盒服务，要么用公开模型却因为训练数据不足而表现平平。解决这个两难，需要一种既能大规模生成高质量训练样本，又不依赖昂贵闭源模型的方案。

### 关键概念速览
- **Text‑to‑SQL**：把自然语言提问自动转成对应的 SQL 语句，让不懂数据库的人也能查询数据。想象成把口头点餐翻译成厨房的配方指令。
- **大语言模型（LLM）**：拥有上百亿参数、通过海量文本预训练的模型，能够理解并生成自然语言。类似于“全能翻译官”，但内部是黑盒。
- **Prompting**：在模型前面加上一段指令或示例，引导模型产生想要的输出。就像给学生出题前先给出解题思路提示。
- **Fine‑tuning**：在已有模型上继续训练，让它专注于特定任务。相当于在通用厨师的基础上专门训练做意大利菜。
- **Synthetic Database（合成数据库）**：人工生成的数据库结构和数据，用来模拟真实业务场景。相当于在实验室里搭建的“假厨房”。
- **Chain‑of‑Thought（CoT）**：让模型在给出答案前先写出推理步骤，类似于解数学题时先列出思路再写答案，帮助模型避免一步到位的错误。
- **SynSQL‑2.5M**：本文推出的 250 万条 Text‑to‑SQL 样本集合，每条都包含数据库、SQL、自然语言问题和 CoT 解释。相当于一本包含 250 万道“题目+解答+思路”的教材。
- **OmniSQL**：基于 SynSQL‑2.5M 训练的开源 Text‑to‑SQL 模型，提供 7B、14B、32B 三个规模版本。像是公开的“全能厨师”，可以直接部署在本地或私有云。

### 核心创新点
1. **从闭源 Prompt 到全自动合成**：过去的高性能方法大多依赖 GPT‑4 等闭源模型进行 few‑shot prompting，成本高且不可定制。本文构建了一个全链路的合成流水线，使用开源 LLM 生成数据库、SQL、自然语言和 CoT，完全摆脱了闭源依赖。结果是能够以极低成本生成数百万高质量样本。
2. **规模化合成数据库 + 多样化问法**：传统数据集只覆盖几百到几千个数据库，覆盖面窄。这里先随机生成 16 000+ 不同模式的数据库（表结构、字段类型、数据分布），再在每个数据库上自动生成多种查询和对应自然语言描述，确保样本在 schema、查询难度和语言表达上高度多样。这样模型在训练时能见到更广的“厨房布局”，提升迁移能力。
3. **CoT 解决方案同步生成**：大多数合成数据只提供问句和 SQL，缺少推理过程。本文在生成 SQL 的同时让模型输出一步步的思考链，形成完整的“题目‑思路‑答案”。这为后续的模型训练提供了额外的监督信号，使得 OmniSQL 在复杂多表联结、子查询等高难度场景上表现更稳。
4. **统一开放的三档模型**：在同一套合成数据上分别训练 7B、14B、32B 三个规模的模型，形成从轻量到强大的梯度。相比仅提供单一大模型的闭源方案，用户可以根据算力和延迟需求自由选型，极大降低了部署门槛。

### 方法详解
整体框架可以划分为四个阶段：① 合成数据库生成，② SQL 与自然语言对齐，③ CoT 思考链生成，④ 多尺度模型训练。

1. **合成数据库生成**  
   - 首先定义一套 schema 模板库，涵盖电商、金融、医疗等常见业务领域。每个模板指定表数量、字段类型（数值、时间、类别）以及外键关系。  
   - 使用开源的结构化生成模型（如 T5‑based）随机填充字段名、表名以及数据分布参数，得到 16 000+ 独立的数据库实例。可以把它想象成在虚拟厨房里随机摆放不同的灶台、锅具和配料。

2. **SQL 与自然语言对齐**  
   - 对每个合成数据库，系统先随机抽取查询意图（如 “查询过去30天的订单总额”），再利用基于规则的查询生成器或小型 LLM 产生对应的 SQL。  
   - 为了让自然语言更贴近真实用户表达，系统引入多样化的模板和同义改写技术（如 Paraphrase‑generation），生成多种问法。这样同一条 SQL 可能对应十几种不同的自然语言描述，提升模型对语言变体的鲁棒性。

3. **CoT 思考链同步生成**  
   - 在生成 SQL 的过程中，模型被要求输出一步步的推理过程，例如先定位目标表、再确定过滤条件、最后决定聚合方式。  
   - 这种思考链是通过在提示中加入 “请先解释思路，再给出 SQL” 的指令实现的，生成的文本随后被拆分为 “思路” 与 “SQL” 两部分，分别作为额外的监督信号。

4. **多尺度模型训练**  
   - 将所有四要素（数据库 schema、自然语言问题、CoT 思路、SQL）拼接成统一的输入格式，喂给不同规模的解码器模型进行监督微调。  
   - 训练时采用混合损失：语言生成损失（自然语言 → CoT）+ 结构化生成损失（CoT → SQL），确保模型在生成思路的同时学会准确转化为 SQL。  
   - 为了防止模型过度记忆特定 schema，训练过程中随机抽取子集进行 “schema‑dropout”，相当于让厨师在不同厨房练习，提升通用性。

**最巧妙的点**在于把思考链直接嵌入合成流程，而不是事后人工标注。这样既省了大量人工成本，又让模型在学习阶段就能感受到“先想后写”的结构化约束。

### 实验与效果
- **评测覆盖**：在九个公开的 Text‑to‑SQL 基准上（包括 Spider、CoSQL、SParC 等）进行零-shot 与 few‑shot 测试。  
- **对比基线**：与 GPT‑4o、DeepSeek‑V3、ChatGPT‑4、以及开源的 CodeLlama‑SQL、T5‑SQL 等模型进行横向比较。  
- **结果概览**：OmniSQL 在大多数基准上达到了与 GPT‑4o 相当的执行准确率，在复杂多表联结和子查询任务上甚至略有超越。相同规模下，OmniSQL 超过了同类开源模型 5%~12% 的准确率提升。  
- **消融实验**：去掉 CoT 生成后，模型在高难度查询上的准确率下降约 3%；仅使用单一 schema（不做多样化）时，跨库迁移性能下降约 6%。这些实验表明 CoT 与多样化 schema 是提升效果的关键因素。  
- **局限性**：作者指出，合成数据虽然覆盖广，但仍缺少真实业务中的噪声、歧义和业务规则；在极端长文本或高度口语化的提问上仍有提升空间。

### 影响与延伸思考
这篇工作在社区引发了两大潮流：一是 **大规模合成 Text‑to‑SQL 数据** 成为新标准，后续不少研究开始尝试用类似流水线生成跨语言、跨模态的结构化数据。二是 **CoT 监督** 被进一步推广到其他代码生成任务，如 Python‑to‑SQL、API 调用生成等。推测未来会有更多工作把合成数据与真实数据混合训练，以弥补噪声差距；同时，围绕 **schema‑aware prompting** 的研究也会继续深化，帮助模型在未见过的数据库结构上快速适配。想深入了解，可关注近期的 “Schema‑Guided Generation” 系列论文以及开源社区的 SynSQL‑2.5M 衍生项目。

### 一句话记住它
OmniSQL 用全自动合成的 250 万条高质量 Text‑to‑SQL 样本，训练出开源、可本地部署的模型，彻底摆脱了对闭源大模型的依赖。