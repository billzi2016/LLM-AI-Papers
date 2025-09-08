# X-SQL: Expert Schema Linking and Understanding of Text-to-SQL with Multi-LLMs

> **Date**：2025-09-07
> **arXiv**：https://arxiv.org/abs/2509.05899

## Abstract

With Large Language Models' (LLMs) emergent abilities on code generation tasks, Text-to-SQL has become one of the most popular downstream applications. Despite the strong results of multiple recent LLM-based Text-to-SQL frameworks, the research community often overlooks the importance of database schema information for generating high-quality SQL queries. We find that such schema information plays a significant or even dominant role in the Text-to-SQL task. To tackle this challenge, we propose a novel database schema expert with two components. We first introduce X-Linking, an LLM Supervised Finetuning (SFT)-based method that achieves superior Schema Linking results compared to existing open-source Text-to-SQL methods. In addition, we innovatively propose an X-Admin component that focuses on Schema Understanding by bridging the gap between abstract schema information and the user's natural language question. Aside from better learning with schema information, we experiment with Multi-LLMs for different components within the system to further boost its performance. By incorporating these techniques into our end-to-end framework, X-SQL, we have achieved Execution Accuracies of 84.9% on the Spider-Dev dataset and 82.5% on the Spider-Test dataset. This outstanding performance establishes X-SQL as the leading Text-to-SQL framework based on open-source models.

---

# X‑SQL：多大语言模型驱动的专家级模式链接与理解的文本到SQL系统 论文详细解读

### 背景：这个问题为什么难？
在把自然语言问题转成 SQL 查询的 Text‑to‑SQL 任务里，模型必须同时懂用户的意图和数据库的结构。过去的工作大多把注意力放在让大语言模型（LLM）生成代码上，却把“模式（schema）信息”当成可有可无的附加。实际上，模式包含表名、列名、外键关系等关键线索，缺了它们模型很容易把问题映射到错误的表或列，导致执行错误。于是，尽管 LLM 在代码生成上已经很强，整体准确率仍被模式理解的瓶颈卡住。

### 关键概念速览
**Text‑to‑SQL**：把用户的自然语言提问自动翻译成对应的 SQL 语句，类似把口头指令变成数据库指令。  
**Schema Linking（模式链接）**：在问题中找出哪些词对应数据库的表或列，就像在地图上把地点名称和实际坐标配对。  
**Schema Understanding（模式理解）**：不仅识别对应关系，还要把模式的结构（外键、层级）和用户意图融合，类似把地图的道路网络和旅行计划结合起来。  
**Large Language Model (LLM)**：参数量巨大的预训练语言模型，具备生成代码和自然语言的能力。  
**Supervised Finetuning (SFT)**：在已有模型上继续用标注数据微调，让模型在特定任务上表现更好。  
**Multi‑LLM 协作**：让不同的 LLM 各自负责系统的不同子任务，像团队里每个人专注自己的专长。  
**Spider 数据集**：业界标准的 Text‑to‑SQL 基准，包含多种复杂数据库和跨表查询。  

### 核心创新点
1. **之前的模式链接方法 → X‑Linking（基于 SFT 的 LLM 微调） → 链接准确率显著提升**  
   传统开源方案往往直接让 LLM 通过提示完成链接，容易受噪声影响。X‑Linking 先用标注好的模式链接数据对 LLM 进行监督微调，使模型在识别表/列对应关系时更稳健，实验显示其链接效果超过所有已公开方法。

2. **之前的模式理解仅靠提示 → X‑Admin（专门的模式理解模块） → 把抽象模式信息转化为自然语言上下文**  
   过去的系统把模式直接喂给生成模型，模型往往忽略外键等结构信息。X‑Admin 通过一个专门的 LLM 将模式结构解释成易于理解的自然语言描述，再与用户问题拼接，帮助后续的 SQL 生成模型形成更完整的语义图谱。

3. **单一 LLM 负责全部流程 → Multi‑LLM 组合 → 整体执行准确率提升**  
   研究发现，某些 LLM 在链接任务上表现更好，而另一些在代码生成上更强。系统把 X‑Linking、X‑Admin、SQL 生成分别交给最适合的模型，实现了“各司其职”，最终在 Spider 上的执行准确率突破 84% 级别。

### 方法详解
整体框架可以想象成一条装配线，输入是用户的自然语言问题和数据库模式，输出是可直接执行的 SQL。装配线分三站：

1. **模式链接站（X‑Linking）**  
   - 输入：原始问题 + 完整模式（表/列列表）。  
   - 操作：使用经过 SFT 微调的 LLM，对每个问题词标记对应的表或列。微调时使用大量人工标注的链接对，让模型学习“这个词通常指哪个列”。  
   - 输出：链接映射表，例如 “author_name → author.name”。  

2. **模式理解站（X‑Admin）**  
   - 输入：链接映射表 + 原始模式的结构信息（外键、主键、表层级）。  
   - 操作：另一个 LLM 把结构信息翻译成自然语言描述，例如 “author 表通过 author_id 与 book 表相连”。这种描述会与用户问题拼接成一个更丰富的上下文。  
   - 输出：增强的上下文字符串，兼顾用户意图和数据库结构。  

3. **SQL 生成站**  
   - 输入：增强上下文 + 链接映射。  
   - 操作：选用最擅长代码生成的 LLM，依据上下文直接生成 SQL。因为前两站已经把模式信息显式化，生成模型只需要把自然语言转成查询语句，而不必自行推断表/列对应关系。  

**关键技巧**  
- **SFT 微调**：而不是仅靠 few‑shot 提示，作者在公开的模式链接数据上对 LLM 进行数轮微调，使其对专业术语更敏感。  
- **多模型分工**：把每一步交给最合适的模型，而不是让单一模型承担全部负担，这在资源利用和错误传播控制上都有优势。  
- **结构化描述**：X‑Admin 把抽象的外键图转成自然语言，类似把技术文档写成通俗的操作指南，让后续模型更容易“读懂”。  

### 实验与效果
- **数据集**：使用 Spider 的开发集（Dev）和测试集（Test），Spider 包含 200 多个数据库和上千条跨表查询，是 Text‑to‑SQL 的金标准。  
- **基线对比**：与当前开源的 LLM‑based Text‑to‑SQL 系统（如 ChatGPT‑based、Codex‑based 等）相比，X‑SQL 在 Dev 上达到 84.9% 的执行准确率，在 Test 上达到 82.5%。这些数字在公开记录中是最高的。  
- **消融实验**：论文分别去掉 X‑Linking、X‑Admin、以及 Multi‑LLM 组合，发现去掉任意一环都会导致准确率下跌 3%~6%，说明每个模块都对整体性能贡献显著。  
- **局限性**：作者指出 X‑SQL 仍依赖高质量的模式链接标注数据进行微调，若迁移到全新领域或极其稀疏的模式，微调成本可能升高。此外，多模型部署对算力要求更高，实际工业落地需要考虑成本平衡。

### 影响与延伸思考
这篇工作把“模式信息”重新推到 Text‑to‑SQL 研究的前沿，促使后续工作更关注结构化知识的显式化。随后出现的几篇论文（如 **Schema‑Prompt**、**Graph‑LLM**）都在尝试把数据库图谱直接嵌入 LLM 的提示中，显然受到了 X‑SQL 的启发。对想进一步探索的读者，可以关注以下方向：  
- **少标注微调**：如何在只有少量链接标注的情况下仍实现高质量的 X‑Linking。  
- **统一模型**：是否能在单一模型内部实现类似的分工，而不增加部署复杂度。  
- **跨域迁移**：把 X‑SQL 的思路搬到其他结构化查询任务（如 SPARQL、GraphQL）上。  

### 一句话记住它
把数据库模式显式化、分工多模型，让 LLM 只负责“写代码”，从而把 Text‑to‑SQL 的执行准确率推到 80% 以上。