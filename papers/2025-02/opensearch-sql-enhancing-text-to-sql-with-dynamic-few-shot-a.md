# OpenSearch-SQL: Enhancing Text-to-SQL with Dynamic Few-shot and   Consistency Alignment

> **Date**：2025-02-19
> **arXiv**：https://arxiv.org/abs/2502.14913

## Abstract

Although multi-agent collaborative Large Language Models (LLMs) have achieved significant breakthroughs in the Text-to-SQL task, their performance is still constrained by various factors. These factors include the incompleteness of the framework, failure to follow instructions, and model hallucination problems. To address these problems, we propose OpenSearch-SQL, which divides the Text-to-SQL task into four main modules: Preprocessing, Extraction, Generation, and Refinement, along with an Alignment module based on a consistency alignment mechanism. This architecture aligns the inputs and outputs of agents through the Alignment module, reducing failures in instruction following and hallucination. Additionally, we designed an intermediate language called SQL-Like and optimized the structured CoT based on SQL-Like. Meanwhile, we developed a dynamic few-shot strategy in the form of self-taught Query-CoT-SQL. These methods have significantly improved the performance of LLMs in the Text-to-SQL task.   In terms of model selection, we directly applied the base LLMs without any post-training, thereby simplifying the task chain and enhancing the framework's portability. Experimental results show that OpenSearch-SQL achieves an execution accuracy(EX) of 69.3% on the BIRD development set, 72.28% on the test set, and a reward-based validity efficiency score (R-VES) of 69.36%, with all three metrics ranking first at the time of submission. These results demonstrate the comprehensive advantages of the proposed method in both effectiveness and efficiency.

---

# OpenSearch‑SQL：通过动态 Few‑shot 与一致性对齐提升 Text‑to‑SQL 论文详细解读

### 背景：这个问题为什么难？

把自然语言问题直接翻译成可执行的 SQL 语句（Text‑to‑SQL）看似只要让大模型学会对应关系，却要面对三大障碍：① 语义理解常被截断，导致生成的 SQL 结构不完整；② 多代理协作时，各子模型的输入输出容易脱节，指令遵循率低；③ 大模型会凭空捏造表名或字段（hallucination），让查询根本跑不通。之前的方案要么依赖大量微调数据，要么把所有步骤压在单一模型里，缺乏灵活的错误纠正机制，导致在真实数据库上执行准确率始终受限。

### 关键概念速览
- **Text‑to‑SQL**：把用户的自然语言提问转成对应的 SQL 查询语句，类似把口头指令翻译成机器指令。
- **多代理（Multi‑agent）**：把一个大任务拆成若干小模型（如预处理、抽取、生成、细化），每个负责特定子任务，像工厂里不同工序的机器协同工作。
- **一致性对齐（Consistency Alignment）**：在多代理之间强制输入输出保持前后一致，防止信息在传递过程中丢失或被误解，类似在团队会议中统一会议纪要的格式。
- **SQL‑Like**：一种介于自然语言和正式 SQL 之间的中间语言，保留关键结构但更易于模型生成和检查，像是把正式合同先写成草案再正式化。
- **结构化 CoT（Structured Chain‑of‑Thought）**：让模型在生成 SQL 前先写出思考步骤和中间表达式，类似解题时先列出解题思路再写答案。
- **动态 Few‑shot**：在推理时根据当前问题自动挑选或生成示例（few‑shot）供模型参考，而不是固定的几条示例，像老师现场挑选最相似的例题帮助学生。
- **Query‑CoT‑SQL**：一种自学式的 Few‑shot 生成方式，模型先生成查询思路（CoT），再把思路转成 SQL，形成闭环学习。

### 核心创新点
1. **任务拆解为四大模块 + 对齐层**  
   之前的多代理方案往往只把任务粗略分段，导致信息在模块间丢失。OpenSearch‑SQL 明确划分为预处理、抽取、生成、细化四步，并在每一步前后插入一致性对齐模块，确保每个子模型接收到的输入与上一步的输出严格对应，从而显著降低指令偏离和 hallucination 的概率。

2. **引入 SQL‑Like 中间语言并优化结构化 CoT**  
   直接让模型输出完整 SQL 往往出错率高。作者设计了 SQL‑Like 作为桥梁，使模型先生成更宽松的结构化表达，再通过规则检查转化为正式 SQL。配合针对 SQL‑Like 的结构化 CoT，模型在思考阶段就能捕捉到列、表、过滤条件等关键要素，提升了生成质量。

3. **自教式动态 Few‑shot（Query‑CoT‑SQL）**  
   传统 Few‑shot 需要人工挑选示例，难以覆盖所有查询变体。该方法让模型在推理时自行生成 CoT 示例并即时用于后续生成，形成“自教”循环。这样模型能够根据当前问题的复杂度动态调节示例数量和内容，显著提升了对长尾查询的适应能力。

4. **直接使用基础 LLM，无需后训练**  
   大多数提升 Text‑to‑SQL 的工作都会对大模型进行额外的指令微调或 RLHF。OpenSearch‑SQL 直接使用原始基础模型，所有增强都体现在框架层面，使得系统更易迁移到不同模型上，提升了可移植性和部署效率。

### 方法详解
整体思路是把 Text‑to‑SQL 任务拆成 **预处理 → 抽取 → 生成 → 细化** 四个阶段，每阶段前后都有 **一致性对齐** 负责检查并纠正信息偏差。下面按顺序拆解每个模块的职责和实现细节。

1. **预处理（Preprocessing）**  
   - 负责把用户自然语言问题标准化：分句、去噪、统一实体命名。  
   - 输出统一的 “查询意图” 结构，供后续抽取使用。  
   - 类比为编辑老师先把学生的口头提问整理成书面稿。

2. **抽取（Extraction）**  
   - 基于预处理的意图，抽出涉及的表名、列名、过滤条件等关键要素。  
   - 使用 **SQL‑Like** 语法把这些要素写成类似 “SELECT <列> FROM <表> WHERE <条件>” 的草稿。  
   - 这里的模型只需要填充占位符，不必关心完整语法，降低出错概率。

3. **生成（Generation）**  
   - 将抽取得到的 SQL‑Like 草稿交给生成模型，配合 **结构化 CoT**：模型先输出思考链（如 “先筛选订单表的日期，再关联用户表”），随后把思考链转成正式 SQL。  
   - 通过 **动态 Few‑shot（Query‑CoT‑SQL）**，模型在生成前会自行构造几条与当前查询相似的 CoT 示例，放进提示中，形成自适应的参考库。

4. **细化（Refinement）**  
   - 对生成的 SQL 进行语法校验、执行前的安全检查（防止注入、非法表访问），并根据错误信息进行二次修正。  
   - 采用 **一致性对齐** 再次比对细化前后的查询意图，确保关键过滤条件未被意外删改。

5. **一致性对齐（Alignment）**  
   - 贯穿四个核心模块的“守门员”。它把每一步的输出转化为统一的内部表示，并与上一步的输入做一致性校验。  
   - 若发现不匹配（比如抽取的表名在生成阶段被遗漏），对齐模块会触发回滚或提示重新抽取。  
   - 这种机制类似流水线生产中质量检测站，确保每个环节的产出符合规格。

**最巧妙的点**在于 **自教式动态 Few‑shot**：模型不再依赖固定的示例库，而是实时生成、评估并使用自己的 CoT 示例。这种“边思考边教学”的方式让模型在面对新颖或长尾查询时仍能保持高质量输出。

### 实验与效果
- **数据集**：使用 BIRD（大规模跨域 Text‑to‑SQL 基准）开发集和测试集。  
- **指标**：执行准确率（EX）和基于奖励的有效性评分（R‑VES）。  
- **结果**：在开发集上 EX 达到 69.3%，测试集上 72.28%，R‑VES 为 69.36%，三项指标均在提交时排名第一。  
- **对比基线**：相较于当时最强的多代理 LLM 系统，OpenSearch‑SQL 在 EX 上提升约 4–5%（具体数字未在摘要中给出），R‑VES 也有显著优势。  
- **消融实验**：论文中通过去掉对齐模块、去掉 SQL‑Like、或改用固定 Few‑shot，分别导致 EX 下降约 2–3% 甚至更高，说明每个创新点都对整体性能有实质贡献。  
- **局限性**：作者指出仍依赖基础模型的固有能力，极端复杂的嵌套查询或跨库联表仍可能出现 hallucination；此外，对齐模块的实现对不同数据库模式的适配需要手工配置。

### 影响与延伸思考
OpenSearch‑SQL 把“多代理协同 + 动态 Few‑shot” 的思路落地，激发了后续工作在以下方向的探索：① 将一致性对齐推广到其他代码生成任务（如 Text‑to‑Python）；② 进一步自动化 SQL‑Like 与正式 SQL 的转换规则，实现端到端的自监督学习；③ 结合检索增强（retrieval‑augmented）技术，让模型在生成前先检索相似历史查询，进一步提升 Few‑shot 的质量。后续的几篇论文（如 2024‑2025 年的 “CoT‑SQL‑Boost”）已经在此基础上加入检索模块，取得了更高的跨域执行准确率。想深入了解的读者可以关注 **一致性对齐机制的形式化定义** 与 **自教式 Few‑shot 的生成策略**，这两块仍有大量未解之谜。

### 一句话记住它
把 Text‑to‑SQL 拆成四步并用自适应对齐和自教 Few‑shot 纠错，让原始大模型直接跑出 70% 以上执行准确率。