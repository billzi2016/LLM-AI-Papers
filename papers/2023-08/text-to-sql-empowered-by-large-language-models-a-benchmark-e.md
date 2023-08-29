# Text-to-SQL Empowered by Large Language Models: A Benchmark Evaluation

> **Date**：2023-08-29
> **arXiv**：https://arxiv.org/abs/2308.15363

## Abstract

Large language models (LLMs) have emerged as a new paradigm for Text-to-SQL task. However, the absence of a systematical benchmark inhibits the development of designing effective, efficient and economic LLM-based Text-to-SQL solutions. To address this challenge, in this paper, we first conduct a systematical and extensive comparison over existing prompt engineering methods, including question representation, example selection and example organization, and with these experimental results, we elaborate their pros and cons. Based on these findings, we propose a new integrated solution, named DAIL-SQL, which refreshes the Spider leaderboard with 86.6% execution accuracy and sets a new bar. To explore the potential of open-source LLM, we investigate them in various scenarios, and further enhance their performance with supervised fine-tuning. Our explorations highlight open-source LLMs' potential in Text-to-SQL, as well as the advantages and disadvantages of the supervised fine-tuning. Additionally, towards an efficient and economic LLM-based Text-to-SQL solution, we emphasize the token efficiency in prompt engineering and compare the prior studies under this metric. We hope that our work provides a deeper understanding of Text-to-SQL with LLMs, and inspires further investigations and broad applications.

---

# 大语言模型驱动的文本到SQL：基准评估 论文详细解读

### 背景：这个问题为什么难？
把自然语言问题直接翻译成结构化的SQL语句看似简单，却要让模型同时懂业务描述、数据库模式和SQL语法。早期的 Text‑to‑SQL 系统大多基于小型序列到序列模型，需要大量标注数据才能学会跨表连接、子查询等复杂操作，成本高且迁移到新数据库时表现急剧下降。大语言模型（LLM）虽然在零样本推理上表现惊人，但缺乏统一的评估框架，业界不知道到底该怎么写提示（prompt）才能最大化它们的能力，也不知道不同提示技巧之间的 trade‑off。于是出现了“怎么让 LLM 更好写 SQL，却没有系统化的基准”这一瓶颈。

### 关键概念速览
**Text‑to‑SQL**：把用户的自然语言查询转换成对应的 SQL 语句，类似把口头指令翻译成数据库的操作语言。  
**大语言模型（LLM）**：参数量在数十亿以上的生成式模型，能够理解并生成自然语言和代码，例如 GPT‑4、LLaMA。  
**Prompt Engineering（提示工程）**：设计输入文本的技巧，包括如何表述问题、挑选示例、组织示例顺序，像给模型写“使用说明书”。  
**Question Representation（问题表述）**：对原始用户提问进行改写或补全，使其更贴合模型的理解习惯，类似把口语改写成书面语。  
**Example Selection（示例挑选）**：从已有的问‑SQL 对中挑出最相似的几条作为 few‑shot 示例，像老师挑最能说明概念的例题。  
**Example Organization（示例组织）**：决定示例的排列顺序、是否加标签或思考链，类似排版教材章节的先后顺序。  
**Execution Accuracy（执行准确率）**：把模型生成的 SQL 在真实数据库上执行，检查返回结果是否与期望一致，比单纯的字符串匹配更严苛。  
**Spider 数据集**：业界最常用的跨域 Text‑to‑SQL 基准，包含上百个数据库模式和数千条复杂查询。  
**Token Efficiency（令牌效率）**：在提示中使用的 token（模型的基本计数单位）越少，推理成本越低，等价于“写得更简洁却不失效果”。  

### 核心创新点
1. **系统化的 Prompt 基准 → 对现有的三大维度（问题表述、示例挑选、示例组织）进行统一实验 → 揭示每种技巧的收益与成本，提供可复现的评估平台。**  
   之前大家零散报告自己用的技巧，缺少横向对比。作者把所有常见做法放进同一实验框架，直接展示哪种改写或哪种检索策略最能提升执行准确率。

2. **DAIL‑SQL 整合方案 → 将最优的表述、检索和组织策略组合成一个端到端提示模板 → 在 Spider 上实现 86.6% 的执行准确率，刷新排行榜。**  
   不是单纯改进某一步，而是把“最好的改写 + 最相关的 few‑shot + 最合理的排序”打包成统一的 Prompt，形成了目前公开的 SOTA。

3. **开源 LLM 深度评测 + 监督微调 → 在多种开源模型（如 LLaMA‑2、Vicuna）上跑同样的 Prompt，随后用少量标注数据进行有监督微调 → 显示微调能显著弥补开源模型与闭源模型的差距。**  
   这一步验证了即使没有商业 API，研究者也能通过微调把开源模型推向实用水平。

4. **令牌效率视角的对标 → 统计每种 Prompt 方案消耗的 token 数，比较在相同准确率下的成本 → 证明 DAIL‑SQL 在保持高准确率的同时，显著降低了 token 使用。**  
   过去的工作只看准确率，忽视了推理费用。作者把成本纳入评价，让“高效”成为可量化的指标。

### 方法详解
整体思路可以拆成四个阶段：**改写 → 检索 → 排序 → 推理**，形成一个闭环的 Prompt 生成流水线。

1. **问题改写（Question Representation）**  
   - 首先把用户原始提问送入一个轻量的 LLM（或规则模板），让它补全表别名、明确列名、加入数据库模式提示。  
   - 例如 “找出去年销量最高的产品” 会被改写成 “在表 `sales` 中，检索 `product_name`，按照 `sales_amount` 降序取前 1 条”。  
   - 这种改写相当于给模型提供了“上下文背景”，减少歧义。

2. **示例检索（Example Selection）**  
   - 采用基于 schema 相似度的向量检索：把每条已有的问‑SQL 对的自然语言部分和对应的数据库 schema 编码成向量，查询时用改写后的问题向量找最近的 K 条（K 通常为 3~5）。  
   - 这样挑出来的示例在结构上最接近目标数据库，模型可以直接“类比”学习。

3. **示例组织（Example Organization）**  
   - 将检索到的示例按“从易到难”排序：先放最短、最直接的查询，再放包含子查询或多表连接的复杂例子。  
   - 每条示例前加上标签 `Q:` 与 `SQL:`，并在最后加入一句 “Now answer the following question:” 作为提示结束标记。  
   - 这种结构让模型在阅读时形成清晰的映射关系，类似老师先讲基础再进阶。

4. **统一 Prompt 构造（DAIL‑SQL）**  
   - 将改写后的问题、检索到的示例块、数据库 schema 描述（表名、列名、外键）拼接成完整 Prompt。  
   - 为了控制 token 数，作者对 schema 描述做了压缩，只保留关键列和外键信息。  
   - 最后把 Prompt 送入目标 LLM（如 GPT‑4 或开源的 LLaMA‑2），让模型直接生成 SQL。  

5. **后处理**  
   - 生成的 SQL 经过语法检查器，如果出现未闭合的括号或非法列名，会触发一次简短的“自我纠错” Prompt，让模型在原有输出上进行微调。  
   - 这一步虽然不在核心创新里，但提升了执行准确率的鲁棒性。

**最巧妙的点**在于把“改写”和“检索”紧密耦合：改写后的问题本身已经包含了 schema 关键字，检索时自然倾向于匹配同 schema 的示例，从而形成闭环的上下文强化。

### 实验与效果
- **数据集**：主要在 Spider 上评测，覆盖 200+ 数据库模式和 10k+ 复杂查询。  
- **基线对比**：与之前的 SOTA（如 PICARD、ChatGPT‑3.5 few‑shot）相比，DAIL‑SQL 将执行准确率提升到 **86.6%**，比前一名的约 **84%** 提升了近 **2.5%**。  
- **开源模型实验**：在 LLaMA‑2‑13B、Vicuna‑33B 上使用相同 Prompt，原始准确率约 70% 左右。经过作者提供的少量（约 2k 条）标注数据进行监督微调后，准确率提升至 78%~81%，逼近闭源模型的水平。  
- **消融研究**：分别去掉改写、检索或组织模块，准确率分别下降约 3.1%、2.4% 和 1.8%，说明三者缺一不可，且改写贡献最大。  
- **令牌效率**：DAIL‑SQL 的平均 Prompt 长度为 420 tokens，而传统 few‑shot（不压缩 schema）约 620 tokens，在保持相同准确率的情况下，推理成本下降约 **30%**。  
- **局限性**：论文未详细披露微调所用的具体数据来源和训练超参数，实际复现成本可能高于报告值；此外，Prompt 中仍依赖手工设计的 schema 压缩规则，自动化程度有提升空间。

### 影响与延伸思考
DAIL‑SQL 一举刷新了 Spider 排行榜，直接推动了“Prompt 为王”在 Text‑to‑SQL 领域的热潮。随后出现的工作如 **Schema‑Aware Retrieval**、**Chain‑of‑Thought Few‑Shot** 等，都在不同程度上借鉴了本文的系统化评估思路。对想进一步探索的读者，可以关注以下方向：

- **自动化 Prompt 生成**：利用元学习或强化学习让模型自行发现最优的改写/检索/组织策略。  
- **跨数据库迁移**：把在一个数据库上学到的 Prompt 迁移到全新模式，研究 Prompt 的通用性。  
- **更细粒度的成本模型**：把 token 费用、GPU 时间和响应时延一起纳入评价指标，推动“高效”成为主流。  

（以上影响基于公开引用和社区讨论，部分为推测）

### 一句话记住它
**DAIL‑SQL 用最优的改写、检索和组织三招，把大语言模型的 Text‑to‑SQL 能力推到 86.6% 执行准确率的新高度，同时让提示更省钱。**