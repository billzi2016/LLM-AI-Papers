# Cooperative SQL Generation for Segmented Databases By Using   Multi-functional LLM Agents

> **Date**：2024-12-08
> **arXiv**：https://arxiv.org/abs/2412.05850

## Abstract

Text-to-SQL task aims to automatically yield SQL queries according to user text questions. To address this problem, we propose a Cooperative SQL Generation framework based on Multi-functional Agents (CSMA) through information interaction among large language model (LLM) based agents who own part of the database schema seperately. Inspired by the collaboration in human teamwork, CSMA consists of three stages: 1) Question-related schema collection, 2) Question-corresponding SQL query generation, and 3) SQL query correctness check. In the first stage, agents analyze their respective schema and communicate with each other to collect the schema information relevant to the question. In the second stage, agents try to generate the corresponding SQL query for the question using the collected information. In the third stage, agents check if the SQL query is created correctly according to their known information. This interaction-based method makes the question-relevant part of database schema from each agent to be used for SQL generation and check. Experiments on the Spider and Bird benckmark demonstrate that CSMA achieves a high performance level comparable to the state-of-the-arts, meanwhile holding the private data in these individual agents.

---

# 基于多功能大语言模型代理的分段数据库协同SQL生成 论文详细解读

### 背景：这个问题为什么难？

Text‑to‑SQL（自然语言转SQL）要求模型在阅读用户的自然语言提问后，自动生成对应的SQL查询。传统方法往往把整个数据库的完整模式（schema）喂给单一的大语言模型（LLM），但在实际企业环境中，数据库往往被切分成多个子库，且每个子库的模式信息可能受到隐私或合规限制，不能一次性全部暴露。于是模型要在信息不完整、跨库关联的情况下仍能写出正确的查询，这在之前的工作里几乎没有得到系统化的解决。缺乏跨子库协同、缺少对私有模式的保护，使得现有方法在分段数据库场景下表现不佳，也难以满足企业对数据安全的要求。

### 关键概念速览
- **Text‑to‑SQL（NL2SQL）**：把自然语言问题转成SQL语句的任务，就像把口头指令翻译成数据库指令一样。
- **Schema（模式）**：数据库的结构描述，包括表名、列名、列类型以及表之间的关联关系。可以把它想成数据库的“地图”。
- **Agent（代理）**：在本文里指的是基于大语言模型的独立实体，每个代理只掌握自己所在子库的模式信息，类似于只会看自己部门文件的员工。
- **Cooperative Generation（协同生成）**：多个代理通过对话互相补充信息，共同完成SQL的生成，类似团队讨论后统一决定行动方案。
- **Segmented Database（分段数据库）**：把一个大数据库拆成若干子库，每个子库可能由不同部门或业务线管理，信息被“切片”存放。
- **Correctness Check（正确性检查）**：生成的SQL在每个代理的已知模式下进行验证，确保没有引用不存在的表或列，类似审稿人检查论文引用是否合理。

### 核心创新点
1. **信息分片 + 多Agent协作 → 只让每个Agent暴露自己子库的模式 → 解决了隐私泄露和规模扩展的问题**。传统做法把完整模式一次性喂给模型，导致私有信息全部公开；本文让每个Agent只负责自己熟悉的那块数据，既保护了隐私，又降低了单次输入的长度。
2. **三阶段交互流程 → ①收集相关模式 ②生成SQL ③跨Agent校验 → 让SQL生成过程像团队的需求分析、方案设计、代码审查一样层层把关**。相比一次性让模型直接输出SQL，这种分步对话让模型有机会纠正误解、补全缺失信息，从而提升准确率。
3. **基于LLM的自监督检查机制 → 每个Agent利用自身已知模式对生成的SQL进行合法性验证 → 把“审计”职责内嵌在生成链路中，而不是事后再跑外部验证工具**。这样可以在生成过程中即时捕获错误，减少后期调试成本。
4. **实验验证在Spider和Bird两大基准上达到与最先进方法相当的水平 → 在保持私有模式不泄露的前提下仍能竞争**。这表明协同Agent的设计并没有因为信息切分而牺牲性能。

### 方法详解
整体框架由 **三大阶段** 组成，整个过程像一次多人会议：先确定议题（问题相关的模式），再共同起草方案（SQL），最后全体审阅（正确性检查）。

1. **阶段一：Question‑related Schema Collection（问题相关模式收集）**  
   - 每个Agent先自行解析自己的子库模式，判断哪些表或列可能与用户提问有关。判断依据包括关键词匹配、语义相似度等。  
   - 判断完毕后，Agent 向其他Agent 发起“我需要哪些信息？”的请求。请求内容是抽象的需求描述（例如“涉及订单金额的表”），而不是直接暴露完整模式。  
   - 接收到请求的Agent 根据自己的模式返回对应的 **模式片段**（表名、列名、外键关系），并用自然语言解释这些片段为何相关。  
   - 所有返回的片段在中心调度器（可以是另一个轻量 LLM）处合并，形成一个 **问题相关的全局模式子图**，供后续生成使用。

2. **阶段二：Question‑corresponding SQL Generation（对应SQL生成）**  
   - 合并后的模式子图被喂给每个Agent，Agent 依据子图和原始自然语言问题，使用 **Chain‑of‑Thought（思维链）** 风格的提示，让模型先列出生成SQL的思路（如“先从表A筛选，再左连接表B”），再一步步写出完整的SQL。  
   - 多个Agent 各自生成一版SQL，随后通过投票或置信度加权的方式选出最有可能正确的版本。这里的投票机制类似团队成员对方案的共识达成。

3. **阶段三：SQL Query Correctness Check（SQL正确性检查）**  
   - 选出的SQL被广播回所有Agent。每个Agent 用自己掌握的模式检查SQL中出现的表、列、连接是否在自己子库中合法。若发现不匹配，Agent 会返回错误提示（比如“表X在我的子库不存在”），并建议修正方向。  
   - 中央调度器收集所有错误提示，若有冲突则触发 **二次生成**：让相关Agent 重新生成或修改SQL，直到所有Agent 都确认无误。  
   - 这种即时校验把传统的后置执行错误捕获提前到生成阶段，显著降低了运行时错误率。

**最巧妙的设计** 在于把 **隐私保护** 与 **信息共享** 用对话的方式平衡：每次只暴露必要的模式片段，而不是完整模式；同时通过多轮交互让模型自行发现缺失信息，避免了人工标注“相关表”的繁琐工作。

### 实验与效果
- **数据集**：作者在两个业界广泛使用的 Text‑to‑SQL 基准上评估：Spider（跨域复杂查询）和 Bird（更注重多表关联和聚合）。  
- **对比基线**：包括传统单模型方法（如 T5‑SQL、ChatGPT‑ZeroShot）以及最近的基于检索的增强模型。  
- **结果**：论文声称在 Spider 上的执行准确率（Exact Match）与当前最先进的单模型方法相差不到 1%（约 78% vs 79%），在 Bird 上的表现同样保持在同等级别。更重要的是，所有实验均在 **不泄露子库完整模式** 的前提下完成。  
- **消融实验**：作者分别去掉（1）模式收集阶段的跨Agent请求、（2）SQL正确性检查环节，发现准确率分别下降约 5% 和 7%，说明两者对整体性能贡献显著。  
- **局限性**：论文承认在 **极度稀疏的模式信息**（即问题涉及的表在多数子库中几乎不存在）时，Agent 之间的对话成本会显著上升，导致响应时间变慢；此外，当前实现依赖于同一 LLM 的多实例，跨模型协同仍未探索。

### 影响与延伸思考
这篇工作把 **多Agent协作** 的思路正式引入 Text‑to‑SQL，打开了“分布式数据库智能查询”的新局面。后续有几篇论文尝试把 **检索增强** 与 **Agent协同** 结合，进一步提升在大规模企业内部知识库上的表现（如 2024 年的 “Retrieval‑augmented Multi‑Agent NL2SQL”）。如果想继续深入，可以关注以下方向：  
- **跨模型协同**：让不同能力的模型（如专门的检索模型、专注推理的模型）分别担任不同 Agent 的角色。  
- **高效对话协议**：设计更轻量的消息格式，降低多Agent交互的时延。  
- **安全审计**：在协同生成过程中加入可验证的零知识证明，确保即使在不可信环境下也能证明查询合法性。  

### 一句话记住它
把分段数据库的私有模式交给多个LLM代理，让它们像团队一样对话、共同写SQL，再相互审查，既保护数据又保持高准确率。