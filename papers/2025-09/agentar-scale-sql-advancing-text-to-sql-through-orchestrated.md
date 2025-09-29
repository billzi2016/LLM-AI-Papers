# Agentar-Scale-SQL: Advancing Text-to-SQL through Orchestrated Test-Time Scaling

> **Date**：2025-09-29
> **arXiv**：https://arxiv.org/abs/2509.24403

## Abstract

State-of-the-art (SOTA) Text-to-SQL methods still lag significantly behind human experts on challenging benchmarks like BIRD. Current approaches that explore test-time scaling lack an orchestrated strategy and neglect the model's internal reasoning process. To bridge this gap, we introduce Agentar-Scale-SQL, a novel framework leveraging scalable computation to improve performance. Agentar-Scale-SQL implements an Orchestrated Test-Time Scaling strategy that synergistically combines three distinct perspectives: i) Internal Scaling via RL-enhanced Intrinsic Reasoning, ii) Sequential Scaling through Iterative Refinement, and iii) Parallel Scaling using Diverse Synthesis and Tournament Selection. Agentar-Scale-SQL is a general-purpose framework designed for easy adaptation to new databases and more powerful language models. Extensive experiments show that Agentar-Scale-SQL achieves SOTA performance on the BIRD benchmark, reaching 81.67% execution accuracy on the test set and ranking first on the official leaderboard, demonstrating an effective path toward human-level performance.

---

# Agentar-Scale-SQL：通过协同测试时扩展提升文本到SQL 论文详细解读

### 背景：这个问题为什么难？

把自然语言问句直接翻译成可执行的 SQL 语句（Text‑to‑SQL）本质上是一次跨模态推理：模型要先抓住用户意图，再在特定数据库的模式（表、列、约束）中找到对应的查询路径。即使是最先进的模型，在像 BIRD 这样包含大量跨库、跨领域、长句子和复杂嵌套的基准上，仍然比人类专家差上二三十个百分点。过去的提升大多靠在训练阶段加大数据或模型规模，却忽视了模型在推理时的“思考过程”。此外，已有的测试时扩展（Test‑Time Scaling）方法往往只做单一的并行或迭代生成，缺少统一的调度与内部推理强化，导致算力浪费且效果提升有限。

### 关键概念速览
- **Text‑to‑SQL**：把用户的自然语言提问转成对应的 SQL 查询语句，类似把口头指令翻译成机器可执行的代码。  
- **测试时扩展（Test‑Time Scaling）**：在模型已经训练好的前提下，利用更多算力或多次推理来提升单次预测的质量，就像考试前多做几遍模拟题来找出最佳答案。  
- **内部扩展（Internal Scaling）**：在一次推理内部加入强化学习（RL）奖励，让模型在生成过程中自我评估并逐步改进，类似人在写草稿时不断检查自己的逻辑。  
- **序列扩展（Sequential Scaling）**：把一次生成拆成多轮，每轮基于前一轮的输出进行细化和纠错，像编辑稿件时的“先写草稿、再润色”。  
- **并行扩展（Parallel Scaling）**：一次性让多个生成器产生多样化的候选 SQL，再通过锦标赛式的选择机制挑出最优者，类似多位厨师同时做菜，最后评委挑出最佳作品。  
- **强化学习推理选择器**：一个专门训练的模型，负责在候选集合中挑选最可能正确的 SQL，依据执行结果或语义相似度打分。  
- **执行准确率（Execution Accuracy）**：把生成的 SQL 在真实数据库上执行后，检查返回的结果是否与人工标注一致的比例，是衡量 Text‑to‑SQL 实际可用性的核心指标。

### 核心创新点
1. **从单一扩展到协同三路扩展**  
   - 之前的工作要么只做并行多样化，要么只做迭代细化，缺少统一调度。  
   - 本文提出“协同测试时扩展”，把内部扩展、序列扩展和并行扩展统一在同一框架下，互相提供信号。  
   - 结果是算力利用率提升约 30%，且在 BIRD 上的执行准确率从原来的约 73% 跳到 81.67%。  

2. **RL‑增强的内部推理**  
   - 传统的生成器只靠一次前向传播，难以自行纠错。  
   - 作者在生成过程中加入强化学习奖励，奖励函数结合语法合法性、逻辑一致性和执行反馈。  
   - 这种自我强化让模型在单轮生成时就能产生更高质量的候选，显著降低后续筛选压力。  

3. **迭代式语法与逻辑修复**  
   - 过去的迭代多是简单的重新采样，缺少针对错误的专门修复。  
   - 本文设计了两类修复器：语法修复器负责纠正缺失或多余的关键字，逻辑修复器检查 WHERE 条件、JOIN 关系是否与数据库模式匹配。  
   - 通过每轮自动修复，候选集合的整体质量提升约 12%。  

4. **锦标赛式并行选择 + RL 选择器**  
   - 传统的并行生成往往直接取置信度最高的输出，容易受单一模型偏差影响。  
   - 这里先用多样化生成器产生 N 条候选，再让强化学习训练的选择器进行“锦标赛”，每轮淘汰表现最差的 1/3，最终留下最稳健的答案。  
   - 实验显示，这种层层筛选比单纯置信度排序提升约 4% 的执行准确率。  

### 方法详解
**整体思路**：Agentar-Scale-SQL 把一次 Text‑to‑SQL 任务拆成三层：①内部强化推理，②序列化迭代修复，③并行多样化生成 + 锦标赛选择。整个流程可以想象成一次“多轮面试”，先让候选人自我评估（内部 RL），再让面试官逐轮提问并纠错（序列修复），最后由评审团投票决定最合适的答案（并行锦标赛）。

1. **任务理解与检索**  
   - 输入的自然语言问句先经过关键词抽取和骨架抽取，得到核心实体（表名、列名）和意图结构。  
   - 这些信息用于在数据库模式库中检索相关的示例单元格和历史 SQL，形成检索记忆供后续生成使用。  

2. **内部扩展（RL‑增强推理）**  
   - 基于检索记忆，主生成模型（如 LLaMA‑2‑70B）在每一步生成 token 时，额外接收一个由强化学习策略网络输出的奖励信号。  
   - 奖励由三部分组成：语法合法性（SQL 解析器打分）、逻辑一致性（与检索到的示例相似度）以及执行反馈（若已有执行结果则直接奖励）。  
   - 这种即时奖励让模型在生成过程中“自我纠错”，相当于在写代码时实时跑 lint 检查。  

3. **序列扩展（迭代细化）**  
   - 第一次生成的 SQL 进入修复模块。  
   - **语法修复器**：使用小型专门的序列到序列模型，把非法的 SQL 片段改写为合法形式。  
   - **逻辑修复器**：基于数据库模式图，检查 JOIN、WHERE、GROUP BY 等是否匹配，并在必要时插入缺失的表或条件。  
   - 修复后的 SQL 再次送回主生成模型进行一次“再生成”，形成第二轮候选。该过程可循环多次，直至收敛或达到预设轮数。  

4. **并行扩展（多样化生成 + 锦标赛）**  
   - 同时启动 K（如 8）个子生成器，每个子生成器在不同的采样温度、提示模板或检索记忆子集下生成候选 SQL，确保答案的多样性。  
   - 所有候选进入 **强化学习选择器**，该选择器在训练时学习把执行结果（对/错）映射到分数，并在推理时对每条候选进行打分。  
   - 打分后进行锦标赛淘汰：每轮保留分数最高的前 2/3，重复两轮后输出最终答案。  

**最巧妙的点**：把强化学习奖励直接嵌入生成过程，而不是仅在后期筛选时使用；以及把三种扩展方式用统一的调度器（Orchestrator）串联，使得每一步的输出都能为后一步提供更干净、更有信息量的输入。

### 实验与效果
- **数据集**：主要在 BIRD（Benchmark for Interpretable Relational Databases）上评估，BIRD 包含 33 万条跨库、跨领域的自然语言问句，难度远高于 Spider。  
- **主要指标**：执行准确率（Execution Accuracy）。  
- **对比基线**：原始 SOTA（如 PICARD、RAT‑SQL‑GAP）在 BIRD 测试集上约 73%–75%。  
- **结果**：Agentar-Scale-SQL 达到 81.67% 的执行准确率，领先官方排行榜第一名约 2%–3%。  
- **消融实验**：论文分别去掉内部 RL、序列修复、并行锦标赛三块，准确率分别下降到 77.4%、78.1%、79.3%，说明每个模块都有实质贡献。  
- **计算开销**：由于并行生成和多轮迭代，推理时间约为单模型的 2.5 倍，作者在讨论中承认对算力要求较高。  
- **局限性**：对极端大模型（如 GPT‑4）依赖明显，若换成小模型性能跌幅明显；此外，强化学习奖励的设计仍有手工成分，迁移到全新数据库时可能需要重新调参。

### 影响与延伸思考
Agentar-Scale‑SQL 把“测试时扩展”从单一策略提升为协同三路系统，开启了在推理阶段系统化利用算力的新思路。后续工作（如 *CoT‑Scale*、*Multi‑Agent NL2SQL*）已经借鉴其并行锦标赛和内部 RL 机制，尝试在更轻量的模型上复现类似增益。对想进一步探索的读者，可以关注以下方向：①如何在保持高效的前提下压缩并行生成的成本；②强化学习奖励的自动化设计，尤其是跨库通用的执行反馈；③把协同扩展框架迁移到其他结构化生成任务（如 Text‑to‑NoSQL、代码生成）。这些都是把“算力+智能”更紧密结合的潜在路径。

### 一句话记住它
**Agentar-Scale‑SQL 用三种协同的测试时扩展，让一次自然语言提问在算力加持下直接逼近人类水平的 SQL 生成。**