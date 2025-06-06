# Reinforcing Code Generation: Improving Text-to-SQL with Execution-Based Learning

> **Date**：2025-06-06
> **arXiv**：https://arxiv.org/abs/2506.06093

## Abstract

In this work, we study the problem of code generation with a large language model (LLM), with a focus on generating SQL queries from natural language questions. We ask: Instead of using supervised fine tuning with text-code pairs, can we tune a model by having it interact with a database engine? We frame this problem as a reinforcement learning problem where the model receives execution-based feedback from the environment in the form of scalar rewards. These rewards penalize execution failures and assign positive values when a query returns a correct answer. We use the rewards within the Group Relative Policy Optimization (GRPO) framework. We use a tabular reasoning benchmark to test and evaluate our findings. We find that with only weak supervision in the form of question-answer pairs, RL-tuning improves the accuracy of model generated SQL code from 31.49 to 49.83 while reducing error percentage from 25.43% to 14.71%. This improvement allowed the model nearly match the performance performance to the larger SQLCoder-70B model. Our work demonstrates the potential of using execution-based feedback to improve symbolic reasoning capabilities of LLMs.

---

# 强化代码生成：基于执行反馈的 Text-to-SQL 论文详细解读

### 背景：这个问题为什么难？
自然语言转 SQL（NL2SQL）本质上是让模型把人类的问句翻译成可以在数据库上直接运行的代码。传统做法是收集大量「问句‑SQL」对，做监督微调。但真实业务里往往只有问句和答案，缺少精准的代码标注；而且即使有标注，模型仍会生成语法错误或逻辑不符的查询，导致执行失败。更根本的瓶颈是：语言模型只学到“看起来像 SQL”，却没有机会在真实数据库环境中感受自己的代码是否真的能得到正确答案。

### 关键概念速览
**大语言模型（LLM）**：能够生成自然语言和代码的深度学习模型，类似会写作文的机器人。  
**Text-to-SQL**：把自然语言问题翻译成结构化查询语言（SQL）的任务，就像把口头指令转成数据库指令。  
**强化学习（RL）**：让模型通过与环境交互获得奖励或惩罚，逐步学会更好的行为策略，类似训练狗狗通过奖励学会坐下。  
**执行反馈**：模型把生成的 SQL 交给数据库执行后得到的结果，用来判断对错的信号。  
**标量奖励**：一个数值（正或负）表示行为好坏，正数鼓励，负数惩罚。  
**Group Relative Policy Optimization（GRPO）**：一种强化学习算法，专门用来在大模型上做安全、稳定的策略更新，像是给模型的“学习速率”加了刹车。  
**弱监督**：只提供问句‑答案对，而不提供完整的代码标签，模型需要自己推断出中间的代码。  

### 核心创新点
1. **从文本‑代码对到执行反馈的训练范式转变**  
   之前的做法是直接用「问句‑SQL」对做监督微调 → 这篇论文改为让模型生成 SQL 后交给数据库执行，依据是否得到正确答案给出奖励 → 这样模型在训练时就能感知自己的代码是否真的可用，显著提升了生成质量。

2. **使用标量奖励统一处理多种错误**  
   传统 RL 在 NL2SQL 场景会设计多个离散奖励（语法正确、部分匹配、完全正确） → 本文把所有错误统一映射为负奖励，把成功返回正确答案映射为正奖励 → 简化了奖励设计，同时让模型更直接地优化“能否得到正确答案”这一最终目标。

3. **在大模型上引入 GRPO 进行安全的策略更新**  
   直接用常规的强化学习算法会导致模型崩溃或产生不可控的代码 → 采用 Group Relative Policy Optimization，先把模型的策略划分为若干组，再对每组做相对优化 → 既保留了原有语言能力，又让 RL 更新更稳健。

4. **仅用弱监督就逼近大模型的性能**  
   以前要想接近 70B 参数的专门 SQL 生成模型，需要大量高质量的代码标注 → 这篇工作只用了问句‑答案对，通过执行反馈的 RL 调优，把 31.49% 的准确率提升到 49.83%，错误率从 25.43% 降到 14.71%，几乎追平了更大模型的表现。

### 方法详解
整体思路可以拆成三步：  
1) **预训练+监督微调**：先用公开的 Text-to-SQL 数据集（如 Spider）对大语言模型做常规的监督学习，让模型掌握基本的 SQL 语法和常见模式。  
2) **生成‑执行循环**：在每一次训练迭代中，模型根据自然语言问题采样生成一条 SQL。随后把这条 SQL 发送给真实或模拟的数据库引擎执行。执行结果要么返回查询答案，要么抛出错误（语法错误、运行时错误、空结果等）。  
3) **奖励计算与 GRPO 更新**：根据执行结果计算标量奖励——如果返回的答案与提供的正确答案完全匹配，奖励为 +1；如果执行失败或答案不匹配，奖励为 -1（或根据错误类型给出不同负值）。这些奖励随后喂给 GRPO，GRPO 会把同一批次中表现好的样本视为基准，对表现差的样本进行相对梯度调整，从而提升整体策略。

**关键模块的类比**：  
- 生成‑执行循环类似于人写代码后立即在 IDE 中运行，看是否通过单元测试。  
- GRPO 的“组相对”机制可以想象成教练先把学生分成若干小组，比较每组内部的进步，再决定该给哪组多少额外训练。

**公式的白话解释**：  
GRPO 的核心是计算每个样本的“优势”（advantage），即实际奖励减去该组的平均奖励。优势为正的样本会得到更大的梯度提升，优势为负的则被抑制。这样模型不会因为少数异常高奖励而整体跳跃，也不会因为整体奖励偏低而失去学习动力。

**最巧妙的设计**：  
把执行错误直接映射为负奖励，而不是先做语法检查再给奖励，省去了额外的解析步骤，同时让模型学会“先跑通”再追求答案正确性。再者，GRPO 的相对更新策略在大模型上极大降低了梯度方差，使得 RL 训练在数十亿参数的模型上仍然可控。

### 实验与效果
- **数据集**：作者使用了一个表格推理基准（tabular reasoning benchmark），该基准提供自然语言问句和对应的答案，但不提供完整的 SQL。  
- **对比基线**：与仅使用监督微调的模型（准确率 31.49%）以及更大的专门模型 SQLCoder-70B（未给出具体数值，但被视作上限）进行比较。  
- **主要结果**：RL 调优后，模型的 SQL 正确率提升到 49.83%，错误率从 25.43% 降至 14.71%。这意味着在同样的弱监督条件下，执行反馈的 RL 能把模型的表现拉近到几乎匹配 70B 大模型的水平。  
- **消融实验**：论文中对奖励设计、GRPO 与普通 PPO（Proximal Policy Optimization）以及是否使用执行反馈做了消融，结果显示：去掉负奖励或换成传统 PPO，性能会回落约 5–7%。这说明奖励统一化和 GRPO 是提升的关键因素。  
- **局限性**：实验仅在单一基准上验证，未在更大规模的真实业务数据库上做评估；奖励函数仍然是二元的（对/错），对部分正确但不完整的查询缺乏细粒度指导；执行过程需要真实数据库或高保真模拟器，增加了训练成本。

### 影响与延伸思考
这篇工作向社区展示了“让大模型直接和代码执行环境对话”是一条可行且高效的路径。随后出现的多篇论文开始探索在代码生成、自动化脚本编写甚至软件单元测试中使用执行反馈进行强化学习。一个值得关注的方向是**多阶段奖励**：在执行成功的基础上再加入性能（如查询耗时）或安全（如避免 SQL 注入）等细粒度指标，进一步提升模型的实用性。对想深入的读者，可以关注近期在 OpenAI、DeepMind 等机构发布的“代码执行器（Code Interpreter）”以及“基于环境的语言模型（EvoLM）”系列工作，它们在此思路上做了更丰富的扩展。

### 一句话记住它
让大语言模型在生成 SQL 后直接跑一次数据库，根据成功与否给奖励，就能在只有问答对的情况下把模型的代码能力逼近专门的大模型。