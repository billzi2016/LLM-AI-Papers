# Context as a Tool: Context Management for Long-Horizon SWE-Agents

> **Date**：2025-12-26
> **arXiv**：https://arxiv.org/abs/2512.22087

## Abstract

Agents based on large language models have recently shown strong potential on real-world software engineering (SWE) tasks that require long-horizon interaction with repository-scale codebases. However, most existing agents rely on append-only context maintenance or passively triggered compression heuristics, which often lead to context explosion, semantic drift, and degraded reasoning in long-running interactions. We propose CAT, a new context management paradigm that elevates context maintenance to a callable tool integrated into the decision-making process of agents. CAT formalizes a structured context workspace consisting of stable task semantics, condensed long-term memory, and high-fidelity short-term interactions, and enables agents to proactively compress historical trajectories into actionable summaries at appropriate milestones. To support context management for SWE-agents, we propose a trajectory-level supervision framework, CAT-GENERATOR, based on an offline data construction pipeline that injects context-management actions into complete interaction trajectories. Using this framework, we train a context-aware model, SWE-Compressor. Experiments on SWE-Bench-Verified demonstrate that SWE-Compressor reaches a 57.6% solved rate and significantly outperforms ReAct-based agents and static compression baselines, while maintaining stable and scalable long-horizon reasoning under a bounded context budget.

---

# 上下文即工具：面向长时程软件工程代理的上下文管理 论文详细解读

### 背景：这个问题为什么难？

在软件工程（SWE）任务里，LLM（大语言模型）代理需要在几千甚至上万行代码的仓库里来回查找、修改、测试，交互往往会持续数百轮。传统的代理把所有对话、检索结果直接拼接进上下文，或者等到上下文快满了才被动触发压缩。这样会导致两大问题：一是上下文体积迅速膨胀，超出模型的 token 限制；二是压缩过程缺乏语义指引，容易把关键信息丢掉，导致后续推理漂移。换句话说，长时程交互的“记忆”既不够精准也不够可控，成为阻碍真实项目级别自动化的瓶颈。

### 关键概念速览

**LLM 代理**：使用大语言模型作为核心决策单元的程序，能够接受指令、调用工具、生成代码等。类似于会写代码的助理机器人。  

**上下文**：模型在一次前向推理时看到的全部文字，包括系统提示、用户指令、历史对话和检索结果。相当于人的短期记忆。  

**压缩（Compression）**：把历史对话或检索信息浓缩成更短的摘要，以节省 token。可以想象成把一本书的章节要点写在一页纸上。  

**可调用工具（Callable Tool）**：在 LLM 推理过程中，模型可以主动选择调用的功能模块。这里把“上下文管理”本身包装成一个工具，就像把“记笔记”变成一个可以随时点开的按钮。  

**长期记忆（Long‑Term Memory）**：经过压缩后保存的历史要点，供后续查询但不直接参与每一步推理。类似于项目文档或会议纪要。  

**短期记忆（Short‑Term Memory）**：最近几轮交互的原始内容，保持高保真度，帮助模型做出细粒度决策。相当于刚刚发生的对话记录。  

**轨迹级监督（Trajectory‑Level Supervision）**：在完整的交互轨迹上标注何时、如何进行上下文管理的动作，供模型学习。类似于给学生演示一次完整的实验过程并标出关键步骤。  

### 核心创新点

1. **把上下文管理提升为可调用工具**  
   之前的系统只能被动压缩，往往在上下文快爆炸时才匆忙删减。本文把“压缩”包装成一个显式工具，模型在每一步决策时可以主动决定是否调用、调用何种压缩策略。这样让记忆管理变成了可规划的行动，而不是被动的副作用。

2. **结构化的上下文工作区**  
   传统方法把所有信息混在一起，难以区分重要性。本文划分出三层：固定区（保存系统提示和核心目标），压缩后的长期记忆，以及保留原貌的短期记忆。相当于把工作台分成“工具箱”“档案柜”“白板”，每层都有明确职责。

3. **轨迹级监督数据管线（CAT‑GENERATOR）**  
   为了让模型学会何时压缩、压缩成什么样，作者构建了离线数据生成流程：先让基线代理完成完整任务，再在关键里程碑插入人工或规则生成的压缩动作，形成标注好的训练轨迹。这样模型在学习时看到的是“思考 → 决策 → 调用压缩 → 继续思考”的完整链路。

4. **专用的上下文感知模型 SWE‑Compressor**  
   在上述数据上训练出一个能够在实际推理时自行调用压缩工具的模型。实验表明，它在保持 4k token 预算的前提下，解决率提升到 57.6%，显著超过只用 ReAct 框架的基线和静态压缩方案。

### 方法详解

整体思路可以拆成四步：  
1) **初始化工作区**：系统提示和用户目标写入固定区；短期记忆为空；长期记忆为空。  
2) **交互循环**：每轮，模型根据当前工作区内容生成行动（如检索、编辑代码）以及是否调用“压缩工具”。  
3) **压缩工具执行**：若模型决定压缩，工具会读取短期记忆和已有长期记忆，生成一个浓缩摘要。摘要被加入长期记忆，同时短期记忆被清空或截断，只保留最新的几条交互。  
4) **迭代更新**：新产生的代码、测试结果等加入短期记忆，循环回到第 2 步。

**关键模块**  
- **决策模块**：基于 ReAct 思路，模型输出“思考+行动”。在输出中加入特殊 token `<CALL_COMPRESS>` 表示调用压缩工具。  
- **压缩工具**：内部实现为一个小型的摘要模型（可复用同源 LLM），接受当前短期记忆和已有长期记忆作为输入，输出结构化的要点列表。要点包括“已完成的子任务”“关键实现细节”“未解决的障碍”。  
- **工作区管理器**：负责把固定区、长期记忆、短期记忆拼接成最终上下文，确保总 token 数不超过预算。它会根据压缩工具的输出自动更新长期记忆的内容。

**最巧妙的设计**  
- 将压缩动作显式化为模型可以主动选择的工具，使得记忆管理成为策略层面的决策，而不是底层实现的硬编码。  
- 采用轨迹级监督，让模型在完整任务轨迹中看到“何时压缩、压缩成什么”这对因果关系，避免了仅靠奖励信号难以学习的稀疏问题。  

### 实验与效果

- **测试平台**：SWE‑Bench‑Verified，这是一个包含真实开源项目、完整编译与测试流程的基准，专门评估长时程软件工程代理的能力。  
- **对比基线**：ReAct‑style 代理（不带压缩工具）、静态压缩方案（在上下文满时统一截断或摘要）以及传统的“只用检索+编辑”流水线。  
- **核心指标**：任务解决率（Solved Rate）。SWE‑Compressor 达到 57.6%，而 ReAct 基线约为 38%，静态压缩约为 44%。在相同 4k token 预算下，SWE‑Compressor 的推理步数更少，错误传播也明显降低。  
- **消融实验**：去掉压缩工具的主动调用（改为被动压缩）后，解决率跌至 49%；仅保留固定区和短期记忆（不使用长期记忆）时，性能下降约 10%。说明三层工作区和主动压缩都是关键因素。  
- **局限性**：论文未在超大模型（如 70B）或多语言项目上做评估；压缩质量仍受摘要模型能力限制，极端复杂的实现细节有时会被过度抽象。

### 影响与延伸思考

这篇工作把“记忆管理”提升为可规划的工具，打开了让 LLM 代理自主管理长期信息的大门。后续有几篇论文尝试把**检索**、**调试**甚至**代码合并**也包装成可调用工具，形成更完整的“工具箱”。如果想进一步探索，可以关注以下方向：  
- **层次化记忆**：在长期记忆上再建索引或向量检索，使得压缩摘要本身可被快速定位。  
- **跨任务记忆共享**：让不同项目的代理共享通用的长期记忆片段，实现知识迁移。  
- **自监督压缩**：利用模型自身的生成能力，在没有人工标注的情况下学习何时压缩、如何压缩。  

### 一句话记住它

把上下文压缩当成模型可以主动调用的“记笔记”工具，让长时程软件工程代理在有限的记忆预算下仍能保持清晰、稳健的推理。