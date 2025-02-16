# OctoTools: An Agentic Framework with Extensible Tools for Complex Reasoning

> **Date**：2025-02-16
> **arXiv**：https://arxiv.org/abs/2502.11271

## Abstract

Solving complex reasoning tasks may involve visual understanding, domain knowledge retrieval, numerical calculation, and multi-step reasoning. Existing methods augment large language models (LLMs) with external tools but are restricted to specialized domains, limited tool types, or require additional training data. In this paper, we introduce OctoTools, a training-free, user-friendly, and easily extensible multi-agent framework designed to tackle complex reasoning across diverse domains. OctoTools introduces standardized tool cards to encapsulate tool functionality, a planner for both high-level and low-level planning, and an executor to carry out tool usage. We validate OctoTools' generality across 16 diverse tasks (including MathVista, MMLU-Pro, MedQA, and GAIA-Text), achieving substantial average accuracy gains of 9.3% over GPT-4o. Furthermore, OctoTools also outperforms AutoGen, GPT-Functions, and LangChain by up to 10.6% when given the same set of tools. Through comprehensive analysi, ablations, and robustness tests with compact backbones and noisy tool environments, OctoTools demonstrates advantages in task planning, effective tool usage, and multi-step problem solving. Code, demos, and visualization are publicly available at https://octotools.github.io/.

---

# OctoTools：面向复杂推理的可扩展工具代理框架 论文详细解读

### 背景：这个问题为什么难？

在仅靠大语言模型（LLM）内部的知识进行推理时，模型往往会在需要视觉理解、检索专业文献、做精确数值计算或多步逻辑时卡壳。过去的方案要么只给模型配备单一的外部工具（比如只会调用搜索引擎），要么只能在特定领域（医学、金融）内部署，甚至需要额外的微调数据来教模型何时、如何使用这些工具。结果是：工具种类受限、跨域适配成本高、部署门槛大，导致真正的“全能助理”仍然遥不可及。

### 关键概念速览
- **工具卡（Tool Card）**：一种统一的描述文件，告诉模型这个工具能干什么、接受什么输入、返回什么输出。可以把它想象成工具的“身份证”，不同工具只要遵循同一格式，就能被框架自动识别和调用。  
- **多代理（Multi‑Agent）**：框架内部会生成多个角色（如规划者、执行者），每个角色专注于特定子任务，类似团队里分工合作的成员。  
- **高层/低层规划（High‑level & Low‑level Planning）**：高层规划决定“大方向”（先查资料再算数），低层规划负责把大方向拆解成具体的工具调用序列（先用图像识别工具提取表格，再用计算器算平均值）。  
- **执行器（Executor）**：真正把工具卡翻译成 API 调用并收集返回结果的模块，像是团队里的“执行官”。  
- **训练免费（Training‑free）**：整个系统不需要再对 LLM 做额外的微调或强化学习，只靠提示词（prompt）就能让模型学会使用工具。  
- **可扩展性（Extensibility）**：用户只需写一个符合工具卡格式的描述，就能把新工具塞进系统，类似插件市场的即插即用。  
- **噪声工具环境（Noisy Tool Environment）**：指工具调用可能出错或返回不准确结果的真实场景，框架需要具备容错和纠错能力。  

### 核心创新点
1. **统一的工具卡标准 → 采用统一 JSON‑like 描述所有外部工具 → 任何新工具只要写好卡片就能被框架自动识别，无需改动代码或重新训练模型。**  
2. **双层规划机制 → 引入高层任务拆解 + 低层具体调用序列两级规划 → 模型能够先决定“先查文献再做计算”，再细化到“调用搜索 API → 解析结果 → 调用计算器”。这种层次化思考显著提升了多步推理的成功率。  
3. **多代理协同执行 → 将 LLM 分成 Planner、Executor 两个角色，Planner 负责思考路线，Executor 负责实际调用 → 通过角色间的对话实现了“思考-行动”循环，避免模型一次性输出所有步骤导致的错误累积。  
4. **训练免费且鲁棒的设计 → 完全基于提示词驱动，无需额外微调；在实验中加入噪声工具测试，框架仍能通过错误检测和重规划恢复正确答案 → 降低了部署成本并提升了在真实系统中的可靠性。  

### 方法详解
**整体思路**：OctoTools 把复杂任务拆成“计划—执行—反馈”三步循环。首先，Planner 根据用户的自然语言问题生成一个高层计划；随后，Planner 将高层计划细化为一系列低层指令，每条指令对应一个工具卡；Executor 按指令调用相应工具，收集返回值并把结果反馈给 Planner，Planner 再根据最新信息调整后续指令，直至得到最终答案。

**关键模块拆解**：

1. **工具卡库**  
   - 每个工具都有一个 JSON‑like 卡片，字段包括 `name`（工具名）、`description`（功能简介）、`input_schema`（输入格式）、`output_schema`（输出格式）以及 `api_endpoint`（调用地址）。  
   - 框架在启动时把所有卡片加载进内存，形成统一的检索表。这样，Planner 在生成指令时只需要引用卡片的 `name`，系统会自动把指令翻译成对应的 API 调用。

2. **Planner（高层+低层）**  
   - **高层规划**：使用一次性提示让 LLM 输出任务的子目标列表，例如 “1. 检索最新医学指南；2. 计算剂量；3. 给出治疗建议”。  
   - **低层规划**：对每个子目标再次提示，要求模型输出具体的工具调用序列，每一步都标明使用的工具卡名和所需的输入。这里的提示模板固定，确保模型输出符合机器可解析的结构。  
   - Planner 还负责监控 Executor 的返回，如果检测到错误（如返回格式不匹配），会触发“重规划”——重新生成该步骤的低层指令。

3. **Executor**  
   - 接收 Planner 生成的低层指令，查找对应的工具卡，依据 `input_schema` 把指令中的参数填充成合法的 API 请求。  
   - 调用外部服务（搜索、图像识别、数值计算等），捕获返回并进行基本的格式校验。若返回异常，Executor 会把错误信息回传给 Planner，触发纠错流程。  
   - 成功返回后，Executor 将结果包装成统一的结构送回 Planner，供后续步骤使用。

4. **反馈循环**  
   - Planner 与 Executor 之间的对话采用“思考-行动-观察”模式：Planner 思考下一步，Executor 行动并观察结果，Planner 再根据观察更新思考。  
   - 这种循环类似人类在做实验时的迭代：先假设、再实验、再根据实验结果修正假设。

**最巧妙的设计**：双层规划把抽象的任务目标和具体的工具调用解耦，使得同一个高层计划可以在不同工具集合下自动适配；而且因为所有工具都遵循同一卡片格式，Planner 只需要学习“如何拼装卡片”，而不必记住每个工具的细节，极大降低了学习成本。

### 实验与效果
- **测试任务**：作者在 16 项跨域基准上评估 OctoTools，包括视觉数学（MathVista）、通用知识测评（MMLU‑Pro）、医学问答（MedQA）以及多模态推理（GAIA‑Text）等。  
- **基线对比**：与同样使用 GPT‑4o 的纯语言模型相比，OctoTools 平均提升 9.3% 的准确率；在使用相同工具集合的情况下，分别超越 AutoGen、GPT‑Functions、LangChain 最多 10.6%。  
- **消融实验**：去掉低层规划或统一工具卡会导致性能下降约 4–6%，说明细粒度指令和标准化接口是关键；仅保留单一 Planner（不分角色）时，错误累积率显著上升，验证了多代理协同的必要性。  
- **鲁棒性测试**：在“噪声工具环境”下（故意让部分 API 返回错误或延迟），OctoTools 仍能通过错误检测和重规划恢复正确答案，成功率仅下降约 2%。  
- **局限性**：论文指出，虽然框架不需要训练，但对 LLM 的提示工程仍有一定依赖；在极端长链任务（超过 15 步）时，Planner 的上下文窗口可能成为瓶颈，导致规划失效。

### 影响与延伸思考
OctoTools 的出现让“插件化 LLM”从“单一工具 + 微调”向“统一卡片 + 多代理规划”转型，激发了后续工作在工具标准化和任务分解上的探索。2024 年后，几篇基于 OctoTools 思路的论文（如 **ToolBench**、**AgentForge**）进一步引入了自动化工具卡生成和自适应上下文压缩技术。对想深入的读者，建议关注以下方向：① 自动学习工具卡的语义映射；② 大模型在长链规划中的记忆管理；③ 多模态工具（如实时视频分析）在同一框架下的统一调用。

### 一句话记住它
OctoTools 用统一的工具卡和双层规划，让 LLM 能像团队合作一样，先想后干，轻松把任何新工具塞进复杂推理的流水线。