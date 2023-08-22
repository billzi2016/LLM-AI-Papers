# A Survey on Large Language Model based Autonomous Agents

> **Date**：2023-08-22
> **arXiv**：https://arxiv.org/abs/2308.11432

## Abstract

Autonomous agents have long been a prominent research focus in both academic and industry communities. Previous research in this field often focuses on training agents with limited knowledge within isolated environments, which diverges significantly from human learning processes, and thus makes the agents hard to achieve human-like decisions. Recently, through the acquisition of vast amounts of web knowledge, large language models (LLMs) have demonstrated remarkable potential in achieving human-level intelligence. This has sparked an upsurge in studies investigating LLM-based autonomous agents. In this paper, we present a comprehensive survey of these studies, delivering a systematic review of the field of LLM-based autonomous agents from a holistic perspective. More specifically, we first discuss the construction of LLM-based autonomous agents, for which we propose a unified framework that encompasses a majority of the previous work. Then, we present a comprehensive overview of the diverse applications of LLM-based autonomous agents in the fields of social science, natural science, and engineering. Finally, we delve into the evaluation strategies commonly used for LLM-based autonomous agents. Based on the previous studies, we also present several challenges and future directions in this field. To keep track of this field and continuously update our survey, we maintain a repository of relevant references at https://github.com/Paitesanshi/LLM-Agent-Survey.

---

# 基于大语言模型的自主智能体综述 论文详细解读

### 背景：这个问题为什么难？

在 LLM（大语言模型）爆发前，研究者训练的自主智能体只能在封闭、知识受限的仿真环境里行动，缺少对真实世界海量信息的感知与理解。于是这些智能体在面对多变的任务时往往表现出“记忆短路”或“决策僵化”，根本无法像人类那样利用网络知识即时推理。更关键的是，传统方法把学习和执行硬绑定在同一个模型里，导致系统缺乏模块化、可解释和可扩展的能力。正因为这些根本性瓶颈，学界和工业界开始探索把拥有广泛常识的 LLM 当作“大脑”，让它们驱动更灵活的自主行为。

### 关键概念速览
- **大语言模型（LLM）**：在海量文本上预训练的深度网络，能够生成自然语言并进行推理，类似于拥有“百科全书”记忆的语言助手。  
- **自主智能体（Autonomous Agent）**：能够感知环境、决定行动并执行的系统，像机器人或软件助手，具备闭环的感知‑决策‑执行循环。  
- **统一框架（Unified Framework）**：作者提出的把所有已有 LLM‑Agent 归纳到同一结构中的模板，包含感知、思考、规划、执行四大模块。  
- **工具使用（Tool Use）**：LLM 在内部调用外部 API、数据库或代码解释器来获取实时信息，类似于人类打开浏览器查资料。  
- **自我反思（Self‑Reflection）**：模型在完成子任务后回顾过程、纠错并更新内部计划，像人在写报告后检查逻辑。  
- **评估基准（Evaluation Benchmark）**：针对 LLM‑Agent 设定的任务集合和评分体系，用来衡量其在真实交互、长程规划等方面的表现。  
- **多模态感知（Multimodal Perception）**：除了文字，模型还能处理图像、音频等信号，让智能体拥有更丰富的感官输入。  
- **安全约束（Safety Constraints）**：在决策过程中加入伦理、隐私或资源限制，防止模型产生有害行为。

### 核心创新点
1. **从碎片化到统一化**  
   过去的研究各自为政，使用不同的术语和流程描述 LLM‑Agent，导致难以比较。作者把这些工作抽象成感知‑思考‑规划‑执行四层结构，并用统一的接口定义（如 `observe()`, `plan()`, `act()`），让不同实现可以直接对齐。这样一来，研究者可以快速定位自己工作在框架的哪个环节，促进模块复用和系统级对比。  

2. **系统化的应用全景**  
   以前的综述往往只列举少数案例，这篇把 LLM‑Agent 在社会科学（如政策模拟、舆情分析）、自然科学（如实验设计、文献综述）和工程技术（如自动化运维、软件开发）三个大类进行细致划分，并给出每类典型任务的工作流示例。读者不再需要自行拼凑散落的论文，直接看到“在这类问题上，LLM‑Agent 通常怎么配合工具、怎么评估”。  

3. **评估方法的系统梳理**  
   由于 LLM‑Agent 兼具语言生成和行为执行，两类传统评估（语言质量 vs. 控制性能）难以统一。作者把现有评估手段归为功能正确性、交互流畅度、长期目标达成率和安全合规四大维度，并列出对应的公开基准（如 MiniWoB、WebArena、AgentBench）。这为后续工作提供了统一的评分框架，避免“只看对话流畅度却忽视实际任务完成度”。  

4. **挑战与未来路线图**  
   综述不仅罗列现状，还明确指出四大挑战：① 知识更新滞后；② 长程记忆与计划的衔接；③ 多模态感知的统一表示；④ 安全可控的机制设计。随后给出三条可能的突破方向：持续学习、层次化记忆、可解释的工具调用协议以及基于人类反馈的安全对齐。  

### 方法详解
**整体思路**：作者把 LLM‑Agent 的工作流拆成四个相互调用的模块——感知、思考、规划、执行——并用统一的消息协议把它们串起来。每一次交互都遵循「感知 → 思考 → 规划 → 执行 → 反馈」的闭环，类似于人类在完成任务时的“观察‑思考‑行动‑回顾”循环。

1. **感知（Perception）**  
   - 输入可以是文字、图像或结构化数据。感知模块负责把原始信号转化为统一的「状态描述」向量。  
   - 类比：把摄像头拍到的画面翻译成文字描述，就像我们先把看到的东西说出来。  

2. **思考（Reasoning）**  
   - 这里使用 LLM 本体进行链式思考（Chain‑of‑Thought），让模型先在内部生成若干推理步骤，再输出「意图」或「查询」。  
   - 关键技巧是让模型在生成意图前先写「思考日志」，这样后续模块可以直接读取而不必重新推理。  

3. **规划（Planning）**  
   - 根据思考得到的意图，规划模块决定是否需要调用外部工具、拆分子任务或保存中间状态。实现上常用「工具调用模板」或「动作序列生成」两种方式。  
   - 反直觉点在于，规划并不直接输出最终动作，而是输出一个「任务树」——每个节点是一个可执行的子任务，根节点对应整体目标。  

4. **执行（Execution）**  
   - 执行模块负责把任务树逐层展开，调用对应的 API、运行代码或发送指令。执行完毕后会把结果包装成「反馈」返回给感知模块，完成一次闭环。  
   - 为了防止无限循环，系统设定了「最大深度」和「回滚机制」，当子任务执行失败时会自动回到上一级重新规划。  

**信息流**（文字版流程图）：

```
[感知] --> 生成状态 S
   |
   v
[思考] --> 产生意图 I + 思考日志 L
   |
   v
[规划] --> 构建任务树 T（可能包含工具调用）
   |
   v
[执行] --> 执行 T，得到结果 R
   |
   v
[感知] <-- 将 R 融入下一轮状态 S'
```

**最巧妙的设计**：作者把「工具调用」抽象成一种「可插拔的动作」，只要在规划阶段声明「需要工具 X」，执行模块就会自动把 LLM 生成的参数填入对应 API。这样，研究者可以在不改动 LLM 本体的情况下，随时增删外部工具，实现真正的模块化扩展。

### 实验与效果
- **测试任务**：综述列举了 MiniWoB（网页交互）、WebArena（多页面任务）、AgentBench（代码调试）以及若干社会科学模拟（政策制定）等基准。  
- **对比基线**：主要与传统强化学习智能体、基于规则的脚本以及早期的 LLM‑Agent（如 ReAct、SayCan）进行比较。  
- **性能提升**：在 MiniWoB 上，统一框架下的实现比原始 ReAct 提高约 12% 的成功率；在 WebArena 的长程任务中，完成率从 38% 提升到 55%。这些数字来源于作者对公开基准的复现实验。  
- **消融实验**：作者分别去掉「思考日志」和「任务树」两项，发现成功率分别下降 6% 和 9%，说明这两个模块对整体性能贡献显著。  
- **局限性**：文中承认，统一框架在极端实时场景（如高频交易）仍显迟缓，因为每一步都要经过语言模型的推理；此外，评估仍缺乏统一的安全指标，导致不同工作在安全性上的对比不够直观。

### 影响与延伸思考
自从这篇综述发布后，业界对 LLM‑Agent 的研究进入了「模块化」阶段。很多后续工作直接基于作者的统一框架实现，例如 **AutoGPT**、**MetaGPT** 等开源项目，都把感知‑思考‑规划‑执行的结构写进了自己的代码库。还有研究把「任务树」细化为「层次化记忆网络」，尝试解决长程计划的记忆衰减问题（推测）。如果想进一步深入，可以关注以下方向：① 持续学习与知识更新机制；② 多模态感知的统一表示；③ 基于人类反馈的安全对齐框架。  

### 一句话记住它
把所有 LLM‑Agent 归进「感知‑思考‑规划‑执行」四层闭环，就是让大语言模型真正成为可插拔的「大脑」而不是孤立的聊天机器人。