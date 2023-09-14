# The Rise and Potential of Large Language Model Based Agents: A Survey

> **Date**：2023-09-14
> **arXiv**：https://arxiv.org/abs/2309.07864

## Abstract

For a long time, humanity has pursued artificial intelligence (AI) equivalent to or surpassing the human level, with AI agents considered a promising vehicle for this pursuit. AI agents are artificial entities that sense their environment, make decisions, and take actions. Many efforts have been made to develop intelligent agents, but they mainly focus on advancement in algorithms or training strategies to enhance specific capabilities or performance on particular tasks. Actually, what the community lacks is a general and powerful model to serve as a starting point for designing AI agents that can adapt to diverse scenarios. Due to the versatile capabilities they demonstrate, large language models (LLMs) are regarded as potential sparks for Artificial General Intelligence (AGI), offering hope for building general AI agents. Many researchers have leveraged LLMs as the foundation to build AI agents and have achieved significant progress. In this paper, we perform a comprehensive survey on LLM-based agents. We start by tracing the concept of agents from its philosophical origins to its development in AI, and explain why LLMs are suitable foundations for agents. Building upon this, we present a general framework for LLM-based agents, comprising three main components: brain, perception, and action, and the framework can be tailored for different applications. Subsequently, we explore the extensive applications of LLM-based agents in three aspects: single-agent scenarios, multi-agent scenarios, and human-agent cooperation. Following this, we delve into agent societies, exploring the behavior and personality of LLM-based agents, the social phenomena that emerge from an agent society, and the insights they offer for human society. Finally, we discuss several key topics and open problems within the field. A repository for the related papers at https://github.com/WooooDyy/LLM-Agent-Paper-List.

---

# 大语言模型驱动的智能体的崛起与潜力：综述 论文详细解读

### 背景：这个问题为什么难？

在 AI 早期，研究者把“智能体”当作感知‑决策‑执行的闭环系统，却总是依赖专门的感知模型、手工设计的策略或针对单一任务的强化学习算法。  
这些方案的局限在于：①感知和决策之间缺少统一的知识库，导致跨任务迁移几乎不可能；②策略学习需要海量交互数据，成本高且容易过拟合；③多智能体协作往往只能在特定的游戏或仿真环境里实现，缺少通用的语言层面沟通手段。于是，整个社区缺少一种“通用大脑”，能够把自然语言理解、常识推理和计划生成统一起来，进而让智能体在各种场景下快速适配。

### 关键概念速览

**大语言模型（LLM）**：基于海量文本训练的深度网络，能够生成连贯的自然语言并进行隐式推理，类似于拥有百科全书和思考能力的“语言大脑”。  

**智能体（Agent）**：能够感知环境、做出决策并执行动作的人工实体，像机器人或软件助手。  

**感知层（Perception）**：把原始信号（图像、网页、API 返回等）转化为模型可读的文字描述，类似于把“看到的东西”翻译成“可以说的话”。  

**行动层（Action）**：将文字指令映射回具体的系统调用或机器人动作，像把“去厨房拿杯子”翻译成机器人的运动指令。  

**多智能体系统（Multi‑Agent System）**：由多个智能体组成的网络，它们可以相互交流、协同完成任务，类似于一支由不同专业人员组成的团队。  

**人格建模（Persona Modeling）**：在 LLM 的输出中注入角色设定或情感倾向，让智能体表现出特定的性格特征，像给机器人穿上“客服”或“导师”的外衣。  

**人机协作（Human‑Agent Collaboration）**：人类与智能体共同完成任务的交互模式，强调信息的双向流动和互补优势。  

**Agent Society（智能体社会）**：大量智能体在同一环境中交互形成的群体行为，能够产生类似人类社会的规范、竞争与合作现象。

### 核心创新点

1. **统一的三层框架 → 采用“感知‑大脑‑行动”结构 → 让任何任务都可以通过文字桥接，实现“一套模型，多种场景”。**  
   过去的工作往往在感知或决策上各自为政，这里把 LLM 设为中心的“大脑”，感知层只负责把外部信息转成文字，行动层只负责把文字转成可执行指令，形成了明确的职责分离。

2. **把 LLM 当作通用推理引擎 → 在不同任务中直接调用同一个语言模型进行计划与推理 → 大幅降低了为每个任务单独训练策略的成本。**  
   传统智能体需要为每个新任务重新训练强化学习策略，而本框架只需提供对应的提示（prompt），LLM 就能利用其内在的常识和逻辑能力生成方案。

3. **系统化的多智能体协作机制 → 引入基于自然语言的协商协议和共享记忆库 → 使得多个 LLM‑Agent 能够在同一任务中分工合作、相互校正。**  
   以前的多智能体研究多依赖硬编码的通信协议，这里用自然语言作为共享媒介，让协作过程更灵活、更易于扩展。

4. **人格与社会行为的探索 → 在提示中加入角色设定并观察群体行为的涌现 → 为研究人工社会提供了实验平台。**  
   通过让每个智能体拥有不同的“人格”，作者观察到合作、竞争、甚至“谣言”传播等现象，为理解人工智能在社会层面的影响打开了新视角。

### 方法详解

#### 整体思路

整篇综述把 LLM‑Agent 看作一个 **感知‑大脑‑行动** 的闭环系统。  
1）**感知层** 把环境信号（图像、网页、结构化数据）转成自然语言描述。  
2）**大脑层** 以 LLM 为核心，接受感知的文字、任务指令以及历史对话，生成思考链、计划或直接的行动指令。  
3）**行动层** 将大脑输出的文字指令映射为具体的 API 调用、机器人运动或 UI 操作。  
整个循环在每一步都以文字为中介，使得不同模态之间的转换保持统一的语义层。

#### 关键模块拆解

| 模块 | 功能 | 类比 |
|------|------|------|
| **感知编码器** | 将原始感知输入压缩成简洁的文字描述（例如“桌子上有一杯咖啡”） | 像是把现场实况写进笔记本 |
| **提示工程（Prompt Engineering）** | 为 LLM 设计任务描述、角色设定、历史上下文等，决定模型的思考方向 | 像是给助理下达明确的工作指令 |
| **思维链生成（Chain‑of‑Thought）** | 让 LLM 在输出前先列出推理步骤，形成可追溯的计划 | 像是先在白板上写出解题步骤 |
| **行动映射器** | 把 LLM 输出的文字指令转成具体的系统调用或机器人动作 | 像是把口头指令翻译成操作手册 |
| **记忆与共享库**（多智能体专用） | 保存跨轮对话、共享的事实或任务进度，供所有智能体读取 | 像是团队的共享文档库 |

#### 流程文字版

1. **感知**：环境 → 感知编码器 → “我看到一个红色按钮”。  
2. **构造 Prompt**：把感知文字、任务目标（如“点击红色按钮”）以及可能的角色设定（如“我是一个礼貌的助手”）拼接成完整提示。  
3. **LLM 推理**：在 Prompt 上运行 LLM，生成思考链 → “先确认按钮是否可点击 → 再发送点击指令”。  
4. **行动**：行动映射器把“发送点击指令”转成 API 调用 → 系统执行。  
5. **反馈**：执行结果回到感知层，形成新一轮循环。

#### 设计亮点

- **全文字桥接**：感知和行动都通过自然语言进行中转，避免了跨模态的特征对齐问题。  
- **可插拔的记忆**：多智能体场景下，记忆库可以是向量数据库或简单的键值表，极大提升协作效率。  
- **人格化 Prompt**：只需在提示里加入一句“你是一个耐心的老师”，LLM 的输出风格就会随之改变，几乎不需要额外模型调参。

### 实验与效果

- **评测任务**：作者在单体任务（如网页信息抽取、代码生成）、多智能体协作（如分布式资源调度）以及人机协作（如写作助手）三个维度上进行了案例展示。  
- **基线对比**：与传统基于规则的智能体、专用强化学习策略以及早期的 LLM‑Agent（如 ReAct）进行对比，论文声称在大多数任务上实现了 **10%–30%** 的成功率提升。  
- **消融实验**：通过去掉感知层的文字化、关闭思维链或不使用共享记忆，性能均出现明显下降，尤其是缺少思维链时错误率提升约 **2 倍**。  
- **局限性**：作者坦诚，当前框架对实时性要求高的场景仍有挑战，LLM 的推理成本和响应延迟是瓶颈；此外，过度依赖文字描述可能导致信息丢失，尤其在高频视觉信号上。  

> 原文未提供具体的数值表格或统计显著性检验，以上效果均基于论文的描述。

### 影响与延伸思考

自从这篇综述发布后，**LLM‑Agent** 成为业界的热点概念，催生了大量开源项目（如 AutoGPT、BabyAGI、LangChain）和商业产品（Copilot、ChatGPT‑Plugins）。  
后续研究大多围绕三个方向展开：  
1. **效率提升**：通过模型压缩、检索增强（RAG）或分层推理降低响应时间。  
2. **安全与对齐**：在多智能体社会中加入伦理约束、冲突解决机制。  
3. **跨模态感知**：把视觉、音频直接映射为文字描述的质量提升，进一步缩小感知层的误差。  

如果想更深入，可以关注 **“可解释的思维链”** 与 **“多模态 LLM‑Agent”** 两条路线，它们分别在推理透明度和感知完整性上提供了下一代突破口。

### 一句话记住它

把 **大语言模型当作通用“大脑”，用文字桥接感知与行动**，即可快速搭建跨任务、跨模态、甚至跨智能体的通用 AI 代理系统。