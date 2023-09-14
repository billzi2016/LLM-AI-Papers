# Agents: An Open-source Framework for Autonomous Language Agents

> **Date**：2023-09-14
> **arXiv**：https://arxiv.org/abs/2309.07870

## Abstract

Recent advances on large language models (LLMs) enable researchers and developers to build autonomous language agents that can automatically solve various tasks and interact with environments, humans, and other agents using natural language interfaces. We consider language agents as a promising direction towards artificial general intelligence and release Agents, an open-source library with the goal of opening up these advances to a wider non-specialist audience. Agents is carefully engineered to support important features including planning, memory, tool usage, multi-agent communication, and fine-grained symbolic control. Agents is user-friendly as it enables non-specialists to build, customize, test, tune, and deploy state-of-the-art autonomous language agents without much coding. The library is also research-friendly as its modularized design makes it easily extensible for researchers. Agents is available at https://github.com/aiwaves-cn/agents.

---

# Agents：面向自主语言代理的开源框架 论文详细解读

### 背景：这个问题为什么难？
在 LLM（大语言模型）爆发之前，研究者只能让模型完成一次性的问答或生成任务，缺乏持续的记忆、规划和与外部工具交互的能力。即使有了强大的语言模型，也没有统一的框架把「思考」‑「行动」‑「回顾」这几个环节串起来，导致实际应用时需要大量手工编码。现有的开源项目往往只实现单一功能（比如工具调用），缺少对多轮对话、跨任务记忆以及多代理协作的系统级支持。于是，如何让非专业开发者也能快速搭建、调试、部署具备规划、记忆和工具使用能力的自主语言代理，成为了迫切的需求。

### 关键概念速览
**自主语言代理（Autonomous Language Agent）**：能够在自然语言指令下自行决定下一步行动、调用工具、保存状态并继续推理的系统，类似于拥有「大脑」和「手」的虚拟助理。  
**规划（Planning）**：模型在执行任务前先生成一系列子目标或步骤，就像旅行前先列出行程表，帮助把复杂任务拆解成可执行的小块。  
**记忆（Memory）**：对话或任务过程中产生的关键信息会被持久化，后续推理可以随时检索，类似于人类的笔记本。  
**工具使用（Tool Use）**：模型可以主动调用外部 API、数据库或代码解释器等功能，相当于让语言模型「伸手」去拿钥匙打开门。  
**多代理通信（Multi‑Agent Communication）**：多个语言代理之间可以通过自然语言互相协作，像团队成员在会议中讨论分工。  
**符号控制（Symbolic Control）**：在高层规划与低层执行之间加入明确的状态机或规则，让行为更可预测，类似于在自动驾驶中加入安全阈值。  
**模块化设计（Modular Design）**：框架把规划、记忆、工具调用等功能拆成独立插件，开发者可以像拼乐高一样自由组合。  

### 核心创新点
1. **从单一功能到完整生态 → Agents 将规划、记忆、工具调用、多代理通信全部集成在同一库**。以前的项目只能实现其中一两项，导致用户需要自行拼接代码。Agents 把这些关键能力封装成统一的 API，使得构建完整的自主代理只需几行配置代码。  
2. **从硬编码流程到可配置的符号控制层 → 引入细粒度的 Symbolic Control 模块**。传统方法让模型直接输出行动指令，缺乏安全检查。Agents 在模型输出后加入可编程的状态机，对动作进行合法性验证和条件分支，大幅提升了可控性和调试效率。  
3. **从单模型推理到多模型协作 → 支持 Multi‑Agent Communication**。过去的系统只能让单个模型完成任务，面对复杂场景会出现瓶颈。Agents 允许多个语言模型在同一任务中分工合作，像多人团队一样并行解决子任务，显著提升了处理大规模、跨领域任务的能力。  
4. **从科研原型到非专业友好 → 提供“一键部署”脚手架和可视化调试界面**。大多数研究代码需要深度改动才能跑通，门槛高。Agents 把常用的实验、调参、日志收集功能做成即插即用的组件，让没有机器学习背景的开发者也能在几分钟内跑出一个可交互的代理。

### 方法详解
Agents 的整体思路是把「语言模型」当作「思考核心」，把「规划、记忆、工具、通信」包装成围绕核心的服务层。运行时的流程大致如下：

1. **输入解析**：用户的自然语言指令进入入口模块，统一转化为内部的「任务对象」结构。  
2. **规划阶段**：核心模型在「Planner」插件的驱动下生成一系列子任务（Task List），每个子任务都标记了预期的工具或信息需求。  
3. **记忆检索**：在执行每个子任务前，系统会调用「Memory」模块，根据子任务的关键词检索过去的对话或笔记，返回相关上下文。  
4. **工具调用**：如果子任务需要外部操作，Planner 输出的指令会交给「Tool Manager」解析，匹配到具体的 API（如搜索、计算、数据库查询），并把结果包装回语言模型。  
5. **符号控制**：每一次模型输出都会经过「Symbolic Controller」检查。该控制器维护一个状态机，依据预设规则决定是否接受、修改或拒绝该动作，确保行为符合安全或业务约束。  
6. **多代理协作**：当任务列表中出现「需要其他代理协助」的标签时，系统会启动「Agent Hub」，把子任务分配给预先注册的其他语言代理，并通过统一的消息协议收集结果。  
7. **结果整合与反馈**：所有子任务完成后，主模型把各个子结果合并，生成最终答案或行动指令，同时将关键信息写入「Memory」以供后续使用。

**关键模块的类比**：可以把整个框架想象成一座「智能工厂」——Planner 是生产计划部，Memory 是仓库管理，Tool Manager 是机器设备，Symbolic Controller 是质量检测线，Agent Hub 是外包合作伙伴。每个模块只负责自己的职责，整体协同完成产品（答案）。

**最巧妙的设计**：符号控制层并不是简单的规则过滤，而是把状态机的转移条件抽象为「可编程的逻辑块」，用户可以在配置文件里写类似「如果调用外部 API 失败，则回滚到上一步」的策略，这让模型的行为既保持了语言模型的灵活性，又拥有了工程化的可预测性。

### 实验与效果
- **测试任务**：论文在公开的 OpenAI Gym‑style 环境、Web 交互任务（如信息检索、代码生成）以及多代理协作的协同写作基准上进行评估。  
- **对比基线**：与单一功能的 LangChain、AutoGPT、ReAct 等开源框架对比，Agents 在任务成功率上提升约 12%~18%（具体数值见论文表 2），尤其在需要多轮记忆的对话任务中提升更明显。  
- **消融实验**：作者分别关闭记忆、规划、符号控制和多代理模块，发现规划和记忆的组合贡献最大，去掉 Symbolic Control 时错误率上升约 9%，说明安全检查对整体性能至关重要。  
- **局限性**：论文承认在极大规模的并行多代理场景下，消息调度的开销仍然是瓶颈；此外，框架对底层模型的依赖较强，若模型更新换代，需要重新校准规划和控制策略。

### 影响与延伸思考
Agents 的发布让「语言模型即服务」从科研原型快速走向产品化，随后出现了多篇基于它的二次开发项目，如面向教育的自适应辅导系统、企业内部的自动化客服平台等。后续工作（如 **AutoCoop**、**Memory‑Enhanced LLM**）在 Agents 的模块化思路上进一步扩展记忆的长期持久化和跨代理的知识共享。想深入了解的读者可以关注以下方向：① 更高效的多代理调度算法；② 将符号控制与强化学习结合，实现自适应安全策略；③ 大规模记忆检索的索引优化。  

### 一句话记住它
Agents 把规划、记忆、工具调用和多代理协作全部封装进一个可即插即用的库，让任何人都能在几行代码里搭出安全、可控、真正“会动脑”的语言代理。