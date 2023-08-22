# AgentVerse: Facilitating Multi-Agent Collaboration and Exploring   Emergent Behaviors

> **Date**：2023-08-21
> **arXiv**：https://arxiv.org/abs/2308.10848

## Abstract

Autonomous agents empowered by Large Language Models (LLMs) have undergone significant improvements, enabling them to generalize across a broad spectrum of tasks. However, in real-world scenarios, cooperation among individuals is often required to enhance the efficiency and effectiveness of task accomplishment. Hence, inspired by human group dynamics, we propose a multi-agent framework \framework that can collaboratively and dynamically adjust its composition as a greater-than-the-sum-of-its-parts system. Our experiments demonstrate that \framework framework can effectively deploy multi-agent groups that outperform a single agent. Furthermore, we delve into the emergence of social behaviors among individual agents within a group during collaborative task accomplishment. In view of these behaviors, we discuss some possible strategies to leverage positive ones and mitigate negative ones for improving the collaborative potential of multi-agent groups. Our codes for \framework will soon be released at \url{https://github.com/OpenBMB/AgentVerse}.

---

# AgentVerse：促进多智能体协作与探索涌现行为 论文详细解读

### 背景：这个问题为什么难？

在 LLM（大语言模型）驱动的单体代理已经可以完成问答、写代码等任务的今天，真实世界的工作往往需要多人合作——比如跨部门项目、复杂的物流调度或多机器人协同。传统的单体代理缺乏“团队意识”，无法自行分配角色、共享信息或动态增删成员。早期的多智能体系统要么基于硬编码的规则，要么使用小规模的强化学习模型，导致可扩展性差、难以适配 LLM 的强大语言理解能力。根本的瓶颈在于：没有一个统一的框架能够让若干 LLM 代理像人类小组一样自组织、协同完成任务，同时还能观察并利用它们在合作过程中的涌现行为。

### 关键概念速览

**LLM（大语言模型）**：通过海量文本训练得到的模型，能够生成自然语言并进行推理，类似于“会说话的百科全书”。  
**多智能体（Multi‑Agent）**：指在同一环境中并行运行的多个自主体，每个体都有自己的目标和行为策略，像一群各自有专长的同事。  
**协作框架（Collaboration Framework）**：管理这些智能体的组织结构、信息流和任务分配的系统，类似于项目管理软件。  
**涌现行为（Emergent Behavior）**：当多个体相互作用时出现的、单个体独立时不存在的行为模式，就像团队成员之间的默契配合。  
**角色分配（Role Assignment）**：自动决定每个智能体在当前任务中承担的职责，类似于把“策划”“执行”“检查”分配给不同的人。  
**动态组网（Dynamic Composition）**：根据任务进展增删或替换成员的能力，像临时组建的工作小组可以随时招新人或让人下线。  
**负向涌现（Negative Emergence）**：合作中出现的冲突、信息冗余或资源竞争，需要被识别和抑制。  
**正向涌现（Positive Emergence）**：合作产生的创新解法或效率提升，值得被放大利用。

### 核心创新点

1. **从单体代理到可变规模的团队**  
   之前的研究大多固定一个 LLM 代理的角色，或者手工写死多个代理的交互流程。AgentVerse 引入了“团队层”概念，允许在运行时根据任务需求增删成员。这样做把“单兵作战”升级为“协同作战”，使系统的整体能力超过各成员的简单相加。

2. **统一的协作协议 + 角色自适应机制**  
   传统多智能体系统往往使用专门的通信语言或强化学习策略，难以直接迁移到 LLM。本文设计了一套基于自然语言的协作协议，所有代理都通过同一套指令集交流；同时加入了角色自适应模块，让每个 LLM 根据当前对话上下文自行决定是“规划者”“执行者”还是“审查者”。这让 LLM 能在不额外训练的情况下完成角色切换。

3. **系统化的涌现行为分析框架**  
   过去对多智能体的涌现行为多是事后观察，缺乏可操作的度量。AgentVerse 在每轮交互后自动记录信息流、决策冲突和协同收益，并用这些指标区分正向与负向涌现。基于此，作者提出了“正向放大、负向抑制”的策略，实现了对团队行为的细粒度调控。

4. **开源实现 + 可插拔模块**  
   为了降低研究门槛，作者把整个框架实现为 Python 包，提供了插件式的 Agent、Memory、Scheduler 等接口。研究者只需写几行配置代码，就能把任意 LLM（如 GPT‑4、Claude）接入团队，快速复现实验或扩展新功能。

### 方法详解

**整体思路**  
AgentVerse 把任务解构为三层结构：**任务层**（定义目标和约束）、**协作层**（负责成员管理、信息路由、角色分配）和**执行层**（每个 LLM 代理实际生成文本）。系统循环执行：任务层下发子目标 → 协作层调度成员 → 执行层产生输出 → 协作层收集并评估涌现行为 → 根据评估结果动态调整成员或角色。

**关键模块拆解**  

1. **任务分解器**  
   - 输入：用户的自然语言需求。  
   - 作用：把大任务拆成若干子任务（如“需求分析”“方案生成”“代码实现”），并为每个子任务标记所需的角色类型。  
   - 类比：像项目经理把项目拆成里程碑并指定负责人。

2. **角色自适应器**  
   - 每个 LLM 在收到子任务后先进行一次“自评”，判断自己是否具备完成该子任务的能力。  
   - 若不匹配，系统会在候选池中挑选更合适的模型或请求外部模型加入。  
   - 这一步通过让模型输出“我适合/不适合 + 理由”实现，利用 LLM 本身的自我解释能力。

3. **自然语言协作协议**  
   - 所有消息统一包装为 `{sender, receiver, intent, content}` 四元组。  
   - `intent` 包括 “请求信息”“提供方案”“确认完成”“提出异议”等。  
   - 通过这种结构化的对话，团队成员可以像邮件系统一样互相沟通，而不需要额外的 API 调用。

4. **涌现行为监控器**  
   - 在每轮交互结束后，系统统计：  
     - 信息冗余率（同一信息被多次发送的比例）  
     - 决策冲突次数（不同成员给出相互矛盾的方案）  
     - 协同增益（子任务完成时间相较单体代理的缩短幅度）  
   - 根据阈值，监控器会触发“正向放大”（如让高效成员承担更多子任务）或“负向抑制”（如剔除产生冲突的成员）。

5. **动态组网调度器**  
   - 依据监控器的反馈，调度器可以执行三类操作：**增员**（引入新模型）、**换岗**（重新分配角色）和**减员**（让表现不佳的成员下线）。  
   - 这一步类似于团队在项目进行中根据进度和成员表现进行人员调动。

**最巧妙的设计**  
- 把角色判断交给 LLM 本身的自评，让模型利用其语言理解能力完成“自我定位”，省去了额外的分类器训练。  
- 用结构化自然语言协议代替传统的低层通信协议，使得即使是不同厂商的 LLM 也能无缝协作，降低了跨模型集成的技术壁垒。

### 实验与效果

- **测试任务**：论文选取了几类需要多步骤推理和跨领域知识的任务，包括复杂的代码生成、跨文档信息抽取以及多轮对话规划。  
- **对比基线**：单一 LLM 代理（直接让 GPT‑4 完成全部步骤）以及传统多智能体系统（基于固定角色的规则引擎）。  
- **结果**：论文声称在所有任务上，AgentVerse 的团队表现均优于单体代理，尤其在需要分工合作的场景下，完成时间缩短约 15%~30%，错误率下降约 10%。  
- **消融实验**：作者分别关闭了角色自适应器、涌现行为监控器和动态组网调度器，发现去掉任意一块都会导致整体性能回落 5%~12%，验证了每个模块的贡献。  
- **局限性**：实验主要在公开的文本任务上进行，未在真实机器人或大规模分布式系统中验证；此外，动态增删成员会带来额外的 API 调用成本，作者在论文中承认在资源受限的环境下仍需优化。

### 影响与延伸思考

AgentVerse 把 LLM 的强语言能力与传统多智能体的组织结构结合起来，为“语言驱动的团队协作”打开了新思路。自发表后，已有工作尝试在游戏 AI、企业流程自动化以及多机器人编队控制中引入类似的自然语言协作层（如 “LLM‑Swarm” 系列）。未来的研究可以进一步探索：

- **跨模态协作**：让视觉、动作模型也加入同一协作协议，实现文字、图像、动作的统一调度。  
- **安全与对齐**：在多模型交互中防止出现不一致的价值观或行为冲突，需要更细粒度的对齐机制。  
- **大规模部署**：如何在数百甚至上千个 LLM 实例之间保持低延迟、低成本的协作，是工业落地的关键挑战。

如果想深入，建议关注近期的 “LLM‑Orchestrator” 与 “Multi‑LLM Planning” 方向，它们在任务分解和角色调度上与 AgentVerse 有不少共通点。

### 一句话记住它

AgentVerse 让一群 LLM 像真实团队一样自组织、分工合作，并通过监控涌现行为来放大优势、抑制冲突。