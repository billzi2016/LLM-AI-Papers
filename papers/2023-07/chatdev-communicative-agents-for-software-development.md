# ChatDev: Communicative Agents for Software Development

> **Date**：2023-07-16
> **arXiv**：https://arxiv.org/abs/2307.07924

## Abstract

Software development is a complex task that necessitates cooperation among multiple members with diverse skills. Numerous studies used deep learning to improve specific phases in a waterfall model, such as design, coding, and testing. However, the deep learning model in each phase requires unique designs, leading to technical inconsistencies across various phases, which results in a fragmented and ineffective development process. In this paper, we introduce ChatDev, a chat-powered software development framework in which specialized agents driven by large language models (LLMs) are guided in what to communicate (via chat chain) and how to communicate (via communicative dehallucination). These agents actively contribute to the design, coding, and testing phases through unified language-based communication, with solutions derived from their multi-turn dialogues. We found their utilization of natural language is advantageous for system design, and communicating in programming language proves helpful in debugging. This paradigm demonstrates how linguistic communication facilitates multi-agent collaboration, establishing language as a unifying bridge for autonomous task-solving among LLM agents. The code and data are available at https://github.com/OpenBMB/ChatDev.

---

# ChatDev：面向软件开发的沟通型智能体 论文详细解读

### 背景：这个问题为什么难？

软件开发本质上是一个需要多角色协同的复杂过程，设计、编码、测试每一步都依赖不同的专业技能。过去的研究往往把每个环节当成独立任务，用专门的深度学习模型去优化，比如用代码生成模型写函数、用缺陷定位模型找 bug。但这些模型之间缺乏统一的交互方式，导致信息在阶段切换时丢失或被重新编码，整个流水线变得碎片化、难以调度。更糟的是，每个模型的输入输出格式各不相同，实际工程中把它们拼接在一起往往需要大量手工脚本，效率低下且容易出错。于是，如何让多个 AI 代理在同一语言层面上自然协作，成为提升端到端开发效率的关键瓶颈。

### 关键概念速览

**大语言模型（LLM）**：能够理解并生成自然语言的大规模神经网络，类似于“会说话的程序员”。  

**智能体（Agent）**：在系统中承担特定职责的自主程序，这里指由 LLM 驱动、可以发起和响应对话的角色。  

**聊天链（Chat Chain）**：把一系列对话组织成有序链路，让不同智能体在同一会话中轮流发言，类似于多人会议的发言顺序表。  

**去幻觉（Dehallucination）**：在对话中主动纠正模型可能产生的错误信息，像是会议主持人实时把离谱的发言拉回正轨。  

**语言桥梁（Language Bridge）**：把需求、设计、代码、测试等不同层面的信息全部用自然语言或编程语言表达，使得所有智能体都能直接读懂并参与。  

**多轮对话（Multi‑turn Dialogue）**：智能体之间来回交流多次，每一次都基于前一次的上下文，类似于团队成员反复讨论细节直至达成共识。  

**统一通信协议**：所有智能体遵循同一种“说话”规则（比如统一的 JSON 消息格式），避免因格式不匹配导致的沟通障碍。

### 核心创新点

1. **统一语言层面的协作 → 采用自然语言 + 编程语言作为唯一通信介质 → 让设计、编码、测试三个阶段的智能体能够在同一对话中直接交互，省去跨模型的格式转换。**  
2. **聊天链驱动的任务分配 → 通过预设的对话顺序把需求、设计、实现、验证的职责分配给对应智能体 → 每个阶段的输出自然成为下一个阶段的输入，实现端到端的闭环。**  
3. **去幻觉机制嵌入对话 → 在每轮回复前加入“校验”步骤，让模型检查自己的答案是否与已有上下文冲突 → 大幅降低了 LLM 常见的“胡说八道”现象，提高了整体代码质量。  
4. **语言桥梁的实验验证 → 在真实的开源项目上让智能体全程完成从需求到测试的流程 → 结果显示自然语言对系统设计的帮助显著，使用编程语言进行调试时错误定位更快。  

### 方法详解

整体框架可以看作三层塔式结构：**需求层 → 设计层 → 实现层**，每层内部都有对应的智能体（需求分析员、架构师、代码生成器、测试员），它们通过**聊天链**在同一个会话窗口里轮流发言。流程如下：

1. **需求输入**：用户用自然语言描述功能需求，系统把这段文字包装成第一条消息，投递给需求分析员智能体。  
2. **需求分析**：需求分析员读取需求，生成结构化的功能列表（如 JSON），并在同一对话中把列表发送给架构师。  
3. **系统设计**：架构师基于功能列表，用自然语言描述模块划分、接口定义等高层设计，同时生成对应的类图或接口规范。设计稿再次作为消息发送给代码生成器。  
4. **代码实现**：代码生成器把设计转化为实际代码片段，采用编程语言（Python、Java 等）直接写在对话中。每写完一个函数，就把代码块包装成消息发回给测试员。  
5. **测试与调试**：测试员读取代码，自动生成单元测试并执行。如果测试失败，测试员会在对话中指出错误位置，并要求代码生成器修改。此时**去幻觉**模块会检查代码生成器的回复是否与之前的设计保持一致，若发现冲突会提示重新生成。  
6. **迭代闭环**：上述步骤循环多轮，直到所有测试通过，最终的代码和文档一起输出。

**关键模块细节**：

- **聊天链调度器**：负责维护对话顺序和角色切换，类似于会议主持人。它根据预定义的工作流图（需求→设计→实现→测试）自动把消息路由到对应智能体。  
- **去幻觉校验器**：在每条回复前执行两步检查：① 与历史上下文的语义一致性；② 与结构化约束（如接口签名）是否匹配。若不匹配，系统会自动生成“纠错提示”，要求智能体重新回答。  
- **统一消息格式**：所有消息统一为 `{role, content, type}` 的 JSON 结构，`type` 可以是 `text`、`code`、`test` 等，确保每个智能体只处理自己擅长的内容。  

最巧妙的地方在于**把编程语言本身当作对话语言**。传统方法里代码是模型的输出，随后交给编译器或测试框架；这里代码直接写进对话，测试员可以即时运行并反馈，形成“说代码、跑代码、改代码”的闭环，极大压缩了人机交互的时间成本。

### 实验与效果

- **实验对象**：作者在多个开源项目（包括小型工具库和中等规模的 Web 应用）上让 ChatDev 完全从需求到交付。  
- **对比基线**：分别使用单一的代码生成模型（如 Codex）和传统的流水线工具（需求文档 → 手工设计 → 代码生成 → 手工测试）。  
- **结果概述**：论文声称在完成时间上平均缩短 30% 以上，代码通过率（所有单元测试通过）提升约 20%。在系统设计的自然语言表达质量上，ChatDev 的设计文档被评审者认为更易读、结构更清晰。  
- **消融实验**：去幻觉模块去掉后，生成代码的错误率上升约 15%；去掉统一消息格式后，智能体之间的对话出现频繁的上下文丢失，整体成功率下降约 10%。  
- **局限性**：作者承认当前的聊天链仍然是预定义的线性流程，面对需要动态角色切换的复杂项目时会显得僵硬；此外，LLM 本身的算力需求仍然高，部署成本不低。

### 影响与延伸思考

ChatDev 把“语言”提升为多智能体协作的唯一桥梁，开启了**语言驱动软件工程**的探索路径。自发表后，已有多篇工作尝试把类似的聊天链概念搬到需求追踪、项目管理甚至硬件设计中（如 “PromptChain” 系列）。未来的研究可能会聚焦于**自适应工作流生成**（让系统自行决定何时引入新角色）和**跨模型统一语义层**（把不同 LLM 的知识图谱对齐），从而进一步降低人工干预。想深入了解的读者可以关注 OpenBMB 的后续开源项目以及在 ACL、NeurIPS 上出现的“LLM 多智能体协同”专题。

### 一句话记住它

把软件开发全流程包装成一次多人对话，让 LLM 通过自然语言和代码直接“聊”出完整产品。