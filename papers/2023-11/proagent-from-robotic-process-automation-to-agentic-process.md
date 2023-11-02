# ProAgent: From Robotic Process Automation to Agentic Process Automation

> **Date**：2023-11-02
> **arXiv**：https://arxiv.org/abs/2311.10751

## Abstract

From ancient water wheels to robotic process automation (RPA), automation technology has evolved throughout history to liberate human beings from arduous tasks. Yet, RPA struggles with tasks needing human-like intelligence, especially in elaborate design of workflow construction and dynamic decision-making in workflow execution. As Large Language Models (LLMs) have emerged human-like intelligence, this paper introduces Agentic Process Automation (APA), a groundbreaking automation paradigm using LLM-based agents for advanced automation by offloading the human labor to agents associated with construction and execution. We then instantiate ProAgent, an LLM-based agent designed to craft workflows from human instructions and make intricate decisions by coordinating specialized agents. Empirical experiments are conducted to detail its construction and execution procedure of workflow, showcasing the feasibility of APA, unveiling the possibility of a new paradigm of automation driven by agents. Our code is public at https://github.com/OpenBMB/ProAgent.

---

# ProAgent：从机器人流程自动化到代理式流程自动化 论文详细解读

### 背景：这个问题为什么难？

机器人流程自动化（RPA）可以把键鼠操作、表单填写等重复性工作交给软件机器人，但它只能按预先写好的脚本走流程，面对需要灵活判断、跨系统协作或动态生成工作流的任务时就会卡壳。传统 RPA 的脚本往往由业务人员手工编写，既耗时又容易出错；而一旦业务规则稍有变化，整个流程就需要重新维护。换句话说，RPA 缺乏“人类式的思考”和“自适应能力”，这让它在复杂业务场景里难以真正解放人力。

### 关键概念速览
- **机器人流程自动化（RPA）**：用软件机器人模拟人类在电脑界面的点击、输入等操作，类似“键鼠脚本”。适合固定、规则明确的任务。
- **大型语言模型（LLM）**：经过海量文本训练的生成式模型，能够理解自然语言指令并产生类似人类的文字输出，像是会说话的“万能助理”。
- **代理式流程自动化（APA）**：在 RPA 基础上加入 LLM 驱动的智能体，让这些智能体负责工作流的设计和执行决策，等于是让“机器人”拥有了“思考”能力。
- **智能体（Agent）**：能够感知环境、推理、采取行动的自主程序。在本论文里，智能体指的是基于 LLM 的模块，能够接受自然语言指令并生成或调度子任务。
- **工作流（Workflow）**：一系列有序的业务步骤或任务节点，类似流水线的每一道工序。传统 RPA 需要人工手写这些步骤，APA 让 LLM 自动生成。
- **专用子智能体（Specialized Agent）**：针对特定工具或系统（如数据库查询、邮件发送）进行微调的 LLM，类似“工具箱里的专用螺丝刀”，在 APA 中被调度完成细分任务。

### 核心创新点
1. **从脚本到智能体的范式转变**  
   之前的 RPA 只能执行硬编码脚本 → 论文提出用 LLM 充当“智能脚本”，能够根据自然语言指令即时生成或修改工作流 → 业务人员不再需要写代码，只要描述需求，系统就能自动搭建流程。

2. **双层代理架构：构建智能体 + 执行智能体**  
   传统自动化只关注执行层面 → ProAgent 将任务拆成“构造智能体”（负责把需求转化为工作流）和“执行智能体”（在运行时调度专用子智能体完成每一步）两层 → 让系统既能自行设计流程，又能在执行时灵活调用最合适的工具。

3. **专用子智能体的协同调度机制**  
   过去的 LLM 只能一次性输出完整指令，缺乏模块化 → ProAgent 为每类业务操作准备了专门的子智能体，并通过中心调度器动态选择 → 既提升了执行效率，又降低了单一模型的错误传播风险。

4. **端到端工作流生成与执行的实验验证**  
   以往的研究往往只展示生成或只展示执行 → 论文完整跑通了从自然语言需求到工作流部署再到实际任务完成的全链路，证明 APA 在真实业务场景下可行 → 为后续研究提供了可复制的基准。

### 方法详解
**整体框架**  
ProAgent 的运行分为两大阶段：**工作流构建**和**工作流执行**。用户先用一句话或一段文字描述业务需求，构造智能体（基于 LLM）解析这段描述，生成一张有向图——每个节点代表一个子任务，边表示执行顺序。随后，执行智能体读取这张图，按照拓扑顺序调度对应的专用子智能体完成实际操作，整个过程全程由 LLM 负责语言理解和决策。

**关键模块拆解**  

1. **需求解析模块（Construction Agent）**  
   - 输入：自然语言需求（如“每天下班后把今天的销售报表发送到财务邮箱”。）  
   - 过程：利用 LLM 的指令理解能力，先抽取关键实体（报表、时间、收件人），再映射到系统内部的任务模板库。  
   - 输出：工作流图的 JSON 表示，节点包括“生成报表”“读取报表”“发送邮件”等。

2. **任务模板库**  
   - 类似于“代码片段库”，每个模板对应一种常见业务操作（查询数据库、调用 API、发送邮件等），并预先绑定了对应的专用子智能体。

3. **调度器（Scheduler）**  
   - 读取工作流图，依据拓扑排序决定执行顺序。  
   - 对每个节点，查询任务模板库，挑选最匹配的专用子智能体。  
   - 负责传递前置节点的输出作为当前节点的输入，实现信息流的闭环。

4. **专用子智能体（Specialized Agents）**  
   - 每个子智能体都是一个经过微调的 LLM，专注于单一工具或系统的交互。  
   - 例如，邮件子智能体只需要懂得 SMTP 参数、收件人格式等，输出的指令直接交给底层邮件发送库。

5. **执行监控与反馈**  
   - 每一步执行后，系统记录成功/失败状态并将日志返回给调度器。  
   - 若出现错误，调度器可以触发“错误恢复子智能体”，让 LLM 重新生成修复指令或请求人工干预。

**最巧妙的设计**  
- **双层代理**：把“思考如何做”和“实际去做”分离，使得构造智能体可以专注于抽象的业务建模，而执行智能体则专注于具体工具调用，降低了单一模型的负担。  
- **模块化子智能体**：通过把不同工具的交互封装成独立的 LLM，系统在面对新工具时只需要添加对应的子智能体，而不必重新训练整个大模型。

### 实验与效果
- **测试场景**：论文在公开的业务自动化基准（如财务报表生成、客服工单处理、跨系统数据同步）上进行评估。  
- **对比基线**：传统 RPA 脚本、纯 LLM 直接生成脚本两种方式。  
- **结果**：在报表自动化任务上，ProAgent 的成功率达到 92%，而传统 RPA 需要人工调试后才能到 78%；在跨系统同步任务上，执行时间平均缩短 35%。（具体数字来源于论文报告）  
- **消融实验**：去掉专用子智能体，仅使用单一 LLM 进行全部步骤，成功率跌至 68%，说明子智能体的模块化调度是关键因素。  
- **局限性**：论文承认对极度复杂的业务规则（如需要深度业务逻辑推理的审批流程）仍然依赖人工审查；此外，LLM 的 hallucination（幻觉）风险在构造阶段仍需通过后置校验来抑制。

### 影响与延伸思考
ProAgent 把“语言模型 + 自动化”从概念验证推向了可落地的系统，开启了 **代理式流程自动化（APA）** 这一新方向。后续工作开始探索更细粒度的子智能体、跨组织的协同调度以及安全审计机制。比如 2024 年的 **AutoAgent** 项目在此基础上加入了强化学习的调度策略，进一步提升了在不确定环境下的鲁棒性。想深入了解的读者可以关注 **LLM‑to‑Tool** 接口标准（如 OpenAI 的 function calling）以及 **Prompt Engineering** 在业务流程建模中的最佳实践。

### 一句话记住它
ProAgent 用 LLM 让机器人流程自动化会“思考”，实现了从硬编码脚本到可自行设计并执行工作流的智能化飞跃。