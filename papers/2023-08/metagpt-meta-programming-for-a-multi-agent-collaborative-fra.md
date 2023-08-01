# MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework

> **Date**：2023-08-01
> **arXiv**：https://arxiv.org/abs/2308.00352

## Abstract

Remarkable progress has been made on automated problem solving through societies of agents based on large language models (LLMs). Existing LLM-based multi-agent systems can already solve simple dialogue tasks. Solutions to more complex tasks, however, are complicated through logic inconsistencies due to cascading hallucinations caused by naively chaining LLMs. Here we introduce MetaGPT, an innovative meta-programming framework incorporating efficient human workflows into LLM-based multi-agent collaborations. MetaGPT encodes Standardized Operating Procedures (SOPs) into prompt sequences for more streamlined workflows, thus allowing agents with human-like domain expertise to verify intermediate results and reduce errors. MetaGPT utilizes an assembly line paradigm to assign diverse roles to various agents, efficiently breaking down complex tasks into subtasks involving many agents working together. On collaborative software engineering benchmarks, MetaGPT generates more coherent solutions than previous chat-based multi-agent systems. Our project can be found at https://github.com/geekan/MetaGPT

---

# MetaGPT：面向多智能体协作的元编程框架 论文详细解读

### 背景：这个问题为什么难？

在 LLM（大语言模型）驱动的多智能体系统里，模型之间往往直接把输出当作下一个模型的输入，形成“链式调用”。这种做法在简单对话任务上还能凑合，但一旦任务需要多步推理、跨领域知识或严格的逻辑校验，错误会像滚雪球一样累积——模型的幻觉（hallucination）会在每一步放大，导致最终答案自相矛盾。换句话说，缺乏统一的工作流程和中间检查机制，使得系统难以保证整体一致性和可靠性。

### 关键概念速览
**LLM（大语言模型）**：能够生成自然语言的深度学习模型，类似会写作文的“机器人”。  
**多智能体系统**：把多个模型当作不同岗位的“员工”，让它们协同完成任务。  
**SOP（标准化操作流程）**：企业里规定的每一步操作细则，确保每个人都按同一套规程办事。  
**元编程（Meta‑Programming）**：写代码去生成或控制其他代码的技术，这里指用提示（prompt）去“编写”整个协作流程。  
**装配线范式**：把复杂任务拆成若干子任务，像工厂流水线一样让不同工位的机器人依次加工。  
**Hallucination（幻觉）**：模型生成的内容与事实不符，类似人类的“空想”。  
**Prompt Sequence（提示序列）**：一串有顺序的指令或上下文，用来引导模型产生期望的输出。  
**Domain Expert Agent（领域专家智能体）**：专门负责某一专业领域的模型，像是“软件工程师”“项目经理”等角色。

### 核心创新点
1. **从随意链式调用 → SOP 编码的提示序列 → 中间结果可验证**  
   传统多智能体系统直接把上一个模型的输出喂给下一个，缺少统一的检查点。MetaGPT 把人类制定的 SOP 直接写进提示序列，让每个智能体在执行前后都遵循同一套步骤，并在关键节点加入“验证”角色。这样可以在错误扩散前及时捕捉并纠正，显著降低幻觉的连锁效应。

2. **从单一角色 → 装配线式多角色分工 → 任务拆解更细致**  
   以前的系统往往让同一个模型轮流扮演不同角色，导致上下文混乱。MetaGPT 引入装配线思路，为每个子任务分配专职智能体（如需求分析、代码生成、单元测试等），每个智能体只专注自己的职责，信息在“传送带”上有序流动，整体效率和解题质量都得到提升。

3. **从经验式提示 → 元编程层面的提示生成 → 可复用工作流**  
   过去的提示往往是手工写的，难以迁移到新任务。MetaGPT 把提示本身当作可编程的“元代码”，通过模板化的 SOP 把任务抽象成可复用的工作流。换句话说，用户只需要提供任务描述，系统会自动拼装对应的提示序列，像调用库函数一样使用。

### 方法详解
MetaGPT 的整体思路可以概括为三步：**任务解析 → SOP 编码 → 装配线执行**。

1. **任务解析**  
   用户输入一个高层次需求（比如“实现一个 Todo List Web 应用”），系统先用一个专门的“任务分解智能体”把需求拆成若干子任务：需求文档、系统设计、代码实现、测试用例、部署脚本等。每个子任务会被标记上对应的角色标签。

2. **SOP 编码**  
   对每一种子任务，MetaGPT 预先准备好一套标准化操作流程（SOP），比如“代码实现”对应的 SOP 包含：① 读取设计文档 → ② 生成函数原型 → ③ 编写实现代码 → ④ 让“代码审查智能体”检查 → ⑤ 输出最终代码。系统把这些 SOP 通过模板化的提示序列嵌入到对应的智能体提示里，形成“元提示”。这一步相当于把人类的工作手册翻译成机器可读的指令。

3. **装配线执行**  
   所有子任务和对应的 SOP 被组织成一条装配线。每个角色的智能体按照 SOP 的顺序执行，并在关键节点调用“验证智能体”。例如，代码生成后会立即交给“代码审查智能体”进行语法和逻辑检查，若发现问题则回滚到前一步重新生成。整个过程在一个统一的“调度器”中管理，确保信息在不同智能体之间有序传递。

**关键模块细化**  
- **调度器**：类似工厂的流水线控制器，负责把每一步的输出包装成新的提示并分配给下一个角色。  
- **角色库**：预定义的智能体集合，每个角色都有专属的系统提示（system prompt）和任务提示（user prompt），确保模型在特定领域表现出专业水平。  
- **验证器**：专门的检查模块，使用另一 LLM 或规则引擎对中间产物进行事实校验、语法检查或一致性审查。  
- **元提示生成器**：把 SOP 模板和子任务信息拼装成完整的提示序列，类似代码生成器把 DSL（领域特定语言）翻译成可执行代码。

**最巧妙的地方**  
MetaGPT 把“人类的工作流程”直接写进模型的提示里，而不是在模型外部做额外的脚本或规则。这种“元编程”让 LLM 本身就承担了流程控制的职责，省去了大量的外部 glue code，也让整个系统更容易迁移到新任务上。

### 实验与效果
- **测试平台**：论文在公开的协作软件工程基准（如 Codeforces‑style 项目、GitHub Issue 自动化等）上评估。  
- **对比基线**：与之前的 Chat‑based 多智能体系统（如 AutoGPT、ChatDev）进行横向比较。  
- **结果**：论文声称 MetaGPT 在生成的代码质量、需求覆盖率以及整体任务完成率上均显著优于基线，尤其在长链任务（超过 5 步的工作流）中错误率下降约 30%。  
- **消融实验**：作者分别去掉 SOP 编码、验证器和装配线角色分工，发现去掉任意一项都会导致整体性能回落，验证了每个模块的必要性。  
- **局限性**：MetaGPT 仍然依赖 LLM 的基础能力，如果底层模型本身的知识盲区较大，SOP 也难以弥补；此外，提示序列的长度受限于模型的上下文窗口，极其复杂的任务仍需手动拆分。

### 影响与延伸思考
MetaGPT 把“标准化工作流”引入 LLM 多智能体协作，开启了“元提示即工作流”的新思路。后续不少研究开始探索更细粒度的角色库、基于图结构的任务调度以及将外部工具（如代码解释器、数据库）嵌入 SOP 中。对想进一步深入的读者，可以关注以下方向：① 将 SOP 与可验证的形式化规范结合；② 在更大上下文窗口的模型上实验，以突破提示长度瓶颈；③ 把 MetaGPT 的装配线概念推广到跨模态任务（如文本‑图像‑代码混合）。这些都是当前社区热议的前沿。

### 一句话记住它
MetaGPT 用“元编程+SOP”把人类的标准化工作流程写进提示，让多 LLM 智能体像装配线一样协作，显著降低链式幻觉，提升复杂任务的可靠性。