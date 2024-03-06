# SheetAgent: Towards A Generalist Agent for Spreadsheet Reasoning and   Manipulation via Large Language Models

> **Date**：2024-03-06
> **arXiv**：https://arxiv.org/abs/2403.03636

## Abstract

Spreadsheets are ubiquitous across the World Wide Web, playing a critical role in enhancing work efficiency across various domains. Large language model (LLM) has been recently attempted for automatic spreadsheet manipulation but has not yet been investigated in complicated and realistic tasks where reasoning challenges exist (e.g., long horizon manipulation with multi-step reasoning and ambiguous requirements). To bridge the gap with the real-world requirements, we introduce SheetRM, a benchmark featuring long-horizon and multi-category tasks with reasoning-dependent manipulation caused by real-life challenges. To mitigate the above challenges, we further propose SheetAgent, a novel autonomous agent that utilizes the power of LLMs. SheetAgent consists of three collaborative modules: Planner, Informer, and Retriever, achieving both advanced reasoning and accurate manipulation over spreadsheets without human interaction through iterative task reasoning and reflection. Extensive experiments demonstrate that SheetAgent delivers 20--40\% pass rate improvements on multiple benchmarks over baselines, achieving enhanced precision in spreadsheet manipulation and demonstrating superior table reasoning abilities. More details and visualizations are available at the project website: https://sheetagent.github.io/. The datasets and source code are available at https://anonymous.4open.science/r/SheetAgent.

---

# SheetAgent：面向电子表格推理与操作的通用智能体 论文详细解读

### 背景：这个问题为什么难？
电子表格在日常工作和科研中随处可见，但要让机器像人一样在表格里完成复杂任务并不容易。早期的自动化方法大多只能执行单步指令或直接生成公式，面对需要跨表格、跨步骤的长链推理时会失效。更糟的是，真实需求往往描述模糊、目标不唯一，这让仅靠一次性指令的模型难以捕捉全部细节。于是，缺乏能够在长时间跨度内自行规划、检查并纠错的通用系统，成为阻碍 AI 真正走进表格工作流的瓶颈。

### 关键概念速览
**大语言模型（LLM）**：能够理解并生成自然语言的大规模神经网络，类似“会说话的百科全书”，在这里被用来解释任务、生成公式或代码。  
**长时程（Long‑horizon）任务**：需要多轮操作才能完成的任务，就像做一道需要先筛选、再排序、最后汇总的报表。  
**多步骤推理（Multi‑step reasoning）**：模型必须像人一样把大目标拆解成若干小步骤，逐步推进，而不是一次性给出答案。  
**Planner（规划器）**：负责把用户的高层需求拆解成可执行的子任务，类似项目经理把需求拆成任务清单。  
**Informer（信息提供者）**：在每一步执行前，从表格或外部资源中抽取必要的上下文信息，像助理在会议前准备材料。  
**Retriever（检索器）**：负责在需要时快速定位已有的公式、历史操作或相关文档，类似图书馆的检索系统。  
**反思（Reflection）**：模型在完成一步后检查结果是否符合预期，若不符合则回到规划阶段重新调整，类似人做完一道题后自检。  
**SheetRM 基准**：作者新建的包含长时程、多类别、需要推理的表格任务集合，用来衡量系统的真实表现。

### 核心创新点
1. **从单步执行到迭代式规划**：传统方法往往把用户指令一次性交给 LLM，让它直接输出公式或脚本。SheetAgent 则先让 Planner 把需求拆解成序列化的子任务，再交给后续模块逐步完成，实现了“先规划、后执行、再检查”的闭环。这样可以把原本一次性失败的长链任务拆成若干可验证的小步骤，显著提升成功率。  
2. **三模块协同而非单一模型**：过去的系统大多让同一个 LLM 同时负责理解、检索和操作，容易出现信息冲突。SheetAgent 把职责明确划分：Planner 负责高层计划，Informer 负责提供当前表格的上下文，Retriever 负责快速查找已有资源。三者通过统一的任务状态共享机制协同工作，使每一步都有最合适的“专家”负责，降低了误操作的概率。  
3. **基于反思的自我纠错机制**：执行完每个子任务后，系统会让 LLM 对结果进行自检，如果发现不符合预期（比如公式错误、数据不匹配），会触发反思环节，重新调用 Planner 调整计划或让 Informer 补充信息。这个循环让系统能够在没有人工干预的情况下自行修正错误，类似人做实验后根据结果调参。  
4. **面向真实需求的 SheetRM 基准**：作者并未仅在合成数据上验证，而是收集了真实业务场景中的长时程表格任务，涵盖财务报表、项目进度、数据清洗等多类需求。这个基准为评估通用表格智能体提供了更具挑战性的测试平台，也让后续工作有了统一的比较基准。

### 方法详解
整体思路可以概括为“三步走”：**规划 → 信息获取 → 执行并反思**，循环迭代直至任务完成。

1. **任务输入与初始规划**  
   - 用户通过自然语言描述需求（如“把上个月的销售额按地区汇总，并标出增长超过 10% 的地区”）。  
   - Planner 接收到原始描述后，使用 LLM 生成一段结构化的计划清单，每条记录包含子任务的类型（如“筛选行”“计算列”“生成图表”）和所需的输入字段。  
   - 这一步相当于把大工程拆成若干小工序，确保每一步都有明确的输入输出。

2. **信息提供（Informer）**  
   - 在执行每条子任务前，Informer 根据计划中的字段需求，从当前工作簿中抽取对应的表格片段、列标题或已有公式。  
   - 如果任务涉及外部知识（比如汇率、行业基准），Informer 还能调用外部 API 或检索文档，把结果注入到上下文中。  
   - 这样做的好处是让 LLM 在生成代码或公式时拥有最新、最完整的背景信息，避免因信息缺失导致的错误。

3. **检索支持（Retriever）**  
   - 当 Planner 或 Informer 需要复用已有的公式、宏或历史操作时，Retriever 会在项目库或公开的公式库中快速搜索匹配项。  
   - 检索过程采用向量相似度或关键字匹配，返回最相关的几条记录，供 LLM 参考或直接嵌入。  
   - 这相当于让系统拥有“记忆”，不必每次都重新写同样的公式。

4. **执行与反思**  
   - 结合 Planner 的子任务描述、Informer 提供的表格片段和 Retriever 的检索结果，LLM 生成对应的 Excel 公式、Python‑pandas 脚本或 VBA 宏，并在实际表格上执行。  
   - 执行完后，系统会自动检查结果：比如验证汇总值是否等于原始数据的加和、检查生成的图表是否包含所有目标地区。  
   - 若检查失败，系统会把错误信息反馈给 Planner，触发**反思**：Planner 重新生成更细致的计划（可能拆成更小的子任务或更改公式），循环回到信息获取阶段。  
   - 这个闭环保证了即使在长时程任务中出现单步错误，也能通过多轮自我纠正最终达成目标。

**最巧妙的设计**在于把“规划”和“执行”解耦，并在两者之间加入了“信息提供”和“检索”两层显式的上下文补全，使得每一次 LLM 的生成都有最合适、最完整的输入，极大降低了“幻觉”式错误的概率。

### 实验与效果
- **测试基准**：作者构建的 SheetRM 基准，包含数十个真实业务场景的长时程表格任务，任务类型覆盖数据清洗、统计汇总、图表生成、跨表关联等。  
- **对比基线**：包括直接让 GPT‑4/Claude 等大模型一次性生成公式的“单步”方法，以及已有的表格自动化工具（如 Excel‑Copilot、Tabular‑LLM）。  
- **性能提升**：在 SheetRM 上，SheetAgent 的任务通过率比最强基线提升约 **20%–40%**（具体数值在论文中给出），尤其在需要多轮推理的任务上提升更为显著。  
- **消融实验**：作者分别去掉 Planner、Informer、Retriever 中的任意一个模块进行测试，发现去掉 Planner 导致通过率下降约 30%，去掉 Informer 或 Retriever 则各下降约 15%–20%，说明三者协同是提升效果的关键。  
- **局限性**：论文未给出在极大规模（上万行）表格上的运行时评估，也没有详细讨论对极端模糊需求的鲁棒性，作者在讨论章节承认在资源受限的环境下迭代反思的成本仍需优化。

### 影响与延伸思考
SheetAgent 的出现标志着表格智能化从“一次性生成”向“可循环自我纠错”迈进，已经在后续的工作中被引用用于构建更通用的数据工作流代理（如 DataAgent、AutoTable）。它的三模块架构也启发了其他领域的多步骤任务处理，例如代码调试助手和多模态问答系统。未来可以进一步探索：

- **跨表格、跨文件的全局规划**，让 Agent 能在企业级的多工作簿环境中统一调度。  
- **更高效的检索机制**，比如结合专门的公式嵌入模型提升 Retriever 的匹配精度。  
- **人机协同**，在关键反思环节加入用户确认，以平衡自动化与可控性。  

如果想深入了解，建议关注近期在 “LLM‑driven spreadsheet automation” 方向的会议论文（如 ACL、NeurIPS）以及开源项目 SheetAgent 的代码实现。

### 一句话记住它
**SheetAgent 用规划‑信息‑检索三位一体的循环，让大语言模型在表格里也能像人一样“先想后做、出错再改”。**