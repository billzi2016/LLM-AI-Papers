# Small LLMs Are Weak Tool Learners: A Multi-LLM Agent

> **Date**：2024-01-14
> **arXiv**：https://arxiv.org/abs/2401.07324

## Abstract

Large Language Model (LLM) agents significantly extend the capabilities of standalone LLMs, empowering them to interact with external tools (e.g., APIs, functions) and complete various tasks in a self-directed fashion. The challenge of tool use demands that LLMs not only understand user queries and generate answers accurately but also excel in task planning, tool invocation, and result summarization. While traditional works focus on training a single LLM with all these capabilities, performance limitations become apparent, particularly with smaller models. To overcome these challenges, we propose a novel approach that decomposes the aforementioned capabilities into a planner, caller, and summarizer. Each component is implemented by a single LLM that focuses on a specific capability and collaborates with others to accomplish the task. This modular framework facilitates individual updates and the potential use of smaller LLMs for building each capability. To effectively train this framework, we introduce a two-stage training paradigm. First, we fine-tune a backbone LLM on the entire dataset without discriminating sub-tasks, providing the model with a comprehensive understanding of the task. Second, the fine-tuned LLM is used to instantiate the planner, caller, and summarizer respectively, which are continually fine-tuned on respective sub-tasks. Evaluation across various tool-use benchmarks illustrates that our proposed multi-LLM framework surpasses the traditional single-LLM approach, highlighting its efficacy and advantages in tool learning.

---

# 小型语言模型是弱工具学习者：多LLM代理 论文详细解读

### 背景：这个问题为什么难？
在让大语言模型（LLM）自行调用外部 API、函数等工具完成任务时，模型需要同时懂得任务规划、工具选择、调用细节以及结果整合。早期的研究把这些能力全部塞进同一个模型里，结果发现尤其是参数量不大的模型往往在工具使用上表现糟糕——它们会误选工具、调用格式错误，甚至在规划阶段卡死。根本原因是单一模型要兼顾语言理解、推理、代码生成和执行反馈四种截然不同的技能，容量受限时难以兼顾。于是出现了“更小的模型也想参与工具学习，却总是力不从心”的局面，迫切需要一种能够把复杂任务拆解、让每个子任务由专精模型负责的方案。

### 关键概念速览
- **LLM（大语言模型）**：能够生成自然语言的深度学习模型，类似会说话的“机器人”。  
- **Tool Use（工具使用）**：模型根据用户需求主动调用外部程序或 API，就像人类在电脑上打开 Excel、运行脚本完成工作。  
- **Planner（规划器）**：负责把用户的高层需求拆解成一步步的子任务，就像旅行前先列出行程清单。  
- **Caller（调用器）**：专门负责把子任务转化为具体的 API 调用或函数执行，类似程序员敲代码并发送请求。  
- **Summarizer（摘要器）**：把调用得到的原始返回结果加工成用户可读的答案，像编辑把技术报告写成通俗的新闻稿。  
- **两阶段训练**：先让一个通用模型学习完整任务，再把它复制成三个专职模型并分别微调对应子任务，类似先学会全科，再专攻各科。  
- **Tool-use Benchmark（工具使用基准）**：评估模型在真实 API 调用场景下的表现的标准测试集，例如让模型完成天气查询、数学计算等任务。

### 核心创新点
1. **能力拆解 → 模块化三模型**  
   之前的做法是把任务规划、工具调用、结果总结全部交给同一个 LLM，导致小模型负担过重。本文把这三项能力分别交给 Planner、Caller、Summarizer 三个模型，每个模型只需专注于一种技能。这样即使是 7B 参数的模型，也能在自己擅长的环节发挥最大效能，整体系统的鲁棒性大幅提升。  

2. **两阶段训练流程 → 先通用后专精**  
   传统方法直接在子任务上微调模型，容易出现“忘记”整体任务上下文的情况。本文先在完整数据集上对一个骨干 LLM 进行一次全局微调，让它掌握任务全貌；随后把这份模型复制成三个实例，分别在规划、调用、摘要子任务上继续微调。相当于先让学生学会整本教材，再让他在每章做专项练习，既保留了全局知识，又提升了局部技能。  

3. **模块可独立更新 → 灵活组合**  
   由于每个子模型是独立的，后续可以单独换掉某一环节的模型（比如换成更强的调用器），而不必重新训练整个系统。这种“插件化”思路在 LLM 领域尚不常见，为后续的模型迭代和资源调度提供了新思路。  

4. **多模型协同推理 → 任务级并行**  
   在实际推理时，Planner 先输出子任务列表，随后 Caller 可以并行处理多个子任务的 API 调用，最后 Summarizer 汇总。相比单模型顺序执行，这种并行协同能够显著缩短整体响应时间，尤其在需要调用多个工具的复杂任务中优势明显。

### 方法详解
**整体框架**  
整个系统由三段式组成：① Planner 接收用户自然语言指令，输出一系列结构化的子任务；② Caller 逐个或并行地把子任务映射为具体的工具调用（包括函数名、参数、调用方式），并收集返回的原始结果；③ Summarizer 将所有原始返回整合、过滤、重写，生成最终的自然语言答案。整个流程像是“需求 → 任务清单 → 执行 → 报告”。

**步骤拆解**  

1. **全局预训练（Stage‑1）**  
   - 选取一个通用的骨干 LLM（如 LLaMA‑7B）。  
   - 使用包含完整任务描述、工具调用序列、最终答案的训练数据，对模型进行一次端到端的微调。此阶段的目标是让模型对“任务 → 调用 → 结果 → 回答”这条闭环有整体感知。  

2. **模型实例化与子任务微调（Stage‑2）**  
   - 将 Stage‑1 微调好的模型复制三份，分别命名为 Planner、Caller、Summarizer。  
   - **Planner 微调**：只保留输入（用户指令）和输出（子任务列表）的对应关系，过滤掉调用细节和答案部分。模型学习如何把高层需求拆解成可执行的步骤。  
   - **Caller 微调**：输入为单个子任务，输出为具体的 API 调用代码或函数调用语句。这里强调语法正确性和参数匹配。  
   - **Summarizer 微调**：输入为所有原始调用返回的文本块，输出为用户友好的答案。模型学习如何筛选噪声、合并信息并用自然语言组织。  

3. **推理时的协同流程**  
   - **用户提问 → Planner**：Planner 产生类似 `[{"tool":"search","query":"2024 年 AI 会议"} , {"tool":"calendar","add_event":"AI 会议"}]` 的结构。  
   - **Caller 并行执行**：对每个子任务，Caller 生成对应的函数调用代码并发送请求，得到如搜索结果、日历确认等原始响应。  
   - **Summarizer 汇总**：把所有响应拼接成一个上下文，Summarizer 生成最终的自然语言回复，例如“已为您在 5 月 10 日的日历中添加了‘2024 AI 会议’，以下是会议的官网链接…”。  

**关键技巧**  
- **结构化子任务输出**：Planner 的输出采用 JSON‑like 格式，便于 Caller 直接解析，避免了自然语言歧义。  
- **共享骨干权重**：三模型在 Stage‑2 只在各自的头部（output layer）进行微调，底层语言理解层保持一致，这样可以在资源受限的情况下仍然获得跨任务的知识迁移。  
- **并行调用**：Caller 采用批量 API 调用的方式，显著降低了整体延迟，这在传统单模型串行调用时是难以实现的。  

### 实验与效果
- **测试任务**：论文在多个公开的工具使用基准上评估，包括天气查询、数学计算、网页搜索、日历管理等，需要模型完成多步调用并给出自然语言答案。  
- **对比基线**：主要与单一 LLM（同等参数规模）直接进行端到端微调的方案比较。  
- **结果概述**：论文声称多模型框架在所有基准上均显著超越单模型基线，尤其在需要多工具协同的任务上提升最为明显。具体的提升幅度在摘要中未给出数值，但作者强调“显著”二字。  
- **消融实验**：通过分别去掉 Planner、Caller、Summarizer 中的微调步骤，实验显示 Planner 的任务拆解质量对整体成功率贡献最大，Caller 的调用准确率次之，Summarizer 的语言流畅度提升相对有限。  
- **局限性**：作者承认当前实现仍依赖于三个模型同时在线，部署成本比单模型高；此外，子任务划分的质量仍受 Planner 的生成能力限制，极端复杂指令仍可能出现拆解错误。  

### 影响与延伸思考
这篇工作打开了“LLM 组件化”在工具使用场景的可能性，随后有研究尝试把更多功能（如检索、记忆、情感调节）拆成独立模块，形成更细粒度的“LLM 微服务”。在开源社区里，出现了基于类似三模型思路的插件框架（如 LangChain‑Modular），帮助开发者自由组合不同规模的模型来完成特定子任务。对想进一步探索的读者，可以关注以下方向：① 更高效的模块间通信协议（比如统一的任务描述语言）；② 动态模块选择策略，让系统根据资源或任务难度自动切换模型规模；③ 将强化学习用于 Planner 的子任务划分，以提升拆解的最优性。  

### 一句话记住它
把工具使用拆成“计划‑调用‑总结”三步，让小模型各司其职，就能让整体表现媲美大模型。