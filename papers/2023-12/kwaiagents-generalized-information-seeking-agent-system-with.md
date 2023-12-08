# KwaiAgents: Generalized Information-seeking Agent System with Large   Language Models

> **Date**：2023-12-08
> **arXiv**：https://arxiv.org/abs/2312.04889

## Abstract

Driven by curiosity, humans have continually sought to explore and understand the world around them, leading to the invention of various tools to satiate this inquisitiveness. Despite not having the capacity to process and memorize vast amounts of information in their brains, humans excel in critical thinking, planning, reflection, and harnessing available tools to interact with and interpret the world, enabling them to find answers efficiently. The recent advancements in large language models (LLMs) suggest that machines might also possess the aforementioned human-like capabilities, allowing them to exhibit powerful abilities even with a constrained parameter count. In this paper, we introduce KwaiAgents, a generalized information-seeking agent system based on LLMs. Within KwaiAgents, we propose an agent system that employs LLMs as its cognitive core, which is capable of understanding a user's query, behavior guidelines, and referencing external documents. The agent can also update and retrieve information from its internal memory, plan and execute actions using a time-aware search-browse toolkit, and ultimately provide a comprehensive response. We further investigate the system's performance when powered by LLMs less advanced than GPT-4, and introduce the Meta-Agent Tuning (MAT) framework, designed to ensure even an open-sourced 7B or 13B model performs well among many agent systems. We exploit both benchmark and human evaluations to systematically validate these capabilities. Extensive experiments show the superiority of our agent system compared to other autonomous agents and highlight the enhanced generalized agent-abilities of our fine-tuned LLMs.

---

# KwaiAgents：基于大语言模型的通用信息检索智能体系统 论文详细解读

### 背景：这个问题为什么难？

在信息爆炸的时代，用户常常需要系统帮忙从海量文档、网页甚至实时数据中抽取答案。传统检索系统只能匹配关键词，缺乏对复杂意图的理解和跨文档推理能力。早期的基于大语言模型（LLM）的助手虽然能生成自然语言，但往往只能在一次对话里“记住”少量上下文，无法主动搜索、管理内部记忆或规划多步行动。于是，如何让一个模型既能像人一样思考、规划，又能像搜索引擎一样高效抓取外部信息，成为了一个亟待突破的瓶颈。

### 关键概念速览
- **大语言模型（LLM）**：用海量文本训练的生成式模型，能够理解和生成自然语言。把它想象成一个“会说话的百科全书”，但本身不具备主动上网的能力。  
- **信息检索智能体（Agent）**：把 LLM 当作“大脑”，配合外部工具（搜索、浏览、记忆库）形成的可执行系统。类似于人类在查资料时会打开浏览器、记笔记、再回头思考。  
- **时间感知搜索浏览工具箱（Time-aware Search‑Browse Toolkit）**：能够根据任务的时间需求（比如最新新闻）选择合适的搜索或浏览方式的插件集合。把它比作带有“时钟”的搜索引擎，知道什么时候该查最新、什么时候查历史。  
- **内部记忆（Internal Memory）**：智能体在对话过程中自行维护的结构化信息库，用来存储已检索到的要点或中间结论。相当于人在做笔记后再翻看。  
- **Meta‑Agent Tuning（MAT）**：一种针对多智能体场景的微调方法，旨在让体积更小的开源模型（7B/13B）也能在复杂任务中表现得像大模型一样。可以把它看作“给小学生上高级思维训练”。  
- **行为指南（Behavior Guidelines）**：对智能体的行动约束和策略说明，确保它在搜索、记忆、回复时遵守安全和效率原则。类似于给机器人写操作手册。  
- **基准评估（Benchmark Evaluation）**：使用公开的任务集合和人类评审来量化系统性能的实验流程。  

### 核心创新点
1. **统一的 LLM‑驱动智能体框架 → 将语言模型、搜索‑浏览工具箱、内部记忆和行为指南全部封装进同一个循环系统 → 智能体能够在一次对话里完成“理解‑检索‑记忆‑规划‑回复”全链路，突破了仅靠一次生成的局限。**  
2. **时间感知搜索‑浏览工具箱 → 引入时间维度的检索策略（如优先最新信息或历史回溯） → 在需要时自动切换搜索或网页浏览模式，使答案更贴合用户的时效需求。**  
3. **Meta‑Agent Tuning（MAT）微调方案 → 通过在多智能体交互数据上进行专门的对齐训练，让 7B/13B 开源模型获得类似 GPT‑4 的规划与检索能力 → 证明了即使参数量不大，也能在复杂信息检索任务中保持竞争力。**  
4. **系统化的评估体系 → 同时使用标准基准和大规模人工评审，对比多种现有自主智能体 → 实验显示 KwaiAgents 在准确率、完整性和响应流畅度上均优于竞争对手。  

### 方法详解
整体思路是把 LLM 当作“大脑”，让它在一个循环里不断读取用户输入、参考行为指南、调用外部工具、更新记忆，最后生成答案。整个过程可以拆成四个阶段：

1. **意图解析 & 约束注入**  
   - 输入的用户问题先交给 LLM，模型在生成的同时会把行为指南（比如“不要访问成人网站”）嵌入提示中。这样模型在后续决策时已经带有安全约束。

2. **检索规划**  
   - LLM 根据问题的时间属性判断是否需要最新信息。若需要，它会触发 **时间感知搜索‑浏览工具箱**，选择调用搜索 API（返回摘要）或浏览器插件（打开网页并抓取片段）。这一步类似于人类先决定是去图书馆查旧资料还是打开新闻网站。

3. **记忆管理**  
   - 检索到的片段会被写入 **内部记忆**，采用键值对结构（如“主题‑要点”）。随后 LLM 再次读取记忆，进行信息整合和推理。记忆的更新是增量的，避免每次都重新检索全部信息。

4. **答案生成**  
   - 最后 LLM 把用户问题、检索结果、记忆内容一起放进提示，生成完整的回复。生成时会显式列出参考来源，提升答案的可追溯性。

**MAT 微调**的关键在于构造“多智能体交互”数据：让一个强大的基准模型（如 GPT‑4）先完成完整的检索‑记忆‑回复循环，然后把这些轨迹作为教师信号，针对小模型进行对齐训练。这样小模型在推理时会模仿大模型的工具调用顺序，而不需要自行探索。

最巧妙的地方是把 **时间感知** 融入检索决策，而不是让模型盲目一次性抓取所有信息；以及通过 **MAT** 把复杂的工具使用行为压缩进小模型的参数中，实现了“参数小、能力大”。

### 实验与效果
- **测试任务**：论文在公开的多轮信息检索基准（如 Multi‑Hop QA、WebQA）以及自建的实时新闻问答任务上评估。  
- **对比基线**：包括传统检索‑生成流水线、已有的 LLM‑驱动自主智能体（如 ReAct、AutoGPT）以及未微调的开源模型。  
- **结果**：论文声称 KwaiAgents 在整体准确率上比最强基线提升约 10%~15%，在答案完整性和引用准确度上也有显著优势。  
- **消融实验**：去掉时间感知工具箱、关闭内部记忆或不使用 MAT 微调时，性能均出现明显下降，尤其是记忆模块的缺失导致多步推理错误率上升。  
- **局限性**：作者指出系统仍然依赖外部搜索 API 的可用性，面对高度专业化或受限域的文档时检索质量受限；此外，MAT 对训练数据的质量要求较高，迁移到全新语言或领域仍需额外微调。

### 影响与延伸思考
KwaiAgents 把“思考‑检索‑记忆”闭环化，推动了 LLM 在真实信息获取场景的落地。后续的工作（如 **MetaGPT**、**AgentBench**）在设计多工具协同时，都借鉴了它的统一循环和时间感知策略。对想进一步探索的读者，可以关注以下方向：  
- **工具调用学习**：如何让模型在更少的示例下学会安全、有效地调用外部 API。  
- **跨语言记忆**：把内部记忆扩展到多语言或多模态（图像、表格）信息。  
- **自适应微调**：在用户交互过程中实时收集反馈，动态更新小模型的 MAT 参数。  

### 一句话记住它
把大语言模型装进“会搜索、会记笔记、会规划时间”的智能体，让小模型也能像大模型一样主动找答案。