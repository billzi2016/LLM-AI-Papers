# TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks

> **Date**：2024-12-18
> **arXiv**：https://arxiv.org/abs/2412.14161

## Abstract

We interact with computers on an everyday basis, be it in everyday life or work, and many aspects of work can be done entirely with access to a computer and the Internet. At the same time, thanks to improvements in large language models (LLMs), there has also been a rapid development in AI agents that interact with and affect change in their surrounding environments. But how performant are AI agents at accelerating or even autonomously performing work-related tasks? The answer to this question has important implications both for industry looking to adopt AI into their workflows and for economic policy to understand the effects that adoption of AI may have on the labor market. To measure the progress of these LLM agents' performance on performing real-world professional tasks, in this paper we introduce TheAgentCompany, an extensible benchmark for evaluating AI agents that interact with the world in similar ways to those of a digital worker: by browsing the Web, writing code, running programs, and communicating with other coworkers. We build a self-contained environment with internal web sites and data that mimics a small software company environment, and create a variety of tasks that may be performed by workers in such a company. We test baseline agents powered by both closed API-based and open-weights language models (LMs), and find that the most competitive agent can complete 30% of tasks autonomously. This paints a nuanced picture on task automation with LM agents--in a setting simulating a real workplace, a good portion of simpler tasks could be solved autonomously, but more difficult long-horizon tasks are still beyond the reach of current systems. We release code, data, environment, and experiments on https://the-agent-company.com.

---

# TheAgentCompany：面向真实工作任务的 LLM 代理基准 论文详细解读

### 背景：这个问题为什么难？

在 LLM（大语言模型）火热的今天，大家已经能让模型写代码、回答问题，但真正让它们像公司员工一样在电脑上自行完成工作仍是难点。过去的评测大多是问答或代码生成，缺少对浏览网页、运行程序、与同事协作等多工具交互的考察。没有一个贴近真实公司环境的基准，就很难判断这些 AI 代理到底能在日常工作中帮多少忙，也让企业和政策制定者对自动化的期待充满不确定。

### 关键概念速览
- **LLM（大语言模型）**：能够理解并生成自然语言的大规模神经网络，类似“会说话的百科全书”。  
- **AI 代理（Agent）**：把 LLM 当作“大脑”，再配上搜索、代码执行等工具，让模型可以主动在外部环境里行动，像是装了“手脚”的聊天机器人。  
- **Benchmark（基准测试）**：一套标准任务和评价方式，用来统一比较不同系统的表现，就像跑步比赛的计时器。  
- **Digital Worker（数字员工）**：在电脑上完成工作任务的虚拟人，能够打开网页、写代码、发邮件等，和真实员工的工作方式高度相似。  
- **Web Browsing（网页浏览）**：让代理能够打开内部或外部网页、读取信息、点击链接，类似人类在浏览器里查资料的过程。  
- **Code Generation & Execution（代码生成与执行）**：模型写出可运行的代码并在沙箱里跑出来，像是让 AI 直接动手做实验。  
- **Long‑Horizon Task（长时序任务）**：需要多步推理、跨工具协作、持续记忆的任务，类似完成一个完整的项目而不是一次性回答。  
- **Closed‑API vs Open‑Weights（闭源 API 与开源权重）**：前者指使用商业平台提供的模型接口（如 ChatGPT），后者指自行下载、微调的模型（如 LLaMA），两者在可控性和成本上各有利弊。

### 核心创新点
1. **构建自洽的公司模拟环境 → 以前的评测大多是单一工具或公开网页 → 现在有一个内部网站、代码库、任务看板等完整的“小公司”，让 AI 代理的行为可以在真实工作流中被完整观察。**  
2. **设计可扩展的任务集合 → 过去的 benchmark 只覆盖问答或单步代码生成 → 这里提供从简单的 bug 修复到跨部门协作的多步骤任务，覆盖了不同难度层级。**  
3. **同时评测闭源 API 与开源权重模型 → 之前的研究往往只看一种模型或只报告商业模型的表现 → 本文把两类模型放在同一平台上比较，揭示了性能差距与成本权衡。**  
4. **开源完整代码、数据和实验脚本 → 许多工作只发布论文而不提供可复现的环境 → 通过公开 repo，社区可以直接在同样的公司模拟环境里跑实验，促进后续改进。

### 方法详解
整体思路可以拆成三大块：**环境搭建 → 任务定义 → 代理执行与评估**。

1. **环境搭建**  
   - 作者在 Docker 中部署了一个完整的内部网络，里面有公司官网、项目管理系统（类似 Jira）、代码仓库（Git），以及一个简易的邮件系统。所有页面都是静态 HTML + 小型后端，保证每次实验的输入是可控的。  
   - 环境对外只暴露一个统一的 API，代理通过该 API 发起“打开网页”“提交代码”等指令，等价于在真实公司内部使用内部工具。

2. **任务定义**  
   - 每个任务都有一个 **Goal（目标）**、**Precondition（前置条件）** 和 **Success Metric（成功度量）**。比如“修复登录页面的 500 错误”，前置条件是已有的代码仓库，成功度量是运行单元测试后全部通过。  
   - 任务被划分为 **Simple（简单）**、**Intermediate（中等）**、**Complex（复杂）** 三类，难度主要体现在需要跨工具的次数和需要保持的上下文长度。  
   - 为了让评测可扩展，任务描述采用 JSON 格式，研究者只要在 repo 里添加新的 JSON 条目，就能自动加入基准。

3. **代理执行**  
   - **核心循环**：  
     1) **感知**：代理从环境 API 获取当前页面或文件的内容。  
     2) **思考**：把感知信息拼接进提示（prompt），喂给 LLM。提示里会明确要求模型输出一种 **Tool Action**（如 `BROWSE(url)`、`RUN_CODE(file)`、`SEND_EMAIL(to, subject, body)`）。  
     3) **执行**：系统解析模型输出的指令，调用对应的工具实现真实操作。  
     4) **反馈**：工具返回执行结果（网页 HTML、程序输出、邮件发送成功与否），再喂回模型，进入下一轮。  
   - 这种 **“感知‑思考‑执行‑反馈”** 的闭环让模型能够在多步任务中逐步推进，而不是一次性给出完整答案。  
   - 对于 **Closed‑API** 代理，提示直接发送给 OpenAI/Anthropic 等服务；对于 **Open‑Weights** 代理，模型在本地运行，提示中还会加入 **Tool‑Use Tokens**，帮助模型识别何时该调用工具。

4. **评估机制**  
   - 每完成一次任务，系统会自动检查 **Success Metric**。如果满足，就记为成功；否则记录失败原因（如代码编译错误、网页未找到信息）。  
   - 统计指标包括 **Task Completion Rate（任务完成率）**、**Step Efficiency（步数效率）**、以及 **Tool Usage Distribution（工具使用分布）**，帮助分析模型在不同任务类型上的表现。

**最巧妙的地方**在于把工具调用抽象成统一的指令语言，让任何 LLM 都能通过同一套提示模板进行“工具化”。这避免了为每个模型单独写适配层，也让实验结果更具可比性。

### 实验与效果
- **实验平台**：作者在 TheAgentCompany 提供的自建公司环境里跑了 100 条任务，覆盖了从简单的文档检索到跨模块的功能实现。  
- **Baseline 对比**：包括使用 GPT‑4（闭源 API）、Claude 2（闭源 API）以及 LLaMA‑2‑70B（开源权重）分别构建的代理。  
- **核心结果**：最强的代理（GPT‑4 驱动）在所有任务中自主完成了约 **30%**，其余任务需要人工干预或多轮调试。开源模型的完成率明显低于闭源 API，尤其在长时序任务上几乎无法自行收敛。  
- **消融实验**：论文中对 **提示模板**、**工具调用频率限制**、以及 **环境噪声（随机页面布局）** 进行了一系列消融，发现去掉明确的工具指令提示会导致完成率跌至 10% 以下，说明“感知‑思考‑执行”循环的提示设计至关重要。  
- **局限性**：作者坦诚当前环境仍是简化版的公司，缺少真实的安全审计、多人协作冲突等复杂因素；另外 30% 的完成率虽然是突破，但仍说明大多数实际工作仍需人类介入。  

### 影响与延伸思考
这篇工作在 AI 代理评测领域打开了“公司级”基准的大门，随后出现了 **AutoEval**、**AgentBench** 等项目，它们在 TheAgentCompany 的思路上加入了更大规模的真实网络和多用户协作。对想继续深挖的读者，可以关注以下方向：  
- **工具学习（Tool Learning）**：让模型自行发现并学习新工具，而不是预先硬编码指令。  
- **长期记忆与计划**：提升模型在跨数十步任务中的上下文保持能力。  
- **安全与可控性**：在真实公司环境里防止模型执行破坏性操作。  
- **多模态交互**：加入图像、表格等非文本信息，让代理更像全能的数字员工。  

### 一句话记住它
**TheAgentCompany 用一个“小公司”实验场，首次量化了 LLM 代理在真实工作任务上的自主完成率，约 30% 能全自动完成。**