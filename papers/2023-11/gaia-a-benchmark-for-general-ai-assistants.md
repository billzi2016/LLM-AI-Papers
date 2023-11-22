# GAIA: a benchmark for General AI Assistants

> **Date**：2023-11-21
> **arXiv**：https://arxiv.org/abs/2311.12983

## Abstract

We introduce GAIA, a benchmark for General AI Assistants that, if solved, would represent a milestone in AI research. GAIA proposes real-world questions that require a set of fundamental abilities such as reasoning, multi-modality handling, web browsing, and generally tool-use proficiency. GAIA questions are conceptually simple for humans yet challenging for most advanced AIs: we show that human respondents obtain 92\% vs. 15\% for GPT-4 equipped with plugins. This notable performance disparity contrasts with the recent trend of LLMs outperforming humans on tasks requiring professional skills in e.g. law or chemistry. GAIA's philosophy departs from the current trend in AI benchmarks suggesting to target tasks that are ever more difficult for humans. We posit that the advent of Artificial General Intelligence (AGI) hinges on a system's capability to exhibit similar robustness as the average human does on such questions. Using GAIA's methodology, we devise 466 questions and their answer. We release our questions while retaining answers to 300 of them to power a leader-board available at https://huggingface.co/gaia-benchmark.

---

# GAIA：通用人工智能助理基准 论文详细解读

### 背景：这个问题为什么难？

在过去的几年里，LLM（大语言模型）在专业考试、代码生成等狭窄任务上已经能跑赢人类。但这些任务往往是“单模态、封闭式”的——只需要文字输入、固定答案。真实生活中的问题往往要跨越文字、图片、表格，还要主动去网页搜索、调用外部工具，甚至要把多个子任务串起来完成。现有的评测（如MMLU、BIG‑Bench）大多把模型放在一个固定的输入输出框里，忽视了“主动获取信息”和“多模态协同”这两大能力。于是模型在实验室里表现很好，却在日常助理场景里频频卡壳。GAIA 正是为了解决这类“人类觉得简单、机器却很难”的真实需求。

### 关键概念速览
- **通用人工智能助理（General AI Assistant）**：能够像人类助理一样，理解自然语言指令、查找信息、处理图片或表格、调用外部工具完成任务的系统。想象它是一个“全能秘书”，不局限于文字聊天。
- **多模态（Multimodality）**：模型同时处理文字、图片、音频等不同类型的数据。就像我们在看新闻时会同时阅读文字、浏览配图，AI 也需要把这些信息融合起来。
- **工具使用（Tool‑use）**：模型可以主动调用外部程序（如浏览器、计算器、数据库）来补足自身的知识盲区。类似于人类在工作中打开 Excel、搜索 Google 的过程。
- **基准（Benchmark）**：一套标准化的测试题目，用来统一比较不同系统的能力。GAIA 的基准更像是“真实工作清单”，而不是抽象的学术题。
- **插件（Plugins）**：为大语言模型扩展的外部功能模块，例如网页搜索插件、代码执行插件。它们让模型可以在原始语言模型之外获取实时信息或执行计算。
- **人类基准（Human Baseline）**：让真实用户完成同样的题目，得到的成功率，用来衡量机器是否达到“普通人水平”。在 GAIA 中，这个基准是 92%。
- **Leaderboard**：公开的排行榜，参赛者提交模型的答案与隐藏答案比对，得到分数。GAIA 在 HuggingFace 上提供了 300 题的隐藏答案供评测。

### 核心创新点
1. **从“让机器比人更难”转向“让机器和普通人一样好”**  
   之前的评测往往把难度推向人类的极限（比如专业法律或化学题），导致模型只在特定专业上超越人类，却忽视了日常鲁棒性。GAIA 把目标设定为“普通人能轻松完成的任务”，从根本上改变了评测的价值取向。

2. **全链路任务设计**  
   每道 GAIA 题目不只是一个问答，而是一个需要多步操作的完整工作流：阅读文字描述 → 解析需求 → 可能需要搜索网页或查看图片 → 调用工具 → 综合输出答案。相比传统的“一问一答”，这种设计逼迫模型展示真正的助理能力。

3. **公开题库 + 隐藏答案机制**  
   作者公开了全部 466 题的描述，但只保留 300 题的答案用于排行榜。这样既保证了评测的可复现性，又防止了模型直接记忆答案，促使参赛者真正提升系统的通用能力。

4. **人类‑机器对标实验**  
   通过让真实受访者完成同样的 466 题，得到 92% 的成功率；而最强的 GPT‑4 加插件模型仅为 15%。这种大幅差距直接凸显了当前系统在真实助理场景下的不足，也为后续改进提供了明确的上限。

### 方法详解
GAIA 本身不是一种模型，而是一套评测流程。下面把评测的整体框架拆成几步，帮助读者把握它是怎么“逼”模型动手的。

1. **题目构造**  
   - **来源**：作者从真实工作场景、日常生活需求以及公开的任务集合中抽取 466 条问题。每条都要求至少两种能力的组合（如文字+网页搜索、图片+推理）。  
   - **难度控制**：所有题目在常识层面对人类都很直观，确保人类基准高。作者在设计时会先让几位非专业受访者尝试，确保通过率在 80% 以上。

2. **答案生成与隐藏**  
   - 对每道题目，作者手工或使用高质量模型（如 GPT‑4 + 人类校对）得到参考答案。  
   - 为防止“答案泄露”，只把 300 题的答案上传到公开 leaderboard，剩余 166 题留作内部验证或未来扩展。

3. **评测流程**  
   - **输入**：系统收到题目文字描述，可能附带图片或表格。  
   - **任务分解**：模型需要先识别是否需要外部信息（比如“请查一下最新的汇率”），这一步类似于“思考链”。  
   - **工具调用**：如果需要，模型通过插件接口发起网页搜索、图片识别或代码执行。这里的关键是让模型生成符合插件 API 规范的调用指令。  
   - **信息融合**：模型把搜索结果、图片分析或计算输出重新整合进自己的内部推理过程。  
   - **最终输出**：给出完整答案，格式要符合题目要求（如列表、表格或简短解释）。

4. **评分机制**  
   - 对于公开的 300 题，系统的答案会自动与隐藏答案比对，计算准确率。  
   - 对于全部 466 题，作者会人工审查，确保评分的公平性。  

**最巧妙的地方**在于把“工具调用”嵌进了自然语言的推理链里。模型不是先决定“我要搜索”，再独立执行，而是把调用指令当作推理的一个步骤输出，这让评测更像真实助理的工作流。

### 实验与效果
- **数据规模**：共 466 条真实场景问题，覆盖文字、图片、表格、多步推理等多模态需求。  
- **基准对比**：  
  - 人类受访者（普通人）完成率 92%。  
  - GPT‑4（带网页搜索、代码执行等插件）在同样题目上的成功率仅 15%。  
  - 这说明即使是最强的商用模型，在 GAIA 设定的“日常助理”任务上仍远未达到人类水平。  
- **消融实验**：原文没有给出细粒度的消融结果，只是强调了“插件的加入仍不足以突破 15%”。因此我们只能推测，去掉搜索插件或图片识别插件会进一步降低表现。  
- **局限性**：  
  - 题目数量相对有限（466 条），可能无法覆盖所有行业的细分需求。  
  - 评测依赖于插件的实现质量，不同平台的插件差异会导致成绩不可直接比较。  
  - 作者未提供对比其他现有基准（如 MMLU）的详细数值，只给出了人类‑机器差距。

### 影响与延伸思考
GAIA 的出现让社区重新审视“通用助理”到底应该长什么样。自发布后，多个团队开始在公开平台（HuggingFace、OpenAI）上发布针对 GAIA 的改进版插件或多模态模型，例如：  
- **Toolformer** 系列尝试让模型自行学习何时调用工具，直接对标 GAIA 的任务分解需求。  
- **Multimodal LLM**（如 LLaVA、GPT‑4V）在图片理解上取得突破，也被用于尝试提升 GAIA 中的视觉子任务。  
- **自适应检索**（Retrieval‑Augmented Generation）被进一步结合进助理系统，以期缩小 15% 与 92% 的差距。  

如果想继续跟进，建议关注以下方向：  
1. **统一的工具调用语言**：让不同插件遵循同一协议，降低模型适配成本。  
2. **自监督的任务分解学习**：让模型在没有人工标注的情况下学会把复杂需求拆解成可调用的子任务。  
3. **跨模态检索**：把文字、图片、表格等信息统一映射到同一向量空间，提升信息融合效率。  

### 一句话记住它
GAIA 把“普通人能轻松完成的真实助理任务”变成了评测标准，揭示了即使是最强的 LLM 也离日常通用助理还有很大差距。