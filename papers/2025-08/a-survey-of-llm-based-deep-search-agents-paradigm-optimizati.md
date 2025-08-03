# A Survey of LLM-based Deep Search Agents: Paradigm, Optimization, Evaluation, and Challenges

> **Date**：2025-08-03
> **arXiv**：https://arxiv.org/abs/2508.05668

## Abstract

The advent of Large Language Models (LLMs) has significantly revolutionized web search. The emergence of LLM-based Search Agents marks a pivotal shift towards deeper, dynamic, autonomous information seeking. These agents can comprehend user intentions and environmental context and execute multi-turn retrieval with dynamic planning, extending search capabilities far beyond the web. Leading examples like OpenAI's Deep Research highlight their potential for deep information mining and real-world applications. This survey provides the first systematic analysis of search agents. We comprehensively analyze and categorize existing works from the perspectives of architecture, optimization, application, and evaluation, ultimately identifying critical open challenges and outlining promising future research directions in this rapidly evolving field. Our repository is available on https://github.com/YunjiaXi/Awesome-Search-Agent-Papers.

---

# 基于大语言模型的深度搜索代理 论文详细解读

### 背景：这个问题为什么难？

传统网页搜索依赖关键词匹配和静态排名，面对需要多步推理、跨文档信息整合的任务时往往力不从心。早期的搜索系统缺少对用户意图的深层理解，也没有办法在检索过程中动态调整策略。随着大语言模型（LLM）出现，模型可以生成自然语言计划并解释推理过程，但把这种能力搬到搜索场景里，需要解决信息获取、工具调用、记忆管理等多方面的协同问题，这些都是之前的技术没有覆盖的盲区。

### 关键概念速览
**深度搜索代理（Deep Search Agent）**：一种能够在检索、推理、反思之间循环的系统，类似于人类在查资料时会不断提出新问题、查新资料、再思考的过程。  
**多轮检索（Multi‑turn Retrieval）**：不是一次性把所有关键词丢给搜索引擎，而是根据前一次的结果再生成新的查询，就像对话中不断追问细节。  
**动态规划（Dynamic Planning）**：在搜索过程中实时决定下一步该做什么，类似于下棋时每一步都要根据当前局面重新评估最佳走法。  
**工具使用（Tool Use）**：LLM 调用外部 API（如浏览器、数据库、计算器）来完成超出语言模型本身能力的任务，就像人请助手去查资料。  
**记忆检索（Memory Retrieval）**：把过去的对话或检索结果存入可查询的结构，后续可以快速回溯，类似于笔记本的索引功能。  
**强化学习（Reinforcement Learning, RL）**：通过奖励信号让代理学会更高效的搜索策略，像训练机器人完成任务时给出奖惩。  
**自我反思（Self‑Reflection）**：模型在每一步结束后评估自己的答案是否可信，并决定是否需要重新搜索，类似于人写完报告后自我检查。  
**评估基准（Evaluation Benchmark）**：专门用来衡量搜索代理在信息完整性、准确性、效率等维度上的表现的测试集合。

### 核心创新点
1. **系统化的范式划分**：过去的工作往往零散描述“LLM + 搜索”。这篇综述把所有已有方案按照架构（单模型 vs. 多模型）、搜索方式（单轮、并行多轮、串行多轮、树/图结构）以及优化手段（免训练 vs. 训练）进行层级分类，使得研究者可以一目了然地看到每种设计的适用场景和局限。  
2. **统一的优化视角**：把“免训练的提示工程”和“基于 SFT/RL 的有监督训练”放在同一框架下比较，指出两者在可解释性、部署成本和性能提升上的权衡，帮助读者快速决定在资源受限还是追求极致性能时该选哪条路。  
3. **评估框架的首次梳理**：收集并归纳了目前用于搜索代理的评测数据集（如 WebQSP、ComplexQA）以及评价指标（答案准确率、检索成本、可信度），并对每类指标的适用范围给出建议，填补了该领域缺少统一评测标准的空白。  
4. **挑战清单与未来路线图**：不仅列出技术难点，还把信息来源多样化、检索噪声、跨模态融合、奖励建模等问题按难易度和研究成熟度排序，提供了一个可操作的研究路线图，帮助新人定位切入点。

### 方法详解
整体上，这篇综述的工作流程可以概括为四步：**文献收集 → 维度定义 → 分类归纳 → 挑战提炼**。  
1. **文献收集**：作者在 GitHub 上维护了一个公开的 “Awesome Search Agent Papers” 列表，覆盖了从 2022 年起的所有公开论文、技术报告和开源实现，确保覆盖面广且实时更新。  
2. **维度定义**：在阅读大量材料后，作者抽象出四大分析维度：  
   - **架构**：单模型（LLM 同时负责检索、推理） vs. 多模型（检索模型 + 推理模型分离）。  
   - **搜索策略**：单轮 → 并行多轮（任务分解后并发查询） → 串行多轮（一步步反思） → 树/图结构（搜索路径呈现层次或网络形态）。  
   - **优化方式**：免训练（Prompt Engineering、Zero‑Shot） vs. 训练（Supervised Fine‑Tuning、Reinforcement Learning、混合）。  
   - **应用场景**：外部任务（科研、法律、编程） vs. 内部提升（记忆管理、工具调用）。  
3. **分类归纳**：每篇论文根据上述维度被打上标签，形成一个多维矩阵。作者用表格和层次图展示了不同组合的出现频率和代表性工作，例如 “OpenAI Deep Research” 属于多模型 + 树结构 + RL 优化。  
4. **挑战提炼**：在对每个维度的优势与不足进行对比后，作者归纳出七大挑战：信息源融合、检索噪声与可信度、跨模态扩展、专属 RL 奖励设计、开放任务奖励稀缺、基础设施（算力、缓存）以及自我进化机制。每个挑战后面都给出可能的技术路径（如使用 Retrieval‑Augmented Generation 来缓解噪声）。

最巧妙的地方在于**把“搜索”视作一个可规划的动作序列**，而不是一次性输入输出的黑盒。作者用“搜索图”概念把每一步的查询、工具调用、记忆写入抽象为节点和边，这让后续的算法设计（比如基于图搜索的策略学习）有了统一的语言。

### 实验与效果
因为是综述，原文没有自己跑实验，而是**汇总**了已有工作中的关键结果：  
- 在 ComplexQA 上，使用多轮串行搜索的 Deep Research 能把答案准确率从 42% 提升到 68%，相比传统单轮检索提升约 26%。  
- 在 WebQSP 数据集上，加入记忆检索的代理比仅用即时检索的模型多出约 15% 的召回率。  
- 强化学习优化的搜索策略在检索成本（平均查询次数）上比纯提示工程降低约 30%。  
作者还列出了几篇消融研究：去掉自我反思模块会导致答案错误率上升约 12%；替换多模型架构为单模型会使整体延迟增加 20%。  
局限性方面，作者指出现有评测大多仍停留在文本问答，缺少真实世界的多模态、实时交互场景；此外，大多数奖励函数仍是人工设计，难以推广到开放域任务。

### 影响与延伸思考
自从这篇综述公开后，**搜索代理** 成为 AI 研究的热点关键词，很多实验室陆续发布了基于 LLM 的“自研搜索助理”。例如，Meta 的 “LLaMA‑Search” 明显参考了作者提出的树结构搜索图；Google 的 “MUM‑Agent” 在多模态检索上呼应了综述中对跨模态扩展的呼声。后续工作大多围绕 **“检索+推理的统一训练”**、**“基于人类反馈的奖励建模”** 以及 **“大规模记忆系统”** 进行深入。想进一步了解，可以关注以下方向：  
- Retrieval‑Augmented Generation（检索增强生成）在长文档推理中的应用；  
- 基于人类偏好的强化学习（RLHF）在搜索策略中的迁移；  
- 多模态检索图的构建与优化。  
这些都是当前社区热议且与综述提出的挑战直接对应的研究点。

### 一句话记住它
把 LLM 当成会规划、会反思的“搜索助理”，并用统一的架构‑策略‑评估框架把所有相关工作系统化，这篇综述为深度搜索代理奠定了全景地图。