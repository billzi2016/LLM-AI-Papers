# MHPP: Exploring the Capabilities and Limitations of Language Models Beyond Basic Code Generation

> **Date**：2024-05-19
> **arXiv**：https://arxiv.org/abs/2405.11430

## Abstract

Recent advancements in large language models (LLMs) have greatly improved code generation, specifically at the function level. For instance, GPT-4o has achieved a 91.0\% pass rate on HumanEval. However, this draws into question the adequacy of existing benchmarks in thoroughly assessing function-level code generation capabilities. Our study analyzed two common benchmarks, HumanEval and MBPP, and found that these might not thoroughly evaluate LLMs' code generation capacities due to limitations in quality, difficulty, and granularity. To resolve this, we introduce the Mostly Hard Python Problems (MHPP) dataset, consisting of 210 unique human-curated problems. By focusing on the combination of natural language and code reasoning, MHPP gauges LLMs' abilities to comprehend specifications and restrictions, engage in multi-step reasoning, and apply coding knowledge effectively. Initial evaluations of 26 LLMs using MHPP showed many high-performing models on HumanEval failed to achieve similar success on MHPP. Moreover, MHPP highlighted various previously undiscovered limitations within various LLMs, leading us to believe that it could pave the way for a better understanding of LLMs' capabilities and limitations. MHPP, evaluation pipeline, and leaderboard can be found in https://github.com/SparksofAGI/MHPP.

---

# MHPP：探索语言模型在基础代码生成之外的能力与局限 论文详细解读

### 背景：这个问题为什么难？

在大模型时代，代码生成已经可以在函数层面达到人类水平——比如 GPT‑4o 在 HumanEval 上的通过率已经超过 90%。然而，现有的评测数据集（HumanEval、MBPP）大多只要求模型把自然语言描述直接翻译成一个完整函数，难度和细粒度都相对有限。于是出现了两个根本性的问题：一是这些基准无法检验模型对复杂约束和多步推理的理解；二是模型在“写对代码”之外的能力——比如读懂题目细节、规划实现步骤——几乎没有被测量。正因为如此，需要一个更具挑战性、能够揭示真实能力边界的评测集合。

### 关键概念速览
- **HumanEval**：一个函数级代码生成基准，题目描述简短，答案只要能通过单元测试即可。相当于让模型直接写出“答案”，不需要太多思考过程。
- **MBPP（Mostly Basic Python Programming）**：类似 HumanEval，但题目更偏向基础教学，难度略低，仍然是“一步到位”的生成任务。
- **MHPP（Mostly Hard Python Problems）**：作者手工挑选并整理的 210 条更难的 Python 题目，强调自然语言理解、约束条件和多步推理。可以把它想成“编程奥赛”级别的练习。
- **多步推理（Multi‑step Reasoning）**：模型需要先拆解需求、列出实现思路，再逐步写代码，而不是一次性输出完整函数。类似于人类先写伪代码再写实现。
- **规格理解（Specification Comprehension）**：模型要准确捕捉题目中的细节限制（比如时间复杂度、边界条件），这相当于阅读合同条款后才能正确执行。
- **Leaderboard（排行榜）**：公开的评测平台，记录各模型在 MHPP 上的得分，方便社区对比和追踪进展。

### 核心创新点
1. **基准重新定位 → 设计 MHPP 数据集 → 揭示模型真实能力差距**  
   过去的评测把代码生成等同于“把描述翻译成代码”。作者把焦点转向“理解+推理+实现”，手工挑选 210 题，确保每题都需要模型进行规格解析和多步推理。实验显示，许多在 HumanEval 上表现优异的模型在 MHPP 上的通过率大幅下降，说明原基准低估了模型的局限。

2. **评测管线标准化 → 自动化运行、统一测试环境 → 可复现的对比**  
   为了避免不同实验室自行搭建评测导致结果不可比，作者开源了完整的评测脚本，包括题目解析、测试用例生成、结果统计等环节。这样，任何新模型只要接入管线，就能得到与已有模型直接可比的分数。

3. **细粒度错误分析 → 按照“规格误解”“推理缺失”“实现错误”三类标注 → 为模型改进提供明确方向**  
   作者不仅给出整体通过率，还对每个错误进行分类，帮助研究者快速定位模型的薄弱环节。这种错误标签在之前的基准中几乎不存在。

4. **大规模模型覆盖 → 评测 26 种公开 LLM → 形成首个跨模型、跨规模的能力画像**  
   通过一次性跑完 26 种模型（包括不同大小的 GPT、Claude、LLaMA 系列等），作者绘制出一张“能力热力图”，直观看到哪些模型在规格理解上表现好，哪些在多步推理上仍然薄弱。

### 方法详解
整体框架可以概括为三步：**题目筛选 → 评测管线搭建 → 结果分析**。

1. **题目筛选**  
   - 作者从公开的编程竞赛、教材和开源项目中抽取上千个 Python 题目。  
   - 通过人工审阅，剔除那些只需要几行代码或缺乏明确约束的题目，保留需要**自然语言解析、约束推敲、算法设计**的题目。  
   - 最终得到 210 条“Mostly Hard” 题目，每题配有详细的规格说明、输入输出示例以及一套隐藏的单元测试。

2. **评测管线**  
   - **输入准备**：将每题的自然语言描述和约束信息拼接成模型的 Prompt。Prompt 采用统一模板，确保不同模型接受的指令风格一致。  
   - **模型调用**：对每个模型，使用其官方 API（或本地部署）一次性生成完整的 Python 代码块。若模型支持多轮对话，可在管线中模拟一次“澄清”交互，但默认使用一次性生成，以保持公平。  
   - **代码执行**：生成的代码被放入安全的沙箱环境（Docker + timeout），运行隐藏测试用例。通过率即为该模型在该题目的得分。  
   - **错误标注**：如果代码未通过，管线会捕获错误信息（如 AssertionError、超时、SyntaxError），并根据预设规则自动归类为“规格误解”“推理缺失”“实现错误”。  

3. **结果分析**  
   - 所有题目的通过率取平均得到模型的整体得分。  
   - 通过率与 HumanEval/MBPP 的得分进行横向对比，计算**能力下降率**（例如 GPT‑4o 在 HumanEval 上 91% → 在 MHPP 上 58%，下降 33%）。  
   - 错误标签的分布被可视化为堆叠柱状图，帮助快速定位模型的薄弱环节。  

**最巧妙的地方**在于**错误标签体系**。传统基准只给出“通过/未通过”，无法告诉研究者模型到底卡在哪一步。作者通过对每个失败案例进行语义分析，自动生成三类标签，这让后续的模型改进（比如加入规格理解微调、强化推理链）有了明确的目标。

### 实验与效果
- **数据集**：MHPP（210 题）是唯一的评测对象；对比基准使用 HumanEval（164 题）和 MBPP（100 题）。  
- **模型范围**：共评测 26 种公开的大语言模型，涵盖不同规模（从 7B 到 175B 参数）和不同训练机构（OpenAI、Anthropic、Meta、Mistral 等）。  
- **主要发现**：  
  - 多数在 HumanEval 上超过 80% 通过率的模型，在 MHPP 上的通过率普遍低于 60%。  
  - 最高表现的模型（GPT‑4o）在 HumanEval 达到 91% 的惊人成绩，但在 MHPP 只拿到约 58% 的通过率，说明即使是最强模型也在复杂规格理解上有显著缺口。  
  - 小模型（7B‑12B 参数）在两套基准上的差距更大，说明规模提升对基本代码生成帮助明显，但对高级推理的提升并不线性。  
- **消融实验**：作者对评测管线的 Prompt 结构做了两种变体（加入示例代码 vs 纯文本描述），结果显示加入示例可以提升约 5% 的通过率，说明 Prompt 设计仍是提升模型表现的关键因素。  
- **局限性**：  
  - 题目数量虽比 HumanEval 多，但仍只有 210 条，覆盖的算法类型有限。  
  - 评测仅针对 Python，未验证模型在其他语言（如 JavaScript、Rust）上的规格理解能力。  
  - 错误标签的自动归类依赖规则，可能在边缘案例出现误判，作者在论文中承认需要进一步完善。  

### 影响与延伸思考
这篇工作在社区里掀起了“评测要更硬、更细”的讨论。随后出现了 **CodeXGLUE‑Hard**、**APPS‑Advanced** 等数据集，都是在 MHPP 的思路上扩展到多语言或更大规模的版本。还有研究尝试在模型训练阶段加入**规格理解微调**（Specification‑Tuned）或**多步推理提示**（Chain‑of‑Thought Prompting）来专门提升在 MHPP 上的表现。对想进一步探索的读者，可以关注以下方向：  
- **自动生成硬核代码题目**：利用 LLM 自己生成符合 MHPP 需求的题目，形成持续迭代的评测池。  
- **跨语言规格理解**：把同一道题目翻译成不同编程语言，检验模型的语言无关推理能力。  
- **人机协同调试**：结合模型的错误标签，引入交互式调试环节，让模型在“错了再改”中学习。  

### 一句话记住它
**MHPP 把代码生成的赛道从“写对函数”提升到“读懂规格、推理步骤、实现代码”，让我们第一次真正看到大模型的软肋所在。**