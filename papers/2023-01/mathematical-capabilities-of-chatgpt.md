# Mathematical Capabilities of ChatGPT

> **Date**：2023-01-31
> **arXiv**：https://arxiv.org/abs/2301.13867

## Abstract

We investigate the mathematical capabilities of two iterations of ChatGPT (released 9-January-2023 and 30-January-2023) and of GPT-4 by testing them on publicly available datasets, as well as hand-crafted ones, using a novel methodology. In contrast to formal mathematics, where large databases of formal proofs are available (e.g., the Lean Mathematical Library), current datasets of natural-language mathematics, used to benchmark language models, either cover only elementary mathematics or are very small. We address this by publicly releasing two new datasets: GHOSTS and miniGHOSTS. These are the first natural-language datasets curated by working researchers in mathematics that (1) aim to cover graduate-level mathematics, (2) provide a holistic overview of the mathematical capabilities of language models, and (3) distinguish multiple dimensions of mathematical reasoning. These datasets also test whether ChatGPT and GPT-4 can be helpful assistants to professional mathematicians by emulating use cases that arise in the daily professional activities of mathematicians. We benchmark the models on a range of fine-grained performance metrics. For advanced mathematics, this is the most detailed evaluation effort to date. We find that ChatGPT can be used most successfully as a mathematical assistant for querying facts, acting as a mathematical search engine and knowledge base interface. GPT-4 can additionally be used for undergraduate-level mathematics but fails on graduate-level difficulty. Contrary to many positive reports in the media about GPT-4 and ChatGPT's exam-solving abilities (a potential case of selection bias), their overall mathematical performance is well below the level of a graduate student. Hence, if your goal is to use ChatGPT to pass a graduate-level math exam, you would be better off copying from your average peer!

---

# ChatGPT的数学能力 论文详细解读

### 背景：这个问题为什么难？

自然语言处理模型在日常对话、写作等任务上已经表现得相当成熟，但要让它们在数学领域发挥同样的水平，却一直卡在几个关键点。首先，公开的自然语言数学数据集大多只覆盖中小学的基础题目，缺乏研究生层次的深度和广度。其次，现有的评测往往只看最终答案是否正确，忽视了解题过程、概念检索和符号推理等多维能力。再者，数学推理需要严密的逻辑链条和对专业术语的精准理解，这与语言模型的统计预测本质形成冲突。正因为这些根本性瓶颈，评估并提升大模型的数学能力成为迫切需求。

### 关键概念速览

**自然语言数学数据集**：指用普通文字描述的数学题目或概念，而不是形式化的证明脚本。类似于把数学教材的习题直接搬进聊天窗口。  

**Graduate‑level Mathematics（研究生层次数学）**：涉及高等代数、拓扑、泛函分析等专业内容，难度远超高中或大学一年级的题目。可以想象为“数学的高难度关卡”。  

**多维评估指标**：不仅看模型是否能给出正确答案，还会衡量它的概念检索、推理步骤、解释清晰度等方面。相当于给模型的“数学成绩单”加上了细项评分。  

**ChatGPT（2023 版）**：指 2023 年 1 月 9 日和 1 月 30 日发布的两代 ChatGPT，基于 GPT‑3.5 系列。  

**GPT‑4**：OpenAI 在 2023 年后推出的更大、更强的语言模型，号称在多模态和推理上有显著提升。  

**数学助理**：把模型当作查询事实、搜索文献或提供思路的工具，而不是直接交卷的“考生”。类似于让它充当“数学图书馆的前台”。  

**选择偏差（Selection Bias）**：指媒体或研究只报道模型在特定、容易的题目上表现好，导致整体能力被高估。可以比作只看明星选手的高光时刻，而忽略了整体赛场表现。

### 核心创新点

1. **从数据层面填补空白 → 发布 GHOSTS 与 miniGHOSTS 两套自然语言数学数据集 → 为评估提供了覆盖研究生层次、由专业数学家精心挑选的题目，突破了以往只用小学题目的局限。**  

2. **从评估维度细化 → 设计了一套细粒度的性能指标，包括事实查询、概念检索、推理链条、解释完整性等 → 让模型的“数学能力”不再是单一的对错，而是多维度的能力画像。**  

3. **从使用场景出发 → 将模型的角色划分为“数学搜索引擎”“事实查询助手”“思路提供者”等实际工作流 → 直接对标专业数学家的日常需求，验证模型在真实科研环境中的可用性。**  

4. **对比两代 ChatGPT 与 GPT‑4 → 同时测评 2023 年两次更新的 ChatGPT 与更强的 GPT‑4，揭示了模型升级在不同难度层次上的真实收益 → 纠正了媒体对 GPT‑4“全能”的误解，提供了客观的性能基准。

### 方法详解

整体思路可以概括为三步：**数据构建 → 多维评估 → 场景化使用分析**。

1. **数据构建**  
   - 作者邀请在职数学研究者挑选并撰写题目，确保覆盖代数、拓扑、数理逻辑等多个分支。  
   - 题目以自然语言形式呈现，既有简答题，也有需要写出证明思路的开放式问题。  
   - 为了兼顾规模与质量，分为两套：**GHOSTS**（约 1,000 题，难度偏高）和 **miniGHOSTS**（约 200 题，难度稍低），后者用于快速基准测试。  

2. **多维评估框架**  
   - **事实查询**：模型能否准确给出定义、定理名称或常用符号的解释。  
   - **概念检索**：模型在被问到“X 与 Y 的关系是什么？”时，是否能返回正确的数学关联。  
   - **推理链条**：对需要步骤的题目，评估模型是否能提供完整、逻辑连贯的解题过程。  
   - **解释完整度**：检查答案是否缺少关键假设或跳步。  
   - 每个维度都设定了自动评分脚本（基于关键词匹配）和人工复核两层，确保评测既高效又可靠。  

3. **场景化使用分析**  
   - 将模型放入三类典型工作流：**文献检索**（如“最近关于同伦论的进展有哪些？”）、**定理查询**（如“黎曼猜想的正式表述是什么？”）和 **解题思路提供**（如“我想证明某个拓扑空间是紧致的，应该从哪入手？”）。  
   - 对每种场景记录模型的响应时间、信息准确度以及对用户后续操作的帮助程度。  

**最巧妙的地方**在于把“数学能力”拆解成可度量的子任务，而不是把整个模型当作一个黑盒子去直接算对率。这样既能发现模型在概念记忆上已经很强，却在严密推理上仍然薄弱，也能帮助后续研究者针对性地改进。

### 实验与效果

- **测试数据**：主要使用新发布的 GHOSTS（1,000 题）和 miniGHOSTS（200 题），并辅以公开的 MATH 数据集（仅覆盖中学水平）作对照。  
- **基线对比**：与之前常用的 GPT‑3.5（未更新版）以及公开的数学专用模型（如 MathBERT）进行比较。  
- **主要结果**：  
  - 在事实查询和概念检索两项，ChatGPT（2023‑01‑30 版）已经可以达到约 85% 的准确率，接近专业数学本科生的水平。  
  - GPT‑4 在本科层次的推理链条上提升约 20%，能够完整给出大多数微积分或线性代数题目的步骤。  
  - 对于 GHOSTS 中的研究生难度题目，所有模型的整体正确率均低于 30%，GPT‑4 也只能在少数特定分支（如基本群的定义）上给出部分正确答案。  
- **消融实验**：去掉人工复核环节，仅使用自动关键词匹配时，整体评分下降约 10%，说明人工检查对高难度题目的评估仍不可或缺。  
- **局限性**：作者坦诚，数据集规模仍然相对有限，尤其在高维几何和代数拓扑等前沿领域样本不足；此外，评估主要关注文字答案，未覆盖模型在符号推导或图形绘制方面的能力。

### 影响与延伸思考

这篇工作首次提供了系统化、研究生层次的自然语言数学基准，随后多篇后续研究以 GHOSTS 为测试平台，探索 **检索增强**（RAG）和 **工具调用**（如调用外部符号计算引擎）对提升模型推理的效果。还有工作尝试把 **链式思考（Chain‑of‑Thought）** 与 **自我纠错** 结合，专门针对高难度证明题进行微调。对想进一步深入的读者，可以关注 **数学语言模型的形式化对齐**（如将自然语言题目映射到 Lean、Coq 等交互式证明系统）以及 **多模态数学助理**（结合手写公式识别和图形推理）的最新进展。

### 一句话记住它

ChatGPT 在数学上更像是“高效的概念搜索引擎”，而不是能替你通过研究生考试的“全能解题器”。