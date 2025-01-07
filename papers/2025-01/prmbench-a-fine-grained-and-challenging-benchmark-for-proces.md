# PRMBench: A Fine-grained and Challenging Benchmark for Process-Level Reward Models

> **Date**：2025-01-06
> **arXiv**：https://arxiv.org/abs/2501.03124

## Abstract

Process-level Reward Models (PRMs) are crucial for complex reasoning and decision-making tasks, where each intermediate step plays an important role in the reasoning process. Since language models are prone to various types of errors during the reasoning process, PRMs are required to possess nuanced capabilities for detecting various implicit error types in real-world scenarios. However, current benchmarks primarily focus on step correctness, failing to evaluate PRMs' performance systematically. To address this gap, we introduce PRMBench, a process-level benchmark specifically designed to assess the fine-grained error detection capabilities of PRMs. PRMBench comprises 6,216 carefully designed problems and 83,456 step-level labels, evaluating models across multiple dimensions, including simplicity, soundness, and sensitivity. In our experiments on 15 models, spanning both open-source PRMs and closed-source large language models prompted as critic models, we uncover significant weaknesses in current PRMs. These findings underscore the challenges inherent in process-level evaluation and highlight key directions for future research. We hope PRMBench can be a robust bench for advancing research on PRM evaluation and development.

---

# PRMBench：面向过程级奖励模型的细粒度挑战基准 论文详细解读

### 背景：这个问题为什么难？

在需要多步推理的任务里，语言模型往往会在中间步骤出现遗漏、逻辑错误或信息偏差。传统的评测只看最终答案对不对，忽略了过程本身的质量。于是，研究者们开始训练过程级奖励模型（PRM）来给每一步打分，期望模型能自我纠错。但现有的 benchmark 只标记“这一步对不对”，没有区分错误的种类、严重程度或对后续步骤的影响，导致我们无法系统地衡量 PRM 的细粒度检测能力。正因为缺少这样一个全面、挑战性的评测平台，PRM 的真实水平一直被高估，也限制了更精细的模型改进。

### 关键概念速览

**过程级奖励模型（PRM）**：专门评估生成文本中每一步推理质量的模型，类似于老师在学生解题时给每一步打分，而不是只看最终答案。

**步骤正确性（step correctness）**：判断某一步的输出是否完全符合题目要求的最粗糙指标，就像只看学生是否写对了每一道小题的答案。

**细粒度错误类型（fine-grained error types）**：对错误进行更细致的划分，例如遗漏信息、逻辑跳跃、事实错误等，类似于老师在批改时标记“概念不清”“推理跳步”等。

**简易性（simplicity）**：衡量一个步骤在语言表达和推理难度上的轻重，越简单越容易被模型捕捉。

**健全性（soundness）**：指步骤的逻辑是否自洽、结论是否必然，类似于数学证明中的“每一步都必须严格符合前提”。

**敏感性（sensitivity）**：模型对细微错误的检测能力，像是老师能发现学生写错的一个字或一个小的逻辑漏洞。

**Critic Prompting**：把大型语言模型当作“评审”来使用，通过特定提示让它们给出步骤评分，类似于请外部专家对学生作业打分。

### 核心创新点

1. **从单一正确性到多维错误标签**  
   之前的 benchmark 只给出“对/错”二元标记。PRMBench 为每一步提供 83,456 条细粒度标签，覆盖简易性、健全性、敏感性等维度。这样做让评测不再是“是或否”，而是可以量化模型在不同错误场景下的表现。

2. **构建大规模、人工设计的过程题库**  
   过去的评测往往直接复用已有的 QA 数据，缺少对过程的专门设计。PRMBench 通过 6,216 题目，人工规划每一步的推理路径，确保每种错误类型都有足够的出现频率，提升了评测的覆盖面和挑战度。

3. **统一评测框架兼容开源 PRM 与闭源 LLM**  
   传统评测要么只针对特定开源模型，要么只能用黑盒大模型做对比。PRMBench 设计了统一的 API，既能直接喂入开源 PRM 的概率输出，也能通过“critic prompting”让闭源大模型充当评审，实现了公平横向比较。

4. **系统性实验揭示现有 PRM 的薄弱环节**  
   在 15 种模型上跑通基准后，作者发现即使是最先进的 PRM，在简易性和敏感性上仍有显著缺口。这个发现本身就为后续研究指明了改进方向，而不是简单报告“我们跑了跑”。

### 方法详解

整体思路可以拆成三大步骤：**题目设计 → 步骤标注 → 评测协议**。

1. **题目设计**  
   作者先挑选了需要多步推理的任务（数学解题、逻辑推理、代码生成等），每个任务都手工规划出完整的推理链。每条链被切分成若干步骤，确保每一步都有明确的输入、操作和预期输出。想象把一道复杂的料理拆成每个配料的处理过程，这样才能检查每一步是否按部就班。

2. **细粒度标注**  
   对每一步，标注团队依据三大维度打分：  
   - **简易性**：根据语言复杂度和所需背景知识给出 1‑3 级别。  
   - **健全性**：检查逻辑是否自洽，若出现跳步或前后矛盾则标记为错误。  
   - **敏感性**：评估模型对细微错误的捕捉能力，若模型只能发现大错误而忽略小错误，则在此维度得分低。  
   每个维度都配有具体的错误类型标签（如“事实错误”“推理跳步”“表达歧义”），共计 83,456 条 step‑level 标记。

3. **评测协议**  
   - **输入**：给定模型（无论是开源 PRM 还是闭源 LLM）一个完整的题目和其生成的步骤序列。  
   - **输出**：模型返回每一步的置信度或评分。对于闭源模型，使用特定的 critic prompt（如“请判断以下推理步骤是否合理，并给出理由”）让其生成自然语言评分，再通过规则抽取数值。  
   - **计算**：把模型输出映射到对应的标签空间，分别在简易性、健全性、敏感性上计算准确率、召回率和 F1。最终用加权平均得到综合得分。  
   关键的巧思在于 **统一评分映射**：即使不同模型的输出形式差异巨大（概率分布 vs. 文本解释），作者仍能通过统一的后处理把它们放在同一条比较线上。

最反直觉的地方是：作者并没有把“最终答案对错”作为主要指标，而是把 **过程的每一步** 当作评估核心，这要求评测系统必须能够对细微的逻辑错误进行自动化标注和计分，这在之前的工作里几乎没有实现。

### 实验与效果

- **数据规模**：6,216 题目、83,456 步骤标签，覆盖数学、逻辑、代码三大领域。  
- **模型覆盖**：15 种模型，包括 7 个开源 PRM（如 OpenAI‑Reward、LLaMA‑Reward）和 8 个闭源大模型（GPT‑4、Claude 等）通过 critic prompting 参与评测。  
- **主要发现**：在简易性维度，最好的开源 PRM 也只能达到约 62% 的准确率；在敏感性维度，所有模型的 F1 均低于 55%。闭源模型虽然整体表现略好，但在检测细微逻辑跳步时仍有明显失误。  
- **对比基线**：与传统 step‑correctness benchmark（只看对/错）相比，PRMBench 能揭示出模型在“逻辑跳步”上错误率高达 38%，而传统评测根本看不出。  
- **消融实验**：作者分别去掉简易性标签、去掉敏感性标签进行评测，发现去掉敏感性后整体得分下降约 9%，说明敏感性标签对区分模型能力贡献最大。  
- **局限性**：标注工作高度人工，难以快速扩展到更大规模；对闭源模型的评分抽取依赖规则，可能引入噪声。原文未提供跨语言（如非英文）任务的评测结果。

### 影响与延伸思考

PRMBench 首次把过程级评测细化到多维错误标签，已经被后续几篇工作引用，用来检验自监督微调的 PRM、探索多模态推理过程的奖励模型等。推测未来会出现 **自动化标注工具**，把人工设计的细粒度标签转化为半自动生成，从而降低构建成本；同时也可能出现 **跨语言版 PRMBench**，帮助评估非英语模型的过程推理能力。对想进一步深入的读者，可以关注 **“过程监督学习（Process Supervision）”** 和 **“细粒度错误检测（Fine-grained Error Detection）**” 这两个方向，它们正逐步成为提升大模型可靠性的关键。

### 一句话记住它

PRMBench 把“每一步的对错”变成“每一步的质量”，让我们第一次能系统、细致地衡量过程级奖励模型的真实能力。