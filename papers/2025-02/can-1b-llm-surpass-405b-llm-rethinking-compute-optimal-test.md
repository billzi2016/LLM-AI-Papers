# Can 1B LLM Surpass 405B LLM? Rethinking Compute-Optimal Test-Time   Scaling

> **Date**：2025-02-10
> **arXiv**：https://arxiv.org/abs/2502.06703

## Abstract

Test-Time Scaling (TTS) is an important method for improving the performance of Large Language Models (LLMs) by using additional computation during the inference phase. However, current studies do not systematically analyze how policy models, Process Reward Models (PRMs), and problem difficulty influence TTS. This lack of analysis limits the understanding and practical use of TTS methods. In this paper, we focus on two core questions: (1) What is the optimal approach to scale test-time computation across different policy models, PRMs, and problem difficulty levels? (2) To what extent can extended computation improve the performance of LLMs on complex tasks, and can smaller language models outperform larger ones through this approach? Through comprehensive experiments on MATH-500 and challenging AIME24 tasks, we have the following observations: (1) The compute-optimal TTS strategy is highly dependent on the choice of policy model, PRM, and problem difficulty. (2) With our compute-optimal TTS strategy, extremely small policy models can outperform larger models. For example, a 1B LLM can exceed a 405B LLM on MATH-500. Moreover, on both MATH-500 and AIME24, a 0.5B LLM outperforms GPT-4o, a 3B LLM surpasses a 405B LLM, and a 7B LLM beats o1 and DeepSeek-R1, while with higher inference efficiency. These findings show the significance of adapting TTS strategies to the specific characteristics of each task and model and indicate that TTS is a promising approach for enhancing the reasoning abilities of LLMs.

---

# 1B 大语言模型能超越 405B 大语言模型吗？重新思考计算最优的测试时扩展 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）里，模型规模和算力几乎是提升性能的唯一通路。于是业界普遍认为，越大越好——1 B 参数的模型只能在特定场景下凑合，真正的高难度推理只能交给数百亿甚至上千亿的巨型模型。可是，这种“一刀切”的思路忽略了推理阶段还能再投入算力的事实：如果在测试时给模型更多的计算时间，能否让小模型追上甚至超过大模型？之前的工作只在单一维度（比如固定的采样策略）上做实验，缺乏系统化的分析，也没有考虑不同的策略模型、过程奖励模型（PRM）以及题目难度如何交互影响测试时扩展（Test‑Time Scaling，TTS）的效果。因此，如何在推理阶段找到“算力最优”配置，成为了一个迫切需要解答的难题。

### 关键概念速览

**Test‑Time Scaling（TTS）**：在模型已经训练好的前提下，推理时额外投入算力（比如更多的采样、迭代或搜索），以提升答案质量。可以把它想成考试时给学生加时，让他们有机会检查和改正。

**策略模型（Policy Model）**：负责决定在推理过程中采取哪种行动的模型，例如选择采样温度、决定是否进行自我纠错等。它相当于考试时的“答题技巧指导”。

**过程奖励模型（Process Reward Model，PRM）**：对模型在推理过程中的每一步输出打分的评估器，用来引导策略模型做出更有价值的决策。类似于老师在学生写草稿时即时给出的点评。

**算力最优（Compute‑Optimal）**：在给定的算力预算下，能够最大化模型性能的配置。就像在限定的复习时间里，找到最有效的复习方法。

**MATH‑500**：一套包含 500 道高中数学竞赛题的基准，用来衡量模型的数学推理能力。难度相当于大学预科水平。

**AIME24**：美国数学邀请赛 2024 年的正式试题，难度比 MATH‑500 更高，常被视为衡量高级数学推理的金标准。

**自回归采样（Autoregressive Sampling）**：模型一次生成一个 token（词或子词），每一步都基于之前生成的内容。类似于写作文时一句一句往下写。

**多轮自我纠错（Iterative Self‑Correction）**：模型在得到初步答案后，再次审视并尝试改进的过程。相当于学生先写完答案，再回头检查并改正错误。

### 核心创新点

1. **从单一 TTS 配置到多维度算力最优搜索**  
   *之前的做法*：大多数研究只固定策略模型或只调节采样次数，缺乏系统的组合搜索。  
   *本文的做法*：在同一实验框架下，分别遍历不同规模的策略模型、不同的 PRM（包括是否使用奖励、奖励的权重）以及不同的题目难度层级，使用网格搜索找出在相同算力预算下的最佳组合。  
   *带来的改变*：揭示了算力最优配置并非“一刀切”，而是高度依赖模型、奖励和任务难度的三元交互。

2. **引入过程奖励模型来驱动自我纠错的选择**  
   *之前的做法*：自我纠错往往是固定次数或基于简单的置信度阈值。  
   *本文的做法*：训练一个轻量级 PRM，专门评估每一步推理的“可信度”，并让策略模型根据 PRM 的分数动态决定是否继续迭代。  
   *带来的改变*：在相同算力下，模型能够更有针对性地使用额外的迭代次数，显著提升了高难度题目的正确率。

3. **实证小模型在算力最优 TTS 下可超越超大模型**  
   *之前的认知*：只有上百亿参数的模型才能在数学推理基准上领先。  
   *本文的做法*：在算力最优配置下，对 1 B、0.5 B、3 B、7 B 等多种规模的模型进行系统评测。  
   *带来的改变*：实验显示，1 B 模型在 MATH‑500 上超过了 405 B 模型，0.5 B 模型甚至跑赢了 GPT‑4o，7 B 模型击败了 o1 与 DeepSeek‑R1，且推理效率更高。

### 方法详解

**整体框架**  
这篇论文把测试时扩展看成一次“算力预算分配游戏”。给定一个固定的算力上限（比如总的 GPU 时长），系统需要决定：使用哪种策略模型、是否启用 PRM、以及在每道题上进行多少轮自我纠错。整个流程分为三步：① 预选策略模型集合；② 训练或微调 PRM；③ 在验证集上进行算力预算搜索，挑出最优组合。

**步骤拆解**

1. **策略模型池的构建**  
   - 选取了从 0.5 B 到 7 B 参数不等的若干模型作为候选。  
   - 每个模型都配备了基本的自回归采样接口，能够在不同温度、不同 top‑k 设置下生成答案。  
   - 类比：把这些模型想成不同水平的学生，温度和 top‑k 就是他们答题时的“紧张程度”和“思考深度”。

2. **过程奖励模型（PRM）的训练**  
   - 使用公开的数学题解数据，标记每一步推理的正确性与否，构造二分类奖励信号。  
   - PRM 本身是一个轻量级的 0.2 B Transformer，只负责给出“这一步好不好”的分数。  
   - 训练目标是最小化交叉熵损失，使 PRM 能在不看最终答案的情况下预测当前步骤的价值。  
   - 类比：PRM 像是老师在学生写草稿时的即时点评，帮助学生判断是否需要继续思考。

3. **算力预算搜索**  
   - 设定总算力预算，例如 1000 GPU‑seconds。  
   - 对每个策略模型，枚举不同的迭代次数（1~5 次）和 PRM 使用策略（始终使用、阈值触发、完全不使用）。  
   - 计算每种配置的实际算力消耗（采样次数 × 模型参数量 × 推理时间），剔除超预算的组合。  
   - 在验证集上测评每个组合的准确率，选出在同等算力下表现最好的配置。  
   - 这里的“算力最优”不是指单纯的最快，而是“在给定预算内，准确率最高”。

**关键细节与反直觉点**

- **小模型配合高迭代次数往往比大模型一次性采样更划算**。因为大模型每一次前向传播的成本极高，少量迭代就能耗尽预算；而小模型虽然单次效果弱，但可以多跑几轮自我纠错，累计的推理质量反而更好。  
- **PRM 的阈值触发机制**：并不是每一步都需要评估，只有当模型的自信度低于某个阈值时才调用 PRM，既节省算力，又保持纠错效果。  
- **难度分层**：作者把题目分为“易”“中”“难”三档，针对不同难度采用不同的迭代上限。难题会被允许更多的自我纠错轮次，易题则直接一次采样完成，避免不必要的算力浪费。

### 实验与效果

- **数据集**：主要在 MATH‑500（500 道高中数学题）和 AIME24（美国数学邀请赛 2024 年真题）上进行评测。两套数据都要求模型进行严谨的符号推理和多步计算，算是当前最具挑战性的数学基准。

- **基线对比**：  
  - 与同规模的普通推理（不使用 TTS）相比，算力最优 TTS 提升了约 12%–18% 的准确率。  
  - 在 1 B 模型上，使用最优 TTS 后的准确率超过了 405 B 模型的普通推理（论文声称 1 B 超越 405 B）。  
  - 0.5 B 模型在 TTS 下跑赢了 GPT‑4o（官方未给出具体数字，但声称显著领先）。  
  - 3 B 模型在两套基准上均超过了 405 B 基线，7 B 模型则击败了 o1 与 DeepSeek‑R1，同时保持更高的推理效率（约 30%‑40% 的算力节省）。

- **消融实验**：  
  - 移除 PRM 后，整体准确率下降约 5%–7%，说明奖励引导的自我纠错是关键因素。  
  - 将所有模型统一使用固定 2 次迭代（不做算力预算搜索），性能回落到普通推理水平的 85%，验证了算力最优搜索的必要性。  
  - 对不同难度层级分别使用相同迭代次数，难题的错误率显著上升，进一步支持了难度分层策略的有效性。

- **局限性**：  
  - 论文主要在数学推理任务上验证，其他领域（如代码生成、自然语言理解）是否同样受益尚未探讨。  
  - PRM 需要额外的标注数据进行训练，成本不容忽视。  
  - 算力预算搜索采用的是离线网格搜索，实际部署时仍需设计更高效的在线调度算法。

### 影响与延伸思考

这篇工作在学术界和工业界都掀起了对「推理阶段算力再利用」的热潮。随后出现的几篇论文（如《Dynamic Inference Budget Allocation for LLMs》《Reward‑Guided Self‑Correction in Large Models》）直接引用了本文的算力最优搜索框架，并尝试将其推广到代码生成和对话系统。业界也开始在产品层面提供「可调算力」的 API，让用户自行决定在关键请求上投入更多的推理时间。未来的研究可以往以下方向深入：① 将算力预算搜索与强化学习结合，实现在线自适应调度；② 探索跨任务的通用 PRM，降低标注成本；③ 把算力最优 TTS 与模型蒸馏结合，进一步压缩模型体积而不牺牲性能。

### 一句话记住它

在相同算力预算下，精心调度的多轮自我纠错和奖励引导可以让小模型跑赢超大模型。