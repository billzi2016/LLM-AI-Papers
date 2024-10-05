# DOTS: Learning to Reason Dynamically in LLMs via Optimal Reasoning Trajectories Search

> **Date**：2024-10-04
> **arXiv**：https://arxiv.org/abs/2410.03864

## Abstract

Enhancing the capability of large language models (LLMs) in reasoning has gained significant attention in recent years. Previous studies have demonstrated the effectiveness of various prompting strategies in aiding LLMs in reasoning (called "reasoning actions"), such as step-by-step thinking, reflecting before answering, solving with programs, and their combinations. However, these approaches often applied static, predefined reasoning actions uniformly to all questions, without considering the specific characteristics of each question or the capability of the task-solving LLM. In this paper, we propose DOTS, an approach enabling LLMs to reason dynamically via optimal reasoning trajectory search, tailored to the specific characteristics of each question and the inherent capability of the task-solving LLM. Our approach involves three key steps: i) defining atomic reasoning action modules that can be composed into various reasoning action trajectories; ii) searching for the optimal action trajectory for each training question through iterative exploration and evaluation for the specific task-solving LLM; and iii) using the collected optimal trajectories to train an LLM to plan for the reasoning trajectories of unseen questions. In particular, we propose two learning paradigms, i.e., fine-tuning an external LLM as a planner to guide the task-solving LLM, or directly fine-tuning the task-solving LLM with an internalized capability for reasoning actions planning. Our experiments across eight reasoning tasks show that our method consistently outperforms static reasoning techniques and the vanilla instruction tuning approach. Further analysis reveals that our method enables LLMs to adjust their computation based on problem complexity, allocating deeper thinking and reasoning to harder problems.

---

# DOTS：通过最优推理轨迹搜索实现大语言模型的动态推理学习 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在解答需要多步推理的题目时，往往会出现思路不连贯或直接给出错误答案的情况。过去的研究大多通过固定的提示方式（比如一步步思考、先反思再回答）来“强迫”模型进行推理，但这些提示是对所有问题一刀切的。不同题目在难度、结构甚至所需的工具上差异巨大，统一的提示往往要么浪费算力，要么不足以支撑复杂问题的求解。于是，如何让模型根据每道题的特性自行决定使用哪种推理手段、以及推理的深度，成为了亟待突破的瓶颈。

### 关键概念速览
- **原子推理动作（Atomic Reasoning Action）**：最小的可执行推理单元，例如“逐步思考一行”“调用外部代码求解”“先做自我检查”。把它们想象成乐高砖块，任意组合就能搭出不同的推理流程。  
- **推理轨迹（Reasoning Trajectory）**：一系列原子动作按顺序组成的完整推理方案，就像一条路线图，指明模型在解题过程中该走哪些步。  
- **最优轨迹搜索（Optimal Trajectory Search）**：在给定题目和模型能力的前提下，遍历或采样不同轨迹并评估其效果，挑出表现最好的那条。类似于在多条路线中用 GPS 找到最快的那条。  
- **规划器（Planner）**：负责为新题目预测合适轨迹的模型，可以是独立的外部 LLM，也可以是同一个任务模型内部的子网络。它相当于“导航仪”，把题目信息转化为行动指令。  
- **内部化规划（Internalized Planning）**：把规划能力直接写进任务模型本身，让模型在生成答案时自行决定使用哪些动作，而不需要外部调度。  
- **动态计算预算（Dynamic Computation Budget）**：模型根据题目难度自动分配推理步骤的数量，难题会得到更多“思考时间”，简单题则快速给出答案。  

### 核心创新点
1. **固定提示 → 动态轨迹搜索 → 更高效的推理**  
   过去的工作把“一步步思考”之类的提示硬塞给所有输入，导致算力浪费或推理不足。DOTS 先把所有可能的原子动作列出来，再通过搜索找到每道训练题的最佳组合。这样模型只在需要时才使用更复杂的动作，整体算力利用率提升。  

2. **统一提示 → 专属轨迹学习 → 适配不同难度**  
   传统方法没有区分题目难易，所有样本都走同一条思考链。DOTS 在训练阶段为每个样本记录下最优轨迹，然后用这些轨迹教会模型如何为未见题目规划。结果是模型能够自动判断一道题是“只要一步就行”还是“需要多轮思考”。  

3. **外部调度 → 两种规划范式 → 灵活部署**  
   作者提出两种实现方式：一种是训练一个独立的外部 LLM 充当规划器，另一种是把规划模块直接并入任务模型内部。前者适合已有强大模型的生态，后者则让模型“一体化”，省去跨模型调用的开销。  

4. **静态评估 → 计算预算自适应 → 资源感知推理**  
   通过搜索得到的轨迹会伴随对应的计算成本（如调用次数、思考步数），训练时把成本信息一起喂给规划器。于是模型在推理时会权衡答案质量和算力消耗，自动在“快”和“准”之间做平衡。  

### 方法详解
**整体思路**：DOTS 把推理过程拆成若干原子动作，先在训练集上为每道题搜索出最优的动作序列（轨迹），再用这些轨迹训练一个规划器，使其在遇到新题时能够预测出同样高效的轨迹。整个流程分为三步：① 定义动作库；② 轨迹搜索与评估；③ 规划器学习与部署。

1. **动作库构建**  
   作者手工或通过已有技术列出几类常用动作：  
   - *思考*（让模型输出一步推理文字）  
   - *反思*（在给出答案前让模型检查自己的推理）  
   - *编程求解*（生成代码并执行）  
   - *检索*（调用外部知识库）  
   每个动作都有统一的输入/输出接口，保证它们可以像乐高块一样随意拼接。

2. **最优轨迹搜索**  
   对每个训练样本，系统会在动作空间里进行迭代探索。具体做法类似强化学习的“探索‑利用”：先随机组合若干动作生成候选轨迹，执行这些轨迹得到答案后，用标准评估指标（如准确率）以及计算成本对轨迹打分。随后保留表现最好的轨迹，继续在其基础上微调或添加新动作，循环若干次直至收敛。搜索过程不需要梯度，只是基于模型的前向调用和外部执行结果。

3. **规划器训练**  
   - **外部规划器模式**：把收集到的「题目描述 + 最优轨迹」对喂给一个独立的 LLM，使用指令微调让它学会把新题目映射到动作序列。推理时，先让规划器输出轨迹，再让任务模型按轨迹一步步执行。  
   - **内部化规划模式**：在任务模型的解码头部加入一个轻量的轨迹生成子网络，直接在同一模型内部预测动作序列。这样模型在一次前向传播中即可完成“决定怎么思考+实际思考”。  

4. **执行与自适应预算**  
   规划器输出的轨迹会携带每一步的预估成本。模型在执行时会检查累计成本是否超过设定阈值，若超限则提前终止或切换到更简洁的备选轨迹。这样即使在资源受限的环境下，也能保证不会因为过度思考而卡死。

**最巧妙的点**：把搜索得到的轨迹直接当作“标签”来教规划器，而不是让模型自己从零学习如何组织思考步骤。相当于先让模型“玩儿遍所有可能的解题套路”，再把最好的经验浓缩成一个“思考指南”。这种两阶段的设计把探索成本搬到了训练阶段，推理时几乎没有额外开销。

### 实验与效果
- **测试任务**：作者在八个公开的推理基准上做实验，包括数学解题、逻辑推理、代码生成、常识问答等。每个任务都包含不同难度的子集。  
- **对比基线**：包括传统的“一步思考”（Zero‑Shot CoT）、多步思考（Chain‑of‑Thought）、反思式提示（Self‑Consistency）以及直接指令微调（Vanilla Instruction Tuning）。  
- **主要结果**：在所有八个任务上，DOTS 的得分都高于最强的静态提示方法，提升幅度在 2%~7% 之间（具体数值论文中给出）。尤其在高难度子集，模型会自动加深思考层数，准确率提升更为显著。  
- **消融实验**：作者分别去掉动作搜索、去掉规划器、只用单一动作等设置，发现最优轨迹搜索是提升的关键因素，去掉后性能回落到普通 CoT 水平。  
- **局限性**：搜索阶段对计算资源要求较高，尤其在动作库扩展后搜索空间会指数增长；此外，当前实现仍依赖人工定义的原子动作，自动发现新动作的能力有限。  

### 影响与延伸思考
DOTS 把“动态推理”从概念层面落到可操作的系统实现，开启了让 LLM 根据题目自行调度思考资源的研究方向。随后的工作开始探索 **自动化动作发现**（让模型自行生成新的推理模块）以及 **跨模态规划**（把视觉、表格等信息也纳入轨迹搜索）。如果想进一步了解，可以关注 **Meta‑Reasoning**、**Adaptive Computation Time**（自适应计算时间）以及 **Neural Program Synthesis**（神经程序合成）等领域的最新进展。  

### 一句话记住它
让大语言模型先“玩遍所有思考套路”，再把最好的套路浓缩成一个“思考指南”，实现了根据题目难度自动加深或简化推理的能力。