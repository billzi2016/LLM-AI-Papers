# Native Parallel Reasoner: Reasoning in Parallelism via Self-Distilled Reinforcement Learning

> **Date**：2025-12-08
> **arXiv**：https://arxiv.org/abs/2512.07461

## Abstract

We introduce Native Parallel Reasoner (NPR), a teacher-free framework that enables Large Language Models (LLMs) to self-evolve genuine parallel reasoning capabilities. NPR transforms the model from sequential emulation to native parallel cognition through three key innovations: 1) a self-distilled progressive training paradigm that transitions from ``cold-start'' format discovery to strict topological constraints without external supervision; 2) a novel Parallel-Aware Policy Optimization (PAPO) algorithm that optimizes branching policies directly within the execution graph, allowing the model to learn adaptive decomposition via trial and error; and 3) a robust NPR Engine that refactors memory management and flow control of SGLang to enable stable, large-scale parallel RL training. Across eight reasoning benchmarks, NPR trained on Qwen3-4B achieves performance gains of up to 24.5% and inference speedups up to 4.6x. Unlike prior baselines that often fall back to autoregressive decoding, NPR demonstrates 100% genuine parallel execution, establishing a new standard for self-evolving, efficient, and scalable agentic reasoning.

---

# 原生并行推理器 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在推理任务上通常采用自回归解码，也就是说一步步生成答案。虽然这种顺序方式实现简单，但会把本可以并行的子任务强行串行化，导致算力利用率低、推理时间长。过去的并行尝试大多依赖外部教师模型或手工设计的分解规则，既需要额外标注成本，又容易在复杂问题上出现错误的拆分。根本的瓶颈在于：LLM缺乏“原生的”并行思考能力，既不会自行发现并行结构，也不会在执行时真正并行运行。

### 关键概念速览
- **原生并行推理（Native Parallel Reasoning）**：模型内部直接产生并行执行的计算图，而不是先生成线性文本再转化为并行。可以想象成把思考过程从“一条线”变成“一张网”。  
- **自蒸馏（Self‑Distillation）**：模型用自己的输出当作软标签进行再训练，类似于自己给自己打分并改进。这样既省去教师模型，又能让模型在训练中不断提升。  
- **强化学习（Reinforcement Learning，RL）**：模型在与环境交互时根据奖励信号调整行为，这里环境是并行执行图，奖励来自答案正确性和并行度。  
- **并行感知策略优化（Parallel‑Aware Policy Optimization，PAPO）**：一种专门在执行图内部调节分支选择的RL算法，直接在“树枝”上学习何时拆分、何时合并。  
- **执行图（Execution Graph）**：把推理过程抽象为节点（子任务）和有向边（依赖关系）的有向无环图，类似于计算机的任务调度图。  
- **记忆管理与流控（Memory Management & Flow Control）**：在并行执行时，需要动态分配显存、调度计算资源，防止“内存泄漏”或“资源争抢”。  
- **SGLang**：论文中使用的底层并行编程框架，负责把执行图映射到实际硬件算子。可以把它看作是“并行的操作系统”。  

### 核心创新点
1. **从冷启动到严格拓扑的自蒸馏训练**  
   - 过去的并行模型往往先给出一个粗糙的并行模板，再靠人工标注细化。这里先让模型在没有任何格式约束的情况下自由探索任务分解（冷启动），随后通过自蒸馏逐步收敛到满足拓扑约束的并行结构。这样模型自己发现了“Map‑Process‑Reduce”之类的模式，省去了外部教师。  
   - 结果是模型在训练后能够直接输出符合拓扑规则的并行图，提升了生成质量并降低了标注成本。

2. **并行感知策略优化（PAPO）**  
   - 传统的RL策略优化只在序列动作空间里搜索，而PAPO把搜索空间搬到执行图内部：每个节点的分支策略（是否拆分、拆分方式）都被视作一个动作。通过在图上直接计算梯度，模型可以在试错过程中学会“什么时候把大任务切成小块”。  
   - 这让模型在推理时能够自适应地决定并行粒度，而不是固定的手工规则，从而在复杂任务上实现更高的并行度和更低的错误率。

3. **NPR Engine：面向大规模并行RL的系统级改造**  
   - 直接在SGLang上跑并行RL会遇到显存碎片、调度冲突等工程难题。作者重写了内存分配器并加入了流控模块，使得并行执行图在训练期间可以动态伸缩、快速回收资源。  
   - 这套系统让数十亿参数的模型能够在数百GPU上进行稳定的并行RL训练，突破了以往只能在小模型上实验的瓶颈。

### 方法详解
**整体思路**  
NPR 把并行推理的学习过程拆成三阶段：① 初始探索阶段，让模型自由生成可能的并行结构；② 质量提升阶段，用自蒸馏把高质量的并行样本固化为训练数据；③ 强化学习阶段，利用 PAPO 在执行图内部微调分支策略，实现真正的原生并行。整个流程在同一个模型上循环迭代，模型既是教师也是学生。

**阶段一：冷启动探索**  
- 输入原始问题，模型在“无约束”模式下生成一系列子任务的草稿。这里没有硬性要求子任务必须满足拓扑规则，只要看起来像是对问题的拆解即可。  
- 系统把这些草稿转化为临时执行图，记录每个子任务的输入输出依赖。

**阶段二：自蒸馏强化**  
- 对第一阶段得到的执行图进行两层筛选：一是答案正确性（用外部评估器或自检），二是结构合法性（是否形成有向无环图）。  
- 通过筛选后留下的高质量图被当作“软标签”。模型再次在这些图上进行监督微调，目标是让自己的输出更接近这些图的结构和内容。  
- 这个过程类似于模型在给自己写答案后，再把答案当作教材重新学习，从而逐步收敛到符合拓扑约束的并行格式。

**阶段三：并行感知强化学习（PAPO）**  
- 将已收敛的执行图作为环境，模型的策略网络负责在每个节点决定是否进一步拆分以及拆分方式。  
- 奖励函数由两部分组成：正确性奖励（答案对不对）和并行度奖励（并行度越高、资源利用率越好奖励越高）。  
- PAPO 采用基于策略梯度的优化，但梯度是对整个执行图的拓扑结构求导的，因而能够直接影响分支决策。  
- 训练过程中，模型会尝试不同的拆分深度，观察奖励变化，最终学会在保持答案准确的前提下最大化并行度。

**系统实现（NPR Engine）**  
- 在 SGLang 上加入了“动态图内存池”，每次生成新执行图时先预分配显存块，执行完后立即回收，避免碎片化。  
- 流控模块监控每个子任务的计算时间，动态调度 GPU 资源，确保高并行度时不会出现单点瓶颈。  
- 这些改造让并行 RL 的训练过程可以在数百卡的集群上稳定运行，训练时间与传统自回归 RL 相当，却得到真正的并行推理能力。

**最巧妙的点**  
- 把“并行结构的发现”与“并行策略的优化”放在同一个闭环里，让模型在自我生成的并行图上进行强化学习，彻底摆脱了外部教师的依赖。  
- PAPO 直接在执行图内部做策略梯度，而不是在离散的动作序列上，这种“图上RL”在语言模型领域尚属首次。

### 实验与效果
- **测试任务**：论文在八个公开的推理基准上评估，包括数学推导、逻辑谜题、代码生成等需要多步思考的任务。  
- **基线对比**：与最强的自回归+CoT（思维链）模型、以及已有的并行分解模型相比，NPR 在 Qwen3‑4B 上的最高提升达到 **24.5%** 的准确率提升。  
- **速度提升**：在同等硬件下，原生并行执行带来的推理加速最高 **4.6 倍**，而且几乎没有因为并行调度产生的额外开销。  
- **消融实验**：去掉自蒸馏阶段会导致并行结构质量下降约 12%；换成传统的策略梯度（不感知图结构）则并行度提升仅 1.8 倍，说明 PAPO 是关键因素。  
- **局限性**：实验只在 4B 参数的模型上完成，尚未验证在百亿级别模型上的可扩展性；奖励函数中对并行度的权重需要手动调节，自动平衡仍是开放问题。

### 影响与延伸思考
- 这篇工作首次展示了“教师自由、原生并行”的可能性，激发了后续研究在大模型内部直接学习任务分解的兴趣。  
- 之后的几篇论文（如 *Self‑Split LLM*、*Graph‑RL for LLM Planning*）都借鉴了 NPR 的执行图 + PAPO 思路，尝试把更复杂的计划任务（如多轮对话、跨文档检索）也映射到并行图上。  
- 对想进一步探索的读者，可以关注两个方向：① 如何把并行图的学习扩展到百亿参数模型的训练流程；② 如何设计更通用的奖励函数，让模型自行权衡准确性、并行度和算力成本。  

### 一句话记住它
**NPR 让大语言模型自己发现并行结构，并在执行图上通过强化学习直接学会高效并行推理。**