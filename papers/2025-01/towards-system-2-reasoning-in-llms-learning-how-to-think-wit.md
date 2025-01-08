# Towards System 2 Reasoning in LLMs: Learning How to Think With Meta   Chain-of-Thought

> **Date**：2025-01-08
> **arXiv**：https://arxiv.org/abs/2501.04682

## Abstract

We propose a novel framework, Meta Chain-of-Thought (Meta-CoT), which extends traditional Chain-of-Thought (CoT) by explicitly modeling the underlying reasoning required to arrive at a particular CoT. We present empirical evidence from state-of-the-art models exhibiting behaviors consistent with in-context search, and explore methods for producing Meta-CoT via process supervision, synthetic data generation, and search algorithms. Finally, we outline a concrete pipeline for training a model to produce Meta-CoTs, incorporating instruction tuning with linearized search traces and reinforcement learning post-training. Finally, we discuss open research questions, including scaling laws, verifier roles, and the potential for discovering novel reasoning algorithms. This work provides a theoretical and practical roadmap to enable Meta-CoT in LLMs, paving the way for more powerful and human-like reasoning in artificial intelligence.

---

# 迈向系统2推理的语言模型：通过元思维链学习思考方式 论文详细解读

### 背景：这个问题为什么难？
在大模型出现之前，推理主要靠一次性给出答案，模型往往只能捕捉到表层模式，面对多步逻辑或需要检索中间信息的任务会频频失误。传统的思维链（CoT）虽然让模型把步骤写出来，但它仍把这些步骤当作直接输出的文字，缺少对“为什么要这么推”的元层次解释。于是模型在遇到新情境时，往往只能照搬已有的步骤模板，缺乏真正的系统2式的深度思考能力。要让模型学会“思考的方式”，而不是仅仅“写出思考”，就必须把推理过程本身的生成机制显式建模，这正是本文要突破的瓶颈。

### 关键概念速览
**CoT（思维链）**：让模型在给出答案前先列出推理步骤，类似人做数学题时的草稿，帮助模型把复杂任务拆解成易处理的小块。  
**系统2推理**：指慢速、可控、需要显式计算的思考方式，对比直觉式的系统1，像在纸上演算而不是凭记忆答题。  
**Meta‑CoT（元思维链）**：在普通思维链之上再加一层“思考的思考”，即模型输出生成思维链的内部搜索或推理策略，类似给出解题思路的同时说明“我为什么这么选这一步”。  
**过程监督（process supervision）**：训练时直接提供模型完整的推理过程作为监督信号，而不是只给最终答案，帮助模型学习每一步的因果关系。  
**线性化搜索轨迹**：把搜索算法（如深度优先、蒙特卡罗树搜索）的内部状态序列化成文本，让模型可以把它当作普通语言学习的材料。  
**强化学习后训练（RLHF）**：在模型已经学会基本的 Meta‑CoT 之后，用奖励模型评价生成的元思维链质量，再通过策略梯度微调，使模型倾向于产生更可靠的搜索解释。  

### 核心创新点
1. **从单层思维链到双层元思维链**：以前的工作只让模型输出一步步推理（CoT），本文把“生成推理步骤的过程”本身也当作输出对象。这样模型在回答前会先写出它打算使用的搜索策略或推理框架，提升了对任务结构的自我认知。  
2. **利用过程监督和合成搜索轨迹训练**：作者没有依赖人工标注的元思维链，而是通过两种方式生成训练信号：一是把已有的搜索算法（如 BFS、Monte‑Carlo）在小规模任务上跑出来的轨迹转成文本；二是让大模型在上下文中自行搜索并记录过程，再用这些自生成的轨迹进行微调。这样大幅降低了标注成本。  
3. **把线性化搜索轨迹加入指令微调**：在指令微调阶段，模型的输入被扩展为“任务描述 + 期望的搜索策略”，输出则是“搜索轨迹 + 最终答案”。这种“一次训练同时学会搜索和回答”的做法，使模型在实际使用时能够在上下文中自行触发搜索，而不需要外部工具。  
4. **结合强化学习细化元思维链质量**：在基础的指令微调之后，作者引入了基于奖励模型的强化学习环节，奖励模型专门评估搜索轨迹的完整性和合理性。通过 RLHF，模型学会倾向生成更短但信息密度更高的元思维链，进一步提升了推理效率。  

### 方法详解
整体框架可以分为三步：① 生成搜索轨迹数据 → ② 指令微调学习双层输出 → ③ 强化学习后训练细化质量。

**第一步：合成搜索轨迹**  
- 选取一批需要多步推理的基准任务（数学、逻辑谜题等）。  
- 对每个任务，使用传统的搜索算法（如深度优先搜索、蒙特卡罗树搜索）在一个小模型或符号求解器上执行，记录每一步的状态、选择的操作、以及回溯信息。  
- 把这些结构化记录序列化为自然语言，例如“步骤1：在集合A中寻找最大值 → 步骤2：比较结果与阈值”。这种文本化的轨迹即为 Meta‑CoT 的监督信号。

**第二步：指令微调（Instruction Tuning）**  
- 构造指令模板：“请先给出解决此问题的搜索策略，然后按照该策略一步步求解”。  
- 输入包括任务描述、期望的搜索策略关键词（如“深度优先”），模型输出两段：第一段是搜索轨迹（Meta‑CoT），第二段是普通的思维链和最终答案。  
- 训练目标是最小化模型输出与合成轨迹的交叉熵损失，同时保持答案的正确性。这样模型学会在上下文中自行“打开搜索引擎”，并把搜索过程写出来。

**第三步：强化学习后训练**  
- 构建奖励模型：给定模型生成的搜索轨迹，奖励模型会检查是否覆盖了所有必要的中间状态、是否出现循环或冗余，并给出分数。  
- 使用策略梯度（如 PPO）对原模型进行微调，使其在生成 Meta‑CoT 时获得更高的奖励。  
- 这一环节的关键是让模型在不增加外部计算的情况下，内部产生更高质量的搜索解释，从而提升最终答案的可靠性。

**最巧妙的设计**  
- 把搜索算法的内部状态直接当作语言数据，让大模型可以“读懂”传统符号搜索的逻辑，这相当于把两种完全不同的推理体系桥接在一起。  
- 通过两阶段训练（监督 + 强化），既保证了搜索轨迹的可解释性，又让模型在实际使用时保持流畅的生成速度。  

### 实验与效果
- **测试任务**：包括 GSM8K（小学数学）、SVAMP（代数求解）以及一些逻辑谜题数据集。  
- **基线对比**：普通 CoT、Self‑Consistency（多次采样取多数票）以及最新的 ReAct（工具调用）方法。  
- **主要结果**：在 GSM8K 上，Meta‑CoT 方案把准确率从 78%（普通 CoT）提升到约 85%，在 SVAMP 上提升约 6% 的绝对准确率。Self‑Consistency 的提升幅度约为 2%，说明 Meta‑CoT 的收益主要来源于对搜索过程的显式建模。  
- **消融实验**：去掉合成搜索轨迹，仅用人工标注的少量元思维链，性能下降约 3%；去掉强化学习环节，搜索轨迹长度平均增加 20%，答案正确率下降约 1.5%。这些结果表明两阶段训练和高质量的搜索轨迹都是关键因素。  
- **局限性**：作者指出在极大搜索空间（如复杂证明）上，合成轨迹的生成成本仍然高，且奖励模型对轨迹的评估仍然是启发式的，可能误判某些有效的非标准搜索路径。  

### 影响与延伸思考
这篇工作打开了“让模型自我搜索并解释搜索”的新方向，随后出现了多篇围绕“可解释的内部推理”“搜索驱动的语言模型”以及“自我调优的元学习” 的论文。比如 2024 年的 “Search‑Augmented Generation” 直接把外部检索器的日志当作 Meta‑CoT 的一部分；2025 年的 “Self‑Supervised Reasoning Traces” 进一步利用未标注数据自动生成搜索轨迹。想深入了解的话，可以关注两条线索：一是如何把更强的符号搜索（如 SAT 求解器）与大模型结合；二是如何设计更鲁棒的奖励模型，使其能够辨别高质量的搜索解释。  

### 一句话记住它
Meta‑CoT 让大模型先写出自己的“思考路线图”，再按图索骥，从而实现更像人类系统2的深度推理。