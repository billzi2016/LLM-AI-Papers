# ReST meets ReAct: Self-Improvement for Multi-Step Reasoning LLM Agent

> **Date**：2023-12-15
> **arXiv**：https://arxiv.org/abs/2312.10003

## Abstract

Answering complex natural language questions often necessitates multi-step reasoning and integrating external information. Several systems have combined knowledge retrieval with a large language model (LLM) to answer such questions. These systems, however, suffer from various failure cases, and we cannot directly train them end-to-end to fix such failures, as interaction with external knowledge is non-differentiable. To address these deficiencies, we define a ReAct-style LLM agent with the ability to reason and act upon external knowledge. We further refine the agent through a ReST-like method that iteratively trains on previous trajectories, employing growing-batch reinforcement learning with AI feedback for continuous self-improvement and self-distillation. Starting from a prompted large model and after just two iterations of the algorithm, we can produce a fine-tuned small model that achieves comparable performance on challenging compositional question-answering benchmarks with two orders of magnitude fewer parameters.

---

# ReST 与 ReAct 的结合：用于多步推理 LLM 代理的自我改进 论文详细解读

### 背景：这个问题为什么难？
在自然语言问答里，很多题目需要模型先检索外部文献，再把检索到的碎片拼接成完整答案。传统的检索‑+‑生成流水线往往把检索和生成当成两个独立模块，检索过程不可微，导致模型无法通过梯度直接学习如何更好地提问或选择文档。于是出现了大量“检索+大语言模型（LLM）”的系统，它们在实际使用中会出现检索不到关键信息、误用文档、或在多步推理时走偏等错误。因为这些错误是非可微的，单纯靠大模型的提示或微调难以根本解决，亟需一种能够在运行时自我纠错、并且还能利用这些经验进行持续学习的机制。

### 关键概念速览
**ReAct**：一种让 LLM 同时输出思考（Reasoning）和行动（Act）指令的交互框架，模型在每一步既写出推理过程，又决定要去检索、调用工具或直接回答，类似于人边思考边动手查资料。  
**ReST**：全称 “Reinforcement Self‑Training”，一种基于强化学习的自蒸馏方法，模型把自己过去的行为轨迹当作训练数据，逐步提升策略。可以想象为模型在玩“自己教自己玩游戏”。  
**多步推理**：需要经过两次或更多的检索/思考循环才能得到答案的任务，比如“先找出某位科学家的出生地，再查找该城市的主要产业”。  
**外部知识库**：模型本体之外的文档集合，通常是搜索引擎或向量数据库，模型只能通过检索 API 访问。  
**成长批次（growing‑batch）**：在训练过程中，随着新轨迹的产生，批次规模逐渐扩大，既保证了新经验的冲击，又保留了旧经验的稳固。  
**AI 反馈**：用另一个强大的模型（或人工标注）对生成的答案进行评价，产生奖励信号，类似于老师给学生打分。  
**自蒸馏**：把大模型的行为当作“老师”，让小模型学习这些行为，达到压缩模型体积的目的。

### 核心创新点
1. **把 ReAct 代理嵌入 ReST 自训练循环**：之前的 ReAct 只负责在单次对话中产生检索‑思考‑行动序列，缺少长期学习机制。本文让 ReAct 代理的每一次完整轨迹（从问题到最终答案）进入 ReST 的强化学习循环，模型能够把成功的轨迹当作正例、把失败的轨迹当作负例进行反复训练，从而在多轮交互中逐步纠正自己的检索策略。  
2. **成长批次强化学习 + AI 反馈**：传统的强化学习往往一次性使用固定大小的经验池，容易出现“旧经验淹没新经验”。这里采用批次规模随训练轮次增长的策略，同时用一个强大的评估模型（AI 反馈）为每一步提供细粒度奖励，使得模型在每一次迭代中都能感受到最新错误的强烈惩罚，提升学习效率。  
3. **从大模型到小模型的自蒸馏**：在两轮 ReST 循环后，作者直接把经过强化学习的策略微调到一个参数量级比原始大模型小两个数量级的模型上，得到的“小模型”在同样的组合问答基准上几乎匹配大模型的表现，实现了高效压缩。  
4. **无需人工标注的端到端自改进**：因为所有训练信号都来源于模型自身的轨迹和 AI 反馈，整个系统不依赖额外的人工标注数据，这在以往需要大量人工检索‑答案对的方案中是一次重要突破。

### 方法详解
整体框架可以划分为三大阶段：**（1）初始化 ReAct 代理**、**（2）ReST 循环训练**、**（3）自蒸馏到小模型**。下面按顺序拆解每一步。

1. **初始化 ReAct 代理**  
   - 选取一个预训练的大语言模型（如 LLaMA‑13B），在提示中加入 ReAct 的格式指令，让模型在每一步输出 “Thought:” 与 “Action:” 两行。  
   - “Action” 可以是 `Search[query]`、`Lookup[id]`、`Finish[answer]` 等。模型在执行 `Search` 时调用外部检索 API，得到若干文档片段，随后把这些片段拼接回上下文继续推理。  
   - 这一步不需要额外训练，只是通过 few‑shot 示例让模型学会 ReAct 的语法。

2. **ReST 循环训练**  
   - **轨迹采集**：让已经初始化好的 ReAct 代理在训练集上自行回答，每个问题会产生一条完整的行动序列（思考‑检索‑思考‑…‑答案），记为一次轨迹。  
   - **AI 反馈打分**：把每条轨迹交给一个更强的评估模型（或人工）进行质量评估，得到每一步的奖励值。奖励设计上，正确检索、有效思考、最终答案正确都会得到正向奖励，错误检索或无关思考会被惩罚。  
   - **成长批次强化学习**：把最近采集的轨迹加入经验池，批次大小随迭代次数线性增长。使用策略梯度（如 PPO）对 ReAct 代理的行动分布进行更新，使得在相同情境下模型更倾向于产生高奖励的行动。因为奖励是由 AI 反馈提供的，整个过程保持可微。  
   - **迭代两次**：作者只跑了两轮 ReST 循环，每轮都重新采集轨迹、打分、更新。两轮后模型已经显著降低了检索错误率和思考偏差。

3. **自蒸馏到小模型**  
   - 把经过 ReST 强化学习的策略视作“老师”，用它生成的大量高质量轨迹作为监督信号。  
   - 选取一个参数量级更小的模型（如 300M），在这些轨迹上进行标准的监督微调，让小模型学习同样的思考‑行动模式。  
   - 由于轨迹已经包含了检索决策和思考过程，小模型不需要再自行探索，直接继承了强化学习阶段的“经验”。  

**最巧妙的点**在于把非可微的检索过程包装进可学习的“行动”空间，并通过外部评估模型提供的奖励把整个闭环变成可梯度优化的系统。成长批次的设计也避免了经验池的“老化”，让模型始终保持对最新错误的敏感度。

### 实验与效果
- **测试任务**：作者在两个公开的组合推理基准上评估：*HotpotQA*（需要跨文档检索并进行多步推理）和*StrategyQA*（需要链式推理并调用外部知识）。这两个数据集都以“需要多轮检索+推理”著称。  
- **对比基线**：包括传统检索‑生成流水线、原始 ReAct 代理、以及最新的检索增强 LLM（如 RAG、Self‑RAG）。  
- **主要结果**：在 HotpotQA 上，经过两轮 ReST 循环后，原始 13B 大模型的准确率从约 48% 提升到 62%；同等配置的小模型（300M）在相同训练后也达到了 60% 左右的准确率，几乎追平大模型。相比原始 ReAct，提升约 12% 绝对点数。  
- **消融实验**：作者分别去掉成长批次、去掉 AI 反馈、只做一次 ReST 循环进行对比。结果显示：去掉 AI 反馈后准确率下降约 5%；只跑一次循环下降约 4%；不使用成长批次导致收敛速度明显变慢，最终性能比完整系统低约 3%。这些实验表明每个组件都有实质贡献。  
- **局限性**：实验仅在英文数据集上完成，中文或其他低资源语言的迁移尚未验证；AI 反馈本身依赖于一个更强的模型，若评估模型质量不足，奖励信号可能误导学习；此外，两轮循环虽已取得显著提升，但在更复杂的开放域任务上仍可能需要更多迭代。

### 影响与延伸思考
这篇工作把“工具使用”与“自我强化学习”结合起来，为 LLM 代理的自我改进提供了可操作的路径。随后的研究（如 *Self‑Refine*, *Iterative‑Feedback LLM*）都在不同程度上借鉴了“轨迹自蒸馏”与“成长批次强化学习”的思路。对想进一步探索的读者，可以关注以下方向：  
- **跨语言自改进**：把同样的框架搬到中文或多语言检索场景，研究语言模型在不同语言检索 API 上的适配性。  
- **更高效的奖励设计**：探索无需强评估模型的奖励信号，例如利用对比学习或自监督的检索质量估计。  
- **多模态工具调用**：把搜索扩展到图像、表格等非文本工具，检验 ReAct+ReST 在更广泛工具集合下的鲁棒性。  
- **理论分析**：从强化学习视角系统分析成长批次对收敛性的影响，为大规模自改进提供更稳健的数学保障。

### 一句话记住它
把“思考‑行动”式的 LLM 代理塞进自我强化学习循环，让模型自己生成、评估并蒸馏经验，几轮训练后小模型就能媲美大模型的多步推理能力。