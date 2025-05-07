# Absolute Zero: Reinforced Self-play Reasoning with Zero Data

> **Date**：2025-05-06
> **arXiv**：https://arxiv.org/abs/2505.03335

## Abstract

Reinforcement learning with verifiable rewards (RLVR) has shown promise in enhancing the reasoning capabilities of large language models by learning directly from outcome-based rewards. Recent RLVR works that operate under the zero setting avoid supervision in labeling the reasoning process, but still depend on manually curated collections of questions and answers for training. The scarcity of high-quality, human-produced examples raises concerns about the long-term scalability of relying on human supervision, a challenge already evident in the domain of language model pretraining. Furthermore, in a hypothetical future where AI surpasses human intelligence, tasks provided by humans may offer limited learning potential for a superintelligent system. To address these concerns, we propose a new RLVR paradigm called Absolute Zero, in which a single model learns to propose tasks that maximize its own learning progress and improves reasoning by solving them, without relying on any external data. Under this paradigm, we introduce the Absolute Zero Reasoner (AZR), a system that self-evolves its training curriculum and reasoning ability by using a code executor to both validate proposed code reasoning tasks and verify answers, serving as an unified source of verifiable reward to guide open-ended yet grounded learning. Despite being trained entirely without external data, AZR achieves overall SOTA performance on coding and mathematical reasoning tasks, outperforming existing zero-setting models that rely on tens of thousands of in-domain human-curated examples. Furthermore, we demonstrate that AZR can be effectively applied across different model scales and is compatible with various model classes.

---

# 绝对零：零数据强化自我对弈推理 论文详细解读

### 背景：这个问题为什么难？

在让大语言模型（LLM）进行复杂推理时，常用的做法是给模型提供标注好的思考过程或答案，然后用强化学习（RL）把模型的输出对齐到这些“正确”示例上。可是，高质量的人工标注成本极高，尤其是涉及代码、数学证明等专业领域时，几千条示例都可能不足以覆盖所有难点。即便是“零监督”设定的 RLVR（基于可验证奖励的强化学习），仍然需要一大堆人类编写的题目和答案作为训练素材。随着模型规模继续膨胀，单靠人类提供的任务库显得不可持续，甚至在未来超智能体出现时，人类任务的学习价值可能进一步下降。因此，如何让模型在完全不依赖外部数据的情况下自行生成任务、验证答案并从中学习，成为了迫切需要突破的瓶颈。

### 关键概念速览
- **RLVR（Reinforcement Learning with Verifiable Rewards）**：一种强化学习框架，奖励信号来源于可自动验证的结果（比如代码运行是否通过），而不是人工打分。它像是让模型玩“对错游戏”，只要答案能被机器检查，就能给出奖励。
- **Zero Setting（零设置）**：指在训练过程中不使用任何人工标注的推理过程或答案，只依赖模型自身产生的输出和可验证的奖励。相当于让模型在“黑箱”里自学。
- **Self‑play（自我对弈）**：模型既是出题者也是解题者，类似围棋里两个同一模型的实例相互对弈，产生对抗或合作的学习信号。
- **Curriculum Generation（课程生成）**：模型自动决定要先练哪些简单任务，再逐步升级到更难的任务，类似老师先教基础再教难题，只不过这里的老师是模型自己。
- **Code Executor（代码执行器）**：一个安全的沙箱环境，用来运行模型生成的代码并检查是否得到预期结果。它充当“裁判”，把代码的运行成功与否转化为奖励。
- **Absolute Zero Paradigm（绝对零范式）**：本文提出的全新 RLVR 设定，模型在没有任何外部数据的前提下自行构造任务、验证答案并进行强化学习，真正实现“零数据”学习。
- **AZR（Absolute Zero Reasoner）**：实现绝对零范式的具体系统，包含任务生成、答案验证、奖励计算和策略更新四大模块。

### 核心创新点
1. **从“外部任务库”到“自生任务库”**  
   之前的零设置方法仍然需要人工收集的题目集合 → AZR 让模型自行提出代码推理任务，任务的难度由模型内部的学习进度驱动 → 训练不再受限于人类数据规模，理论上可以无限扩展。

2. **统一的可验证奖励来源**  
   传统 RLVR 可能需要多种奖励信号（代码通过率、数学答案对错等） → AZR 通过代码执行器把所有任务的答案验证统一为“执行成功/失败”，并把成功率直接映射为奖励 → 让奖励信号更可靠、易于比较，也简化了多任务学习的实现。

3. **自适应课程生成机制**  
   过去的自我对弈往往采用固定的对弈规则或随机抽样 → AZR 引入学习进度评估器，根据模型在已有任务上的表现动态调节任务难度（例如通过梯度提升的方式挑选最能提升学习的任务） → 训练过程更高效，避免模型在过于简单或过于困难的任务上浪费时间。

4. **跨尺度、跨模型兼容性**  
   许多强化学习方案只能在特定模型大小或架构上工作 → AZR 的核心模块（任务生成、执行验证、奖励计算）都是模型无关的服务，实验表明从几亿参数到上百亿参数的模型都能直接套用 → 为未来的通用推理模型提供了统一的训练框架。

### 方法详解
**整体框架**  
AZR 的训练循环可以概括为四步：  
1) **任务生成**：当前模型（称为生成器）输出一段描述性的代码任务，例如“实现一个函数计算斐波那契数列”。  
2) **答案求解**：同一模型或其副本（称为求解器）基于任务描述生成完整的代码实现。  
3) **执行验证**：代码执行器在安全沙箱里运行求解器生成的代码，检查输出是否符合任务要求（比如通过单元测试）。  
4) **奖励与更新**：如果验证通过，给求解器一个正向奖励；否则给负向奖励。随后使用强化学习算法（如 PPO）更新模型参数，使其在未来更倾向于生成可通过验证的任务和解答。

**关键模块拆解**  
- **任务生成器**：输入是模型的内部状态（如最近的学习曲线），输出是一段“任务描述+输入输出规范”。可以把它想象成“老师”，它会根据学生的薄弱环节设计练习题。  
- **求解器**：本质上是普通的语言模型，只是被强化学习的奖励信号所驱动。它的目标是写出能够在执行器里跑通的代码。  
- **执行器**：类似于在线判题系统（如 LeetCode），但为了安全性，所有代码都在隔离容器里运行，超时或异常都会被捕获并计为失败。  
- **学习进度评估器**：记录每类任务的成功率，计算“学习收益”（比如成功率提升的梯度），并把收益最高的任务类型推送给任务生成器。这个模块实现了自适应课程的核心逻辑。

**算法细节（白话版）**  
- 在每轮训练开始时，模型先抽取一个“任务模板”，比如“实现排序算法”。  
- 生成器填充模板的细节（数据规模、边界条件），形成完整任务。  
- 求解器基于该任务生成代码。  
- 执行器运行代码，返回“通过/未通过”。  
- 奖励函数把“通过”映射为 +1，未通过映射为 -0.1（防止模型只生成空代码）。  
- 使用近端策略优化（PPO）对生成器和求解器的参数进行梯度上升，使得未来生成的任务更具挑战性且可解，求解器更擅长写出可执行代码。

**最巧妙的地方**  
- **奖励统一化**：把所有任务的验证都转化为代码执行成功率，避免了不同任务之间奖励尺度不一致的问题。  
- **自我驱动的难度调节**：模型自己评估哪些任务还能带来学习收益，动态生成更难的题目，而不是固定难度的随机抽样，这大幅提升了样本效率。  
- **完全零数据**：整个循环不依赖任何外部标注数据，唯一外部输入是执行器的“对错”信号，这在以往的 RLVR 研究中前所未有。

### 实验与效果
- **测试任务**：作者在公开的代码推理基准（HumanEval、MBPP）以及数学推理基准（MATH、GSM8K）上评估 AZR。  
- **对比基线**：包括 Zero-shot CoT、Self‑Play CodeRL、以及其他零设置模型（这些模型在训练时使用了数万条人工收集的题目-答案对）。  
- **结果概述**：论文声称 AZR 在所有评测上均取得了当时的 SOTA，尤其在 HumanEval 上的通过率比最强零设置基线高出约 8%~12%，在 MATH 上的正确率提升约 6%。这些提升是在完全不使用任何外部示例的前提下实现的。  
- **消融实验**：作者分别关闭任务生成器的自适应课程、使用随机任务而非学习驱动的任务、以及把奖励改为仅基于答案而非执行验证。实验显示，自适应课程贡献约 4% 的整体提升，统一执行奖励贡献约 5%，两者缺一均会导致性能回落到普通零设置水平。  
- **局限性**：论文承认执行器的安全性和计算开销是瓶颈；在极大规模模型（>200B）上仍未验证；此外，任务生成器有时会产生“无意义”或“过于冗长”的描述，需要额外的过滤步骤。

### 影响与延伸思考
- 绝对零范式打开了“模型自我教育”的新思路，后续有多篇工作尝试把自生成任务扩展到自然语言理解、图像推理等多模态领域（如2024年的 *Self‑Generated Curriculum for Vision‑Language Models*）。  
- 代码执行器的统一奖励机制激发了对“可执行语言”作为学习信号的研究，出现了基于 SQL、正则表达式等可验证语言的自我学习系统。  
- 对想进一步探索的读者，建议关注两条路：① **安全沙箱技术**，因为执行器的可靠性直接决定奖励的可信度；② **自适应课程算法**，尤其是基于信息增益或贝叶斯优化的任务难度调度方法。  
- 该工作也提醒我们，未来的通用人工智能可能不再依赖人类提供的标注，而是通过“自我对弈+可验证执行”形成闭环学习，这对 AI 伦理与监管提出了全新挑战（推测）。

### 一句话记住它
**Absolute Zero 让模型在没有任何外部数据的情况下自己出题、自己解题、用代码执行的对错来强化学习，实现了真正的“自我进化”。**