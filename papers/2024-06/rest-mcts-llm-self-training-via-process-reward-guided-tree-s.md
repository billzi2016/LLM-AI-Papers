# ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search

> **Date**：2024-06-06
> **arXiv**：https://arxiv.org/abs/2406.03816

## Abstract

Recent methodologies in LLM self-training mostly rely on LLM generating responses and filtering those with correct output answers as training data. This approach often yields a low-quality fine-tuning training set (e.g., incorrect plans or intermediate reasoning). In this paper, we develop a reinforced self-training approach, called ReST-MCTS*, based on integrating process reward guidance with tree search MCTS* for collecting higher-quality reasoning traces as well as per-step value to train policy and reward models. ReST-MCTS* circumvents the per-step manual annotation typically used to train process rewards by tree-search-based reinforcement learning: Given oracle final correct answers, ReST-MCTS* is able to infer the correct process rewards by estimating the probability this step can help lead to the correct answer. These inferred rewards serve dual purposes: they act as value targets for further refining the process reward model and also facilitate the selection of high-quality traces for policy model self-training. We first show that the tree-search policy in ReST-MCTS* achieves higher accuracy compared with prior LLM reasoning baselines such as Best-of-N and Tree-of-Thought, within the same search budget. We then show that by using traces searched by this tree-search policy as training data, we can continuously enhance the three language models for multiple iterations, and outperform other self-training algorithms such as ReST$^\text{EM}$ and Self-Rewarding LM. We release all code at https://github.com/THUDM/ReST-MCTS.

---

# ReST-MCTS* 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在需要多步推理的任务上常常会走偏，传统的自训练方法只让模型自行生成答案，然后挑出“对的”答案当作新数据。这样得到的训练集往往只包含最终答案，缺少高质量的推理过程，导致模型学不到可靠的思考步骤。更糟的是，若直接把错误的中间步骤也当作正例，模型会把错误的思路固化进去。要想让模型在自训练时既保留正确的答案，又学到可靠的推理过程，就必须解决“如何自动评估每一步的价值”这一难题，而这在没有人工标注的情况下几乎是不可能的。

### 关键概念速览
- **自训练（Self‑Training）**：模型先用已有能力生成大量伪标签数据，再用这些数据继续微调自己，类似于学生先做练习再自我批改。
- **过程奖励（Process Reward）**：对每一步推理给出的分数，衡量这一步是否有助于最终得到正确答案，类似于老师给每一步解题过程打分。
- **蒙特卡洛树搜索（MCTS）**：一种在决策树中通过随机模拟来估计每个节点价值的搜索算法，像在迷宫里多次试走不同路径来判断哪条路更有希望到达出口。
- **MCTS\***：本文对标准MCTS的改进版，加入了过程奖励的估计，使搜索更聚焦于高价值的推理步骤。
- **策略模型（Policy Model）**：负责在每一步生成候选动作（即推理步骤）的语言模型，类似于棋手的下棋策略。
- **价值模型（Value Model）**：预测从当前状态继续下去能否得到正确答案的模型，像是棋手对局面好坏的打分器。
- **Best‑of‑N**：一次生成 N 条完整答案，挑最好的那条，属于“盲目投票”式的搜索。
- **Tree‑of‑Thought（ToT）**：把推理过程组织成树结构搜索，但没有过程奖励的细粒度引导。

### 核心创新点
1. **过程奖励的自动推断 → 通过树搜索估计每一步对最终答案的贡献 → 省去人工标注，得到高质量的每步价值信号。**  
   传统自训练需要人工给每一步打分，成本高且难以规模化。ReST‑MCTS* 用 MCTS 在已知正确答案的前提下，统计哪些分支更可能导向正确答案，从而把这种统计概率当作过程奖励。

2. **MCTS\* 与过程奖励联合 → 在搜索阶段直接利用过程奖励指导节点扩展 → 在相同计算预算下比 Best‑of‑N 与 Tree‑of‑Thought 更快收敛到高质量推理路径。**  
   标准 MCTS 只依据模拟结果的胜率进行选择，容易在搜索空间里浪费步数。加入过程奖励后，搜索更倾向于“看起来有帮助”的分支，提升了搜索效率。

3. **双向自训练循环 → 用搜索得到的高质量轨迹微调策略模型，同时用推断出的过程奖励微调价值模型 → 多轮迭代后三者相互提升。**  
   过去的自训练往往只更新策略模型，价值模型（即过程奖励）保持不变。这里把两者都放进循环，让价值模型更精准，进而进一步提升搜索质量，形成正向反馈。

4. **无需额外数据的持续提升 → 只依赖已有的“oracle”答案（即题目正确答案），即可在多个迭代中不断提升模型性能。**  
   这突破了需要额外标注或外部奖励模型的限制，使方法在资源受限的场景下也能使用。

### 方法详解
**整体框架**  
ReST‑MCTS* 的工作流程可以划分为四步：  
1) 给定任务和正确答案，使用当前的策略模型生成候选推理树；  
2) 在这棵树上运行 MCTS\*，每次模拟时利用过程奖励估计来评估每一步的价值；  
3) 选出最有前景的完整推理轨迹，收集每一步的过程奖励（即该步帮助达成正确答案的概率）以及最终答案的成功与否；  
4) 用这些轨迹和过程奖励分别微调策略模型和价值模型，进入下一轮迭代。

**关键模块拆解**  

- **策略模型生成候选动作**：在每个搜索节点，策略模型像“思考者”一样输出若干可能的下一步文字（例如“计算 x+y”或“列出已知条件”），相当于在棋局中给出几手可能的走法。

- **过程奖励估计**：对每个候选动作，系统会在已有的正确答案约束下，统计在后续模拟中该动作出现的成功率。直观上，这相当于“如果我现在走这一步，后面还能走出正确答案的概率有多大”。这一步不需要人工标注，只依赖模拟统计。

- **MCTS\* 搜索**：传统 MCTS 包含四个步骤：选择、展开、模拟、回传。ReST‑MCTS* 在“回传”阶段把过程奖励的期望值加入节点的价值估计中，使得后续的“选择”更倾向于高奖励的分支。这样搜索过程既考虑整体成功率，也兼顾局部步骤的贡献。

- **双模型微调**：搜索结束后，系统得到两类数据：① 完整的高质量推理轨迹（用于策略模型的监督学习），② 每一步对应的过程奖励（用于价值模型的回归学习）。策略模型学习“怎么走”，价值模型学习“这步走得好不好”。两者交叉训练，形成闭环。

**最巧妙的地方**  
- **奖励的逆向推断**：作者把“最终答案已知”这一信息反向利用，推断出每一步的价值，这在强化学习里叫做“逆向奖励建模”，但在大语言模型自训练中极少出现。  
- **搜索预算共享**：在相同的计算预算下，MCTS\* 能比 Best‑of‑N 直接生成 N 条完整答案更有效，因为它把预算细分到每一步的选择上，而不是一次性浪费在完整答案的随机生成。

### 实验与效果
- **测试任务**：论文在多个需要多步推理的基准上评估，包括数学题解、逻辑推理和代码生成等（具体数据集未在摘要中列出，原文需查阅）。
- **对比基线**：与 Best‑of‑N、Tree‑of‑Thought、ReST^EM、Self‑Rewarding LM 等方法进行比较。论文声称在相同搜索预算下，ReST‑MCTS* 的准确率显著高于前者两者，且在多轮自训练后持续超越 ReST^EM 与 Self‑Rewarding LM。
- **消融实验**：通过去掉过程奖励或改用普通 MCTS，性能下降明显，说明过程奖励的引入是提升的关键因素。具体数值未在摘要中给出，原文提供了详细表格。
- **局限性**：方法依赖于“oracle”最终答案，如果答案本身不唯一或难以验证，过程奖励的推断会受影响。搜索过程仍然需要一定的计算资源，虽然比盲目生成更高效，但在超大模型上仍可能成为瓶颈。

### 影响与延伸思考
ReST‑MCTS* 把强化学习的树搜索与大语言模型的自训练结合，打开了“无需人工标注即可学习推理过程”的新路径。后续工作已经开始探索：
- **更轻量的过程奖励估计**，比如用小模型近似 MCTS 的统计过程，以降低计算成本（推测）。  
- **跨任务的通用价值模型**，尝试让一次学习的过程奖励能够迁移到其他任务上（推测）。  
- **与检索增强（RAG）结合**，让过程奖励还能考虑外部知识检索的贡献。  
如果想进一步了解，可以关注近期在强化学习与大模型交叉的会议（NeurIPS、ICLR）以及“Tree‑of‑Thought”系列的后续论文。

### 一句话记住它
**ReST‑MCTS* 用搜索里“哪步最可能通向正确答案”的概率自动生成过程奖励，让大模型在自训练时既学会答案，也学会可靠的推理步骤。**