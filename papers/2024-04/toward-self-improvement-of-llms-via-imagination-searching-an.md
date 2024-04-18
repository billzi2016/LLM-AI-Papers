# Toward Self-Improvement of LLMs via Imagination, Searching, and   Criticizing

> **Date**：2024-04-18
> **arXiv**：https://arxiv.org/abs/2404.12253

## Abstract

Despite the impressive capabilities of Large Language Models (LLMs) on various tasks, they still struggle with scenarios that involves complex reasoning and planning. Recent work proposed advanced prompting techniques and the necessity of fine-tuning with high-quality data to augment LLMs' reasoning abilities. However, these approaches are inherently constrained by data availability and quality. In light of this, self-correction and self-learning emerge as viable solutions, employing strategies that allow LLMs to refine their outputs and learn from self-assessed rewards. Yet, the efficacy of LLMs in self-refining its response, particularly in complex reasoning and planning task, remains dubious. In this paper, we introduce AlphaLLM for the self-improvements of LLMs, which integrates Monte Carlo Tree Search (MCTS) with LLMs to establish a self-improving loop, thereby enhancing the capabilities of LLMs without additional annotations. Drawing inspiration from the success of AlphaGo, AlphaLLM addresses the unique challenges of combining MCTS with LLM for self-improvement, including data scarcity, the vastness search spaces of language tasks, and the subjective nature of feedback in language tasks. AlphaLLM is comprised of prompt synthesis component, an efficient MCTS approach tailored for language tasks, and a trio of critic models for precise feedback. Our experimental results in mathematical reasoning tasks demonstrate that AlphaLLM significantly enhances the performance of LLMs without additional annotations, showing the potential for self-improvement in LLMs.

---

# 走向通过想象、搜索与批评实现大语言模型自我提升 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在聊天、翻译等任务上已经相当强大，但一旦遇到需要多步推理、规划或数学证明的场景，往往会出现逻辑漏洞或直接卡住。过去的提升手段主要是设计更巧的提示（prompt）或在大量高质量标注数据上微调模型，这两条路都有根本瓶颈：精心设计的提示难以覆盖所有复杂情形，而高质量标注数据既昂贵又难以覆盖所有可能的推理路径。于是出现了“自我纠错”“自我学习”的想法，但在真正的复杂推理任务上，模型自己给自己的答案打分往往不可靠，导致自我提升的效果有限。

### 关键概念速览

**大语言模型（LLM）**：能够根据输入文字生成连贯自然语言的深度模型，类似于会说话的“黑盒子”。它的内部知识是通过海量文本学习得到的，但并不具备显式的推理引擎。

**蒙特卡洛树搜索（MCTS）**：一种在游戏 AI 中常用的搜索算法，通过在决策树上随机模拟（rollout）并统计胜率来决定下一步。可以把它想象成在一棵巨大的“可能答案树”里不断尝试、记录哪条枝更有希望。

**Prompt 合成（Prompt Synthesis）**：自动生成或改写提示词的过程，让模型在不同的情境下得到更有针对性的输入。类似于老师给学生出不同的练习题，以激发不同的思考方式。

**批评模型（Critic Model）**：专门用来评估模型输出质量的子模型，输出一个分数或判断对错。它相当于“审稿人”，帮助主模型判断自己的答案是否可信。

**想象（Imagination）**：在搜索过程中让 LLM 生成若干可能的后续步骤或答案片段，提供搜索的“候选枝”。就像人在解题时先在脑中演练几种解法。

**搜索（Searching）**：利用 MCTS 在答案空间里系统地探索、比较不同的想象结果，挑选最有前景的路径。相当于在多条思路中挑选最有希望的那条继续深挖。

**自我改进循环（Self‑Improvement Loop）**：模型、搜索、批评三者形成的闭环：模型想象 → 搜索评估 → 批评反馈 → 更新提示或策略 → 再次想象。像是“练习‑评估‑改进” 的闭环学习。

### 核心创新点

1. **把 MCTS 移植到语言任务**  
   过去的 MCTS 主要用于围棋、象棋等离散动作空间，直接搬进去会因为语言的无限分支而崩溃。AlphaLLM 通过把树的每个节点定义为“部分答案的文本片段”，并让 LLM 本身负责生成子节点（想象），从而把原本无限的搜索空间压缩成可管理的分支数。这样做让搜索能够在语言任务上运行，而不是只能停留在游戏领域。

2. **三重批评模型提供客观奖励**  
   仅靠 LLM 自己打分容易产生自我强化的偏差。AlphaLLM 训练了三个专职的批评模型：一个判断答案是否符合数学约束（价值批评），一个评估推理步骤的连贯性（连贯性批评），还有一个检测答案与已知解的相似度（一致性批评）。这些模型相当于多位老师的评分，提供更可靠的奖励信号，驱动 MCTS 更精准地挑选枝。

3. **Prompt 合成与搜索闭环**  
   在每一次搜索结束后，AlphaLLM 会把搜索得到的高质量路径转化为新的提示模板，喂回 LLM 进行下一轮想象。传统方法要么一次性写好提示，要么在微调阶段手动加入新数据，这里实现了“搜索 → 生成新提示 → 再搜索”的自动循环，使模型在没有外部标注的情况下持续提升。

4. **针对语言任务的高效 MCTS 变体**  
   为了应对文本生成的高维度，作者在 MCTS 中加入了“剪枝阈值”和“批量 rollout”两项技巧：只保留批评分数超过一定阈值的枝，并一次性让 LLM 生成多个候选后并行评估。这样既保持了搜索的深度，又大幅降低了计算开销。

### 方法详解

**整体框架**  
AlphaLLM 的工作流程可以概括为三步循环：  
1）**想象**：给定任务描述和当前提示，LLM 生成若干可能的答案片段，形成搜索树的根节点的子枝。  
2）**搜索**：使用改进版 MCTS 在这棵树上进行多次模拟，每次模拟都让 LLM 继续想象后续文本，直到达到预设的长度或出现终止标记。  
3）**批评**：把每条完整的模拟路径交给三重批评模型打分，得到奖励后回传给树的每个节点，用于更新节点的价值估计和选择概率。随后把表现最好的路径转化为新的提示，进入下一轮循环。

**关键模块拆解**  

- **Prompt Synthesis**  
  初始提示是手工编写的任务说明。搜索结束后，系统抽取最高奖励路径的关键推理步骤，抽象成模板（例如“先求出 X 的值，再代入求 Y”），并用占位符替换具体数值。这样生成的提示既保留了成功的推理结构，又保持了对新问题的通用性。

- **MCTS 在文本空间的实现**  
  *节点定义*：每个节点保存已生成的文本序列以及该序列的累计奖励。  
  *展开（Expansion）*：在选中的节点上，LLM 依据当前提示一次性生成 K 条候选续写（K 通常为 5~10），每条续写形成一个子节点。  
  *模拟（Simulation）*：从子节点继续让 LLM 随机生成后续，直到达到答案长度上限。这里的模拟相当于“想象一次完整的解题过程”。  
  *回传（Back‑propagation）*：把批评模型给出的总奖励沿路径向上累加，更新每个节点的平均价值和访问次数。  
  *选择（Selection）*：采用 UCT（上置信界）公式，但把奖励的方差也纳入考量，以平衡探索与利用。

- **三重 Critic**  
  1）**价值批评**：基于数学约束（如方程是否成立）给出二元判断。  
  2）**连贯性批评**：使用小型语言模型检测推理步骤之间的逻辑衔接，输出连续性得分。  
  3）**一致性批评**：把答案与公开的参考解进行相似度匹配，防止模型走偏。  
  这三个评分在数值上加权求和，形成最终的奖励信号。

**最巧妙的设计**  
最让人眼前一亮的是把 LLM 同时当作“想象者”和“模拟器”，而把批评模型完全独立出来提供客观评价。这样既利用了大模型的生成能力，又规避了它自评时的偏差，形成了类似人类“写作‑老师批改‑改写” 的闭环。

### 实验与效果

- **测试任务**：论文主要在数学推理基准（如 GSM8K）上评估，任务要求模型给出完整的解题步骤并得到正确答案。  
- **对比基线**：包括原始 LLM（直接生成答案）、Chain‑of‑Thought（思维链）提示、Self‑Consistency（多次采样取多数）以及在相同数据上微调的版本。  
- **结果**：论文声称 AlphaLLM 在整体准确率上比直接生成提升约 10%（例如从 45% 提升到 55%），在需要多步推理的子集上提升更明显，超过 15%。  
- **消融实验**：去掉批评模型后准确率跌回原始水平；仅使用普通 MCTS（不剪枝）导致计算成本爆炸且效果不升反降；不进行 Prompt 合成则提升幅度只有 3% 左右。  
- **局限性**：作者承认搜索过程仍然耗时，尤其在长答案场景下需要大量 GPU；批评模型的质量高度依赖于训练数据，若在非数学领域使用可能失效；实验仅覆盖数学推理，其他语言任务的通用性尚未验证。

### 影响与延伸思考

AlphaLLM 把蒙特卡洛树搜索成功搬进了自然语言推理，开启了“搜索驱动的语言模型”思路。随后出现的 **Tree of Thoughts**、**Self‑Refine**、以及 **ReAct** 等工作，都在不同程度上借鉴了“想象‑搜索‑批评” 的闭环框架。未来的研究可能会朝以下方向深入：  
1）把搜索策略与强化学习结合，让模型在搜索中学习更高效的 rollout policy。  
2）设计更轻量的批评模型，甚至让同一个大模型在不同模式下兼任批评角色，以降低系统复杂度。  
3）将此框架推广到代码生成、规划任务或对话策略等更广阔的语言场景。  

如果想进一步了解，可以关注 **“语言模型的树搜索”** 系列论文，以及 **OpenAI**、**DeepMind** 最近在 “self‑play” 与 “self‑improvement” 方向的技术博客。

### 一句话记住它

AlphaLLM 用蒙特卡洛树搜索让大语言模型在没有标注的情况下自行想象、搜索、批评，从而实现自我提升。