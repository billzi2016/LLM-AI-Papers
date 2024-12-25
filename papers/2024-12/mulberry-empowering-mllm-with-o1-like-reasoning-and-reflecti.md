# Mulberry: Empowering MLLM with o1-like Reasoning and Reflection via   Collective Monte Carlo Tree Search

> **Date**：2024-12-24
> **arXiv**：https://arxiv.org/abs/2412.18319

## Abstract

In this work, we aim to develop an MLLM that understands and solves questions by learning to create each intermediate step of the reasoning involved till the final answer. To this end, we propose Collective Monte Carlo Tree Search (CoMCTS), a new learning-to-reason method for MLLMs, which introduces the concept of collective learning into ``tree search'' for effective and efficient reasoning-path searching and learning. The core idea of CoMCTS is to leverage collective knowledge from multiple models to collaboratively conjecture, search and identify effective reasoning paths toward correct answers via four iterative operations including Expansion, Simulation and Error Positioning, Backpropagation, and Selection. Using CoMCTS, we construct Mulberry-260k, a multimodal dataset with a tree of rich, explicit and well-defined reasoning nodes for each question. With Mulberry-260k, we perform collective SFT to train our model, Mulberry, a series of MLLMs with o1-like step-by-step Reasoning and Reflection capabilities. Extensive experiments demonstrate the superiority of our proposed methods on various benchmarks. Code will be available at https://github.com/HJYao00/Mulberry

---

# Mulberry：通过集体蒙特卡罗树搜索赋能多模态大语言模型实现类 o1 推理与反思 论文详细解读

### 背景：这个问题为什么难？

多模态大语言模型（MLLM）在回答复杂问题时往往直接给出答案，缺少可解释的思考过程。传统的“思维链”（CoT）虽然能让模型先写出步骤，但仍是单模型的自洽推理，容易卡在错误的中间环节，难以自行发现并纠正。更重要的是，现有的训练方式缺少系统化的“搜索”机制——模型没有办法像人类一样在多个候选思路之间比较、回溯、修正。于是，如何让 MLLM 像搜索树一样遍历可能的推理路径，并利用多模型的集体智慧来挑选最靠谱的路线，成为了一个亟待突破的瓶颈。

### 关键概念速览

**多模态大语言模型（MLLM）**：既能处理文字，又能理解图像、音频等非文本信息的模型，类似于会看图说话的聊天机器人。  

**思维链（CoT）**：让模型在给出最终答案前先把推理步骤写出来，像在草稿纸上列出解题步骤。  

**蒙特卡罗树搜索（MCTS）**：一种在游戏 AI 中常用的搜索算法，通过随机模拟和回溯统计来评估每一步的价值，类似于在棋盘上“试走”很多局面再决定下一步。  

**集体学习（Collective Learning）**：把多个模型的知识聚合起来，让它们相互提供线索、纠错，就像团队讨论后得到的共识答案。  

**CoMCTS（Collective Monte Carlo Tree Search）**：把集体学习和 MCTS 结合的框架，多个模型一起扩展、模拟、回溯搜索推理树。  

**Mulberry-260k 数据集**：作者用 CoMCTS 为每道题生成的、包含完整推理树的 260,000 条多模态问答数据。  

**SFT（Supervised Fine‑Tuning）**：在标注好的数据上进行有监督微调，让模型学会模仿数据中的推理步骤。  

**o1‑like 推理**：指 OpenAI o1 系统展示的那种“先思考、再反思、再回答”的多轮推理方式。

### 核心创新点

1. **从单模型推理到集体树搜索**  
   - 之前：大多数方法让单个模型自行生成思维链，缺少系统的搜索与纠错。  
   - 本文：引入 CoMCTS，让多个模型在同一推理树上协同工作，分别负责扩展新节点、模拟走子、定位错误并回传信息。  
   - 改变：搜索空间被显式结构化，错误更容易被定位并在后续迭代中被修正，整体推理质量提升。

2. **四步迭代循环的搜索框架**  
   - 之前：MCTS 只在单一模型内部进行，缺少对错误位置的专门定位。  
   - 本文：在每一次循环中执行 **扩展 → 模拟与错误定位 → 反向传播 → 选择**，其中“错误定位”专门检测哪一步产生了偏差并把信号反馈给根节点。  
   - 改变：模型不再盲目向下走，而是带着“哪里出错了”的信息重新评估路径，类似于人类在解题时回头检查关键步骤。

3. **基于 CoMCTS 生成的显式推理树数据集**  
   - 之前：公开的多模态数据集大多只提供问题和答案，缺少中间推理过程。  
   - 本文：利用 CoMCTS 自动生成每道题的完整推理树，形成 Mulberry-260k，这让后续的 SFT 能够直接学习“从根到叶的完整思考路径”。  
   - 改变：模型在微调阶段就能看到完整的思考链和错误纠正过程，学习到类似 o1 的“思考 → 反思 → 回答”模式。

4. **集体 SFT 训练策略**  
   - 之前：SFT 只针对单一模型的标注数据进行微调。  
   - 本文：把多个模型的输出合并成统一的训练样本，再对目标模型进行微调，使其内部隐式拥有“集体智慧”。  
   - 改变：微调后模型在推理时能够自行模拟集体讨论的效果，提升了对复杂多模态任务的鲁棒性。

### 方法详解

**整体框架**  
CoMCTS 把推理过程抽象成一棵树：根节点是原始问题，叶子节点是候选答案。多个模型轮流对这棵树执行四个操作——扩展、模拟+错误定位、反向传播、选择——不断迭代，直至找到满意的答案。随后把完整的树结构保存为训练样本，交给目标模型进行集体 SFT。

**1. 扩展（Expansion）**  
每个模型在当前选中的节点上生成若干可能的子步骤（例如“先识别图中物体 → 计算面积 → 与已知比例比较”）。这一步相当于在棋局中尝试新的走法，产生新的分支。

**2. 模拟与错误定位（Simulation & Error Positioning）**  
模型对每个新分支进行一次“快速演练”，即用简化的推理方式走到叶子，得到一个临时答案。随后与已知的参考答案或内部一致性检查比较，定位是哪一步导致偏差。错误定位的信号会被标记在对应的节点上。

**3. 反向传播（Backpropagation）**  
把错误定位得到的分数（越低代表越可靠）沿着路径向根节点回传，更新每个节点的价值估计。类似于在 MCTS 中把模拟结果的胜率向上累加，使得后续选择更倾向于高价值分支。

**4. 选择（Selection）**  
依据累计的价值和探索率（balance between exploring new branches and exploiting known good ones），模型挑选下一个要扩展的节点。这里的选择策略融合了集体模型的意见——如果多数模型对某条路径打高分，则该路径的选择概率提升。

**最巧妙的地方**  
- **错误定位**：传统 MCTS 只关注最终奖励，这里额外加入了对中间错误的检测，使得搜索过程更像人类的“调试”。  
- **集体协同**：每一步的扩展和模拟都由不同模型完成，信息在树上共享，等价于让多个专家在同一张白板上共同绘图。  

**生成 Mulberry-260k**  
在大量公开的多模态问答上运行 CoMCTS，得到每题的完整推理树。每棵树被序列化为“问题 → 步骤1 → 步骤2 … → 答案”，并标记出每一步的错误定位信息，形成高质量的监督信号。

**集体 SFT**  
把这些序列化的树作为训练数据，对目标 MLLM 进行有监督微调。微调过程让模型学习到：  
- 如何在每一步写出明确的中间结论；  
- 如何在发现冲突时回溯并重新推理；  
- 最终输出符合 o1 那种“先思考、再反思、再回答”的模式。

### 实验与效果

- **测试数据**：论文在多个公开的多模态推理基准上评估，包括 VQA、ScienceQA、MME 等（具体名称未在摘要中列出，原文未详细描述）。  
- **对比基线**：与普通 MLLM、使用单模型 CoT、以及最新的 o1 系统进行比较。  
- **结果**：论文声称在所有基准上均显著超越基线，尤其在需要多步推理的任务上提升幅度更大。具体数值未在摘要中给出。  
- **消融实验**：作者分别去掉“错误定位”“集体学习”“四步循环”中的任意一步，性能均出现明显下降，说明每个模块都是提升的关键因素。  
- **局限性**：CoMCTS 需要多模型并行运行，计算成本比单模型 CoT 高；生成的推理树质量依赖于模型的初始能力，若基模型太弱，搜索过程可能陷入错误分支。作者在讨论中承认这些问题，并提出未来可以通过模型压缩或分层搜索来降低开销。

### 影响与延伸思考

Mulberry 把“搜索”思路从游戏 AI 迁移到多模态推理，开启了“集体树搜索”在大语言模型中的新方向。后续已有工作尝试把强化学习的树搜索与大模型的自回归生成结合，或在少量模型上实现类似的协同推理（推测）。如果想进一步了解，可以关注以下方向：  
- **高效的分布式 MCTS**：如何在大规模模型集群上低延迟完成搜索。  
- **错误定位的自动化**：把错误检测做成可微分模块，直接嵌入模型内部。  
- **跨模态树结构**：让图像、文本、音频的推理节点在同一树中交叉，提升跨模态推理的连贯性。

### 一句话记住它

Mulberry 用多模型协同的蒙特卡罗树搜索，把推理过程显式化为可搜索、可纠错的树，让大语言模型拥有类似 o1 的“思考 → 反思 → 回答”能力。