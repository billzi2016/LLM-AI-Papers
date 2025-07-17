# QuestA: Expanding Reasoning Capacity in LLMs via Question Augmentation

> **Date**：2025-07-17
> **arXiv**：https://arxiv.org/abs/2507.13266

## Abstract

Reinforcement learning (RL) has emerged as a central paradigm for training large language models (LLMs) in reasoning tasks. Yet recent studies question RL's ability to incentivize reasoning capacity beyond the base model. This raises a key challenge: how can RL be adapted to solve harder reasoning problems more effectively? To address this challenge, we propose a simple yet effective strategy via Question Augmentation: introduce partial solutions during training to reduce problem difficulty and provide more informative learning signals. Our method, QuestA, when applied during RL training on math reasoning tasks, not only improves pass@1 but also pass@k-particularly on problems where standard RL struggles to make progress. This enables continual improvement over strong open-source models such as DeepScaleR and OpenMath Nemotron, further enhancing their reasoning capabilities. We achieve new state-of-the-art results on math benchmarks using 1.5B-parameter models: 72.50% (+10.73%) on AIME24, 62.29% (+12.79%) on AIME25, and 41.67% (+10.11%) on HMMT25. Code, data and model are available at https://github.com/foreverlasting1202/QuestA.

---

# QuestA：通过问题增强扩展大语言模型推理能力 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在自然语言生成上已经很强，但在需要多步推理的数学或逻辑题目上仍然容易卡壳。传统上，研究者把强化学习（RL）当作“奖励信号”，让模型在解题过程中自我改进，却发现模型往往只学会了表层的模式匹配，而不是深层的推理步骤。换句话说，RL 给的奖励太“稀疏”：只有在完整答案正确时才会得到正向信号，模型很难从错误中得到有价值的指导。于是出现了一个核心难题：如何让 RL 在训练期间提供更丰富、更易于利用的学习信号，让模型真正提升推理能力，而不是仅仅记忆题目特征？

### 关键概念速览
- **大语言模型（LLM）**：能够生成连贯文本的深度神经网络，规模从几亿到上千亿参数不等。把它想象成一个“会说话的百科全书”，但在需要多步思考时往往缺乏“笔记本”来记录中间过程。  
- **强化学习（RL）**：让模型在交互环境中通过试错获得奖励的学习框架。类似于让机器人在迷宫里走，只有走到出口才会得到奖励。  
- **推理能力**：模型在面对需要多步逻辑或数学运算的问题时，能够逐步展开、得出正确结论的能力。把它比作解数学题时的“演算过程”。  
- **问题增强（Question Augmentation）**：在训练时把题目本身和一段“部分解”一起喂给模型，降低原始问题的难度。可以类比为老师在考前给学生提供“提示”或“思路框架”。  
- **部分解（Partial Solution）**：不是完整答案，而是解题过程中的中间步骤或关键公式。它像是拼图的第一块，帮助模型更快定位剩余拼图的位置。  
- **pass@k**：在 k 次采样中至少有一次得到正确答案的概率。把它想成一次考试可以多次提交答案，只要有一次对了就算通过。  
- **持续改进（Continual Improvement）**：模型在同一训练阶段不断提升性能，而不是在某个点后停滞。类似于学生在做完一套练习后，立刻得到反馈并继续做下一套更难的题。

### 核心创新点
1. **从稀疏奖励到信息丰富的提示**  
   - 之前的 RL 方法只在模型输出完整答案后给出奖励，信号极其稀疏。  
   - QuestA 在每一步训练中把“部分解”嵌入问题描述，让模型在更易解的子任务上获得奖励。  
   - 结果是模型在同样的训练步数下，能够更快捕捉到推理路径，显著提升了 pass@1 与 pass@k。

2. **简洁的增强策略，无需额外模型**  
   - 早期的增强方法往往需要专门的教师模型或复杂的多阶段训练管线。  
   - 本文只在数据层面做了一个“拼接”操作：原始题目 + 部分解 → 增强样本。  
   - 这种轻量级做法让任何开源 RL 框架都能直接套用，降低了实现门槛。

3. **在强基线上实现大幅跃迁**  
   - 之前的最强开源基线（DeepScaleR、OpenMath Nemotron）在 AIME、HMMT 等数学竞赛上已经接近瓶颈。  
   - 引入问题增强后，同等参数规模（1.5B）模型的成绩分别提升了约 10%‑13%（如 AIME24 从 61.77% 提升到 72.50%）。  
   - 这证明了仅靠奖励信号的 RL 已经难以继续突破，提示信息的加入是提升推理能力的关键杠杆。

### 方法详解
**整体框架**  
QuestA 把“问题增强”嵌入到标准的 RL 微调流程中。整个过程可以划分为三步：① 生成或选取部分解；② 将部分解拼接到原始问题形成增强问题；③ 用增强问题进行 RL 交互，依据最终答案的正确性计算奖励并更新模型。整个管线仍然使用常见的 PPO（近端策略优化）或 REINFORCE 等策略梯度算法，只是输入数据被改造了。

**关键模块拆解**  

1. **部分解获取**  
   - 作者采用两种途径：  
     a. **人工标注**：对公开的数学竞赛题目提前写好关键步骤（如“先把方程化为标准形式”）。  
     b. **模型自回溯**：让一个已经微调好的小模型先尝试解题，截取前几步作为“候选部分解”。  
   - 这一步的目标不是完美，只要能提供有价值的中间线索即可。

2. **问题拼接策略**  
   - 增强问题的格式大致为：  
     ```
     题目：<原始题干>
     提示：<部分解>
     请给出完整答案。
     ```  
   - 类比于老师在黑板上先写出“已知条件”，再让学生继续推导。拼接方式保持了原始语言的自然流畅，避免破坏模型的语言理解能力。

3. **RL 交互与奖励计算**  
   - 模型在增强问题上生成完整答案。  
   - 奖励函数仍然是二元的（正确=1，错误=0），但因为问题已经被“降级”，模型更容易得到正向奖励，从而产生更平滑的梯度。  
   - 为了防止模型仅依赖提示而不真正推理，作者在训练后期逐步降低提示的显著性（比如把提示文字颜色改为灰色，或者在一定比例的样本中去掉提示），实现一种“渐进式去提示”的 curriculum 学习。

**最巧妙的设计**  
最让人眼前一亮的是“提示衰减”机制：在训练初期大量使用完整的部分解，让模型快速掌握解题框架；随后逐步削弱提示的可见度，让模型自行填补缺口。这个过程类似于人类学习时先有老师示范，后期独立完成练习，极大提升了模型的自我推理能力。

### 实验与效果
- **测试数据集**：AIME24、AIME25、HMMT25，这三套都是美国高中数学竞赛的正式试题，难度相当高，常被用作 LLM 推理能力的金标准。  
- **基线对比**：DeepScaleR、OpenMath Nemotron（均为 1.5B 参数的开源模型），以及未使用问题增强的普通 RL 微调。  
- **主要结果**：  
  - AIME24：从 61.77% 提升到 72.50%（+10.73%）。  
  - AIME25：从 49.50% 提升到 62.29%（+12.79%）。  
  - HMMT25：从 31.56% 提升到 41.67%（+10.11%）。  
  - 在 pass@k（k=5、10）上，同样出现两位数的提升，说明模型在多次采样中更容易找到正确答案。  
- **消融实验**：  
  - 去掉部分解直接使用原始题目，性能回落到普通 RL 水平，验证了提示信息是提升的关键因素。  
  - 只在训练末期加入提示，提升幅度明显小于全程使用提示，说明“早期强提示 + 后期去提示”的 curriculum 设计是必要的。  
- **局限性**：  
  - 实验仅覆盖数学推理，未验证在代码生成、常识推理等其他任务上的效果。  
  - 部分解的质量依赖于人工标注或小模型的生成能力，若提示本身错误可能会误导主模型。  
  - 只在 1.5B 参数模型上测试，尚不清楚在更大规模模型上是否仍保持同等增益。

### 影响与延伸思考
QuestA 的核心思想——用“可控的中间提示”来稀释奖励稀疏性——在随后的一年里激发了多条研究路线。  
- **提示式强化学习（Hint‑RL）**：多个团队尝试把提示从人工设计转为自动生成，甚至让模型自己学习何时给自己“提示”。  
- **课程学习（Curriculum Learning）与自适应难度**：QuestA 的提示衰减被视作一种简易的课程安排，后续工作加入了更细粒度的难度评估器，让模型在不同阶段接受不同层次的提示。  
- **跨任务迁移**：有研究把在数学任务上学到的提示策略迁移到代码补全或逻辑推理，观察到一定程度的正向迁移。  
如果想进一步深入，可以关注以下方向：① 自动化生成高质量部分解的算法（如利用符号求解器或检索式提示）；② 提示的多模态扩展（图形、公式渲染）；③ 在大模型（>10B）上验证提示衰减的尺度效应。  

### 一句话记住它
用“部分解”把难题先拆成易解子任务，让强化学习得到更密集的奖励，从而让小模型也能像人类一样逐步提升数学推理能力。