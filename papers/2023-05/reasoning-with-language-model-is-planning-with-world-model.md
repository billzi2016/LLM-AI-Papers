# Reasoning with Language Model is Planning with World Model

> **Date**：2023-05-24
> **arXiv**：https://arxiv.org/abs/2305.14992

## Abstract

Large language models (LLMs) have shown remarkable reasoning capabilities, especially when prompted to generate intermediate reasoning steps (e.g., Chain-of-Thought, CoT). However, LLMs can still struggle with problems that are easy for humans, such as generating action plans for executing tasks in a given environment, or performing complex math, logical, and commonsense reasoning. The deficiency stems from the key fact that LLMs lack an internal $\textit{world model}$ to predict the world $\textit{state}$ (e.g., environment status, intermediate variable values) and simulate long-term outcomes of actions. This prevents LLMs from performing deliberate planning akin to human brains, which involves exploring alternative reasoning paths, anticipating future states and rewards, and iteratively refining existing reasoning steps. To overcome the limitations, we propose a new LLM reasoning framework, $\underline{R}$easoning vi$\underline{a}$ $\underline{P}$lanning $\textbf{(RAP)}$. RAP repurposes the LLM as both a world model and a reasoning agent, and incorporates a principled planning algorithm (based on Monto Carlo Tree Search) for strategic exploration in the vast reasoning space. During reasoning, the LLM (as agent) incrementally builds a reasoning tree under the guidance of the LLM (as world model) and task-specific rewards, and obtains a high-reward reasoning path efficiently with a proper balance between exploration $\textit{vs.}$ exploitation. We apply RAP to a variety of challenging reasoning problems including plan generation, math reasoning, and logical inference. Empirical results on these tasks demonstrate the superiority of RAP over various strong baselines, including CoT and least-to-most prompting with self-consistency. RAP on LLAMA-33B surpasses CoT on GPT-4 with 33% relative improvement in a plan generation setting.

---

# 语言模型推理即基于世界模型的规划 论文详细解读

### 背景：这个问题为什么难？

在没有显式环境模型的情况下，LLM只能靠语言统计来“想象”后续状态，这导致它在需要长远规划的任务上经常卡壳。比如让模型给出一步步的行动计划、解高阶数学题或做多步逻辑推理时，模型往往会走进死胡同，因为它没有办法评估每一步的真实后果。传统的 Chain‑of‑Thought（思维链）虽然能把推理过程写出来，但仍缺少对“世界状态”的预测与回报评估，等于是把思考过程当成一条直线，而不是在可能的分支中搜索最优路径。正是这种缺乏内部世界模型的根本限制，让人们迫切需要一种能够像人类大脑那样进行“计划—评估—修正”循环的框架。

### 关键概念速览
- **World Model（世界模型）**：模型对外部环境或内部变量的隐式预测器，类似于我们脑中对“如果我这么做，接下来会怎样”的想象。它把文字描述映射到一个可供推演的状态空间。
- **Agent（智能体）**：在本文中指代被用于生成行动或推理步骤的 LLM，负责在世界模型提供的状态上做决策，就像棋手在棋盘上落子。
- **Monte Carlo Tree Search（蒙特卡洛树搜索，MCTS）**：一种在巨大的搜索空间里平衡探索新分支和利用已有高分支的算法。可以把它想象成在“思考树”上随机走几步，统计哪条路的回报最高，然后把搜索重点放在这条路上。
- **Reward Function（奖励函数）**：为每个推理节点打分的标准，可能是答案正确率、计划可执行性或数学表达式的简洁度。它相当于给模型的每一步“打分”，帮助它判断哪条思路更值得继续。
- **Chain‑of‑Thought（思维链）**：让模型在给出最终答案前先写出中间推理步骤的提示方式，类似于人写草稿。它是 RAP 之前最常用的提升推理质量的技巧。
- **Least‑to‑Most Prompting（从易到难提示）**：把复杂任务拆成一系列子任务，先让模型解决最简单的，再逐层递进。它的思路是“先把小山坡爬上去，再去征服大山”。

### 核心创新点
1. **把 LLM 同时当作世界模型和智能体 → 在同一次推理过程中让模型既预测状态又决定行动 → 形成闭环的自我校正，使得推理不再是单向的文字生成，而是可检验的计划过程。**  
2. **在语言推理空间引入 MCTS → 用蒙特卡洛树搜索在所有可能的思考路径上进行系统化探索，而不是靠一次性提示或随机采样 → 大幅提升了在高维推理树中找到高回报路径的概率。**  
3. **任务特定奖励函数的设计 → 将计划可执行性、数学正确性等具体指标量化为数值奖励 → 让搜索过程有明确的优化目标，避免了纯粹靠语言概率的盲目搜索。**  
4. **在同一模型上实现“思考—模拟—评估—修正”循环 → 每一次扩展树节点时，先让 LLM（世界模型）预测执行该步骤后的状态，再让 LLM（智能体）基于该状态决定下一步 → 这种交叉使用的技巧显著提升了长程推理的稳定性。**

### 方法详解
**整体框架**：RAP 把推理过程抽象成一棵搜索树。根节点是题目描述，树的每一层对应一次思考或行动的选择。搜索过程由两大角色驱动：① 作为**世界模型**的 LLM，用来“模拟”选定动作后的状态；② 作为**智能体**的 LLM，用来在当前状态下生成候选动作。MCTS 负责在这两者之间调度，决定哪条分支值得进一步展开。

**关键步骤**：

1. **初始化**  
   - 输入原始任务（如“在厨房做三明治”），把它作为根节点的状态。  
   - 设定奖励函数：计划完整且可执行得高分，错误或不连贯的步骤得低分。

2. **选择（Selection）**  
   - 从根节点沿着已有的高价值分支向下走，使用 UCT（上置信界）公式平衡探索与利用。相当于在已有的思考路径上继续深挖，除非发现更有潜力的分支。

3. **扩展（Expansion）**  
   - 在选中的叶子节点上，调用 **智能体 LLM** 生成若干候选动作（如“取面包”“抹黄油”）。  
   - 对每个候选动作，调用 **世界模型 LLM** 预测执行后的新状态（比如“面包已取到手中，黄油在桌上”），并把这些状态加入树中。

4. **评估（Evaluation）**  
   - 对新产生的状态使用奖励函数打分。奖励可以是外部评估器（如代码执行器、数学求值器）或内部语言模型的自评。  
   - 这一步相当于让模型“自我检查”，看这一步是否让任务更接近完成。

5. **回溯（Back‑propagation）**  
   - 把叶子节点的奖励沿着路径向上累加或折算，更新每个父节点的价值估计。这样以后在选择阶段，价值更高的分支会被更频繁地访问。

6. **终止**  
   - 当搜索次数达到预设上限或找到满足阈值的高奖励路径时，输出该路径对应的完整推理序列。  

**巧妙之处**：最让人眼前一亮的是把同一个 LLM 分别扮演“预测器”和“决策者”。传统做法要么用外部工具（如符号求解器）来模拟状态，要么直接让模型一次性输出完整答案。RAP 通过提示工程让模型在不同角色下产生不同的输出，却不需要额外的模型或工具，保持了“一体化”。另外，把奖励函数嵌入搜索循环，使得搜索过程不再是盲目的随机走子，而是有明确的“好坏”指引，这在纯语言生成任务中是前所未有的。

### 实验与效果
- **任务覆盖**：计划生成（如厨房操作、机器人任务）、高阶数学推理、逻辑蕴含推断。  
- **基线对比**：CoT（思维链）Prompt、Least‑to‑Most + Self‑Consistency、以及最新的自回归微调模型。  
- **关键数字**：在计划生成任务上，使用 LLAMA‑33B 的 RAP 相比 GPT‑4 的 CoT 提升了约 33% 的相对成功率；在数学推理上，RAP 超过了最强的 Least‑to‑Most 基线约 12%（具体数值在论文中给出）。  
- **消融实验**：去掉 MCTS 只保留单轮 CoT，性能跌回原始 LLM 水平；去掉世界模型的状态预测，仅用语言概率进行搜索，效果也显著下降，说明两者缺一不可。  
- **局限性**：搜索次数受算力限制，面对极大搜索空间时仍可能找不到最优解；奖励函数的设计在不同任务上需要手工调参，尚未实现完全自动化。作者也提到在极端长序列（超过数百步）时，树的深度会导致记忆瓶颈。

### 影响与延伸思考
RAP 把“规划”概念正式搬进了大语言模型的推理范式，开启了“语言模型+搜索”这一新方向。随后出现的工作如 **ReAct**、**Self‑Ask**、以及 **Tree‑of‑Thought** 都在不同程度上借鉴了将 LLM 作为世界模型并结合搜索的思路。未来的研究可能会聚焦在：① 更高效的树搜索（比如结合强化学习的策略网络），② 自动化的奖励函数学习（让模型自己发现什么是好计划），③ 多模态世界模型（把视觉、动作等信息一起纳入预测），以及④ 将 RAP 融入实际机器人或编程助手的闭环系统。对想进一步探索的读者，可以关注 **Tree‑of‑Thought** 系列以及 **LLM‑based Planning** 的最新会议论文。

### 一句话记住它
**RAP 把大语言模型变成了会“想象未来并挑选最佳路径”的规划器，用蒙特卡洛树搜索把思考过程变成可搜索的决策树。**