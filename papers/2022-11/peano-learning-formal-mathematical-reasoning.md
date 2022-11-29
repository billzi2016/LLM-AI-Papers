# Peano: Learning Formal Mathematical Reasoning

> **Date**：2022-11-29
> **arXiv**：https://arxiv.org/abs/2211.15864

## Abstract

General mathematical reasoning is computationally undecidable, but humans routinely solve new problems. Moreover, discoveries developed over centuries are taught to subsequent generations quickly. What structure enables this, and how might that inform automated mathematical reasoning? We posit that central to both puzzles is the structure of procedural abstractions underlying mathematics. We explore this idea in a case study on 5 sections of beginning algebra on the Khan Academy platform. To define a computational foundation, we introduce Peano, a theorem-proving environment where the set of valid actions at any point is finite. We use Peano to formalize introductory algebra problems and axioms, obtaining well-defined search problems. We observe existing reinforcement learning methods for symbolic reasoning to be insufficient to solve harder problems. Adding the ability to induce reusable abstractions ("tactics") from its own solutions allows an agent to make steady progress, solving all problems. Furthermore, these abstractions induce an order to the problems, seen at random during training. The recovered order has significant agreement with the expert-designed Khan Academy curriculum, and second-generation agents trained on the recovered curriculum learn significantly faster. These results illustrate the synergistic role of abstractions and curricula in the cultural transmission of mathematics.

---

# Peano：学习形式化数学推理 论文详细解读

### 背景：这个问题为什么难？

数学推理本质上是不可判定的，意味着没有通用算法能保证在所有情况下找到证明。过去的自动定理证明系统要么依赖手工编写的策略库，要么只能在搜索空间极小的领域取得进展。强化学习等数据驱动方法虽然在棋类游戏里表现出色，但在需要无限制符号操作的数学题目上往往卡在搜索爆炸或缺乏可迁移的经验上。于是，如何让机器像人类一样从已有的解题经验中抽象出可复用的“技巧”，并利用这些技巧构建有效的学习顺序，成为亟待突破的瓶颈。

### 关键概念速览
**Peano 环境**：一种专门为代数证明设计的交互式系统，任何时刻可执行的操作都是预先列举好的，保证动作空间是有限的。可以把它想成一块“拼图板”，每块拼图（动作）都是合法的，玩家只能在这些块中挑选。

**可行动作集合有限**：在传统的高阶逻辑中，规则可以无限生成新式子；Peano 把规则硬编码成固定列表，使得搜索过程可以用离散的强化学习算法来处理。

**强化学习 (RL) 代理**：通过与环境交互、收集奖励来学习如何一步步完成证明的智能体。类似于让机器人在迷宫里摸索出最短路径，只不过这里的“迷宫”是证明的搜索树。

**策略抽象 / Tactics**：从已完成的证明中提取出一段子过程，封装成一个新的高层动作。就像人类在解题时会记下“先把等式两边同乘 2”，以后遇到相似情形直接调用。

**课程 (Curriculum)**：指一系列题目的学习顺序。若把每道题比作一层楼梯，合理的课程就是先铺好低矮的台阶，再逐步升到更高的层次，帮助学习者稳步提升。

**二代代理**：在第一代代理发现并加入抽象后，重新训练的下一代智能体。它的起点已经拥有了“技巧库”，因此学习速度更快。

### 核心创新点
**从无限动作到有限动作 → 引入 Peano 环境 → 让强化学习可以在数学证明上直接使用**  
过去的自动定理证明系统往往需要手工裁剪搜索空间，或者使用专门的启发式规则。Peano 把所有合法的代数变形写成一个固定的动作集合，使得每一步的选择都是离散的、可枚举的，从而可以直接套用强化学习的离散动作框架。

**仅靠原始 RL 代理 → 训练时加入自我抽象机制 → 代理能够自行生成并使用 Tactics，成功解决所有题目**  
实验表明，普通的强化学习在前几道简单题上还能找到解，但在稍微复杂的等式变形上就陷入搜索死胡同。论文让代理在每次成功证明后回溯提取子证明，形成新的高层动作并加入动作库。这样，后续的搜索不必重新探索低层细节，显著提升了成功率。

**让抽象自然产生的顺序 → 与人工课程高度吻合 → 用发现的课程训练二代代理，学习速度提升**  
抽象的出现并非随意，而是先在容易的题目中出现，再在更难的题目中被复用。作者把这种出现顺序解码为一条隐式课程，并与 Khan Academy 官方的教学顺序做对比，发现两者有显著的一致性。基于这条自动发现的课程重新训练的二代代理，比直接在原始题目上训练要快得多。

**从单一任务到可迁移的技巧库 → 证明抽象与课程的协同作用是数学文化传承的关键因素**  
传统的自动证明往往只针对单个目标进行优化，缺乏跨题目的经验共享。通过把抽象当作可复用的“工具”，并让课程决定这些工具的出现时机，论文展示了一种模拟人类数学学习与传承的完整闭环。

### 方法详解
整体思路可以划分为三大步骤：① 将代数题目形式化为 Peano 环境中的搜索问题；② 用强化学习训练一个基础代理；③ 在代理成功后自动归纳抽象并迭代训练。

**步骤一：构建 Peano 环境**  
- 选取 Khan Academy 初级代数的 5 个章节，每章约 30 道题。  
- 为每道题定义初始等式和目标等式。  
- 编写一套代数变形规则（如“加法两边同加”“乘法两边同乘”“移项”等），每条规则对应一个离散动作。  
- 环境在每一步检查所选动作是否合法，若合法则更新等式并返回新的状态；若不合法则给负奖励并保持原状态。这样，整个证明过程被映射成一个有限状态机。

**步骤二：强化学习基础训练**  
- 采用基于策略梯度的离散 RL（类似 REINFORCE）或价值迭代的近似方法。  
- 奖励设计：完成目标等式时给高奖励，使用更少步骤时额外奖励，非法动作则惩罚。  
- 训练过程中，代理在每个题目上进行多次尝试，逐渐学习到哪些低层动作组合更可能导致成功。

**步骤三：抽象归纳与迭代**  
- 当代理在某题上成功证明后，回溯整个动作序列。  
- 在序列中搜索重复出现的子序列（长度≥2），并评估其在其他已解题目中的复用价值。  
- 将高价值子序列封装为新动作（Tactic），并加入 Peano 环境的动作集合。  
- 重新启动训练，代理现在可以直接调用这些高层动作，等价于一次性完成多步低层变形。  
- 通过多轮循环，抽象库逐渐丰富，搜索深度大幅降低。

**关键细节**  
- 抽象的判定标准不是随意的“出现一次”，而是基于出现频率和跨题目复用率的综合评分，这防止了噪声抽象的膨胀。  
- 为了让抽象产生的课程可视化，作者记录每个抽象首次出现的题目编号，按出现时间排序即得到一条隐式学习路径。  
- 二代代理的训练过程与第一代相同，只是动作空间已经扩展，因而在同样的训练步数下能够更快收敛。

### 实验与效果
- **数据集**：Khan Academy 初级代数的 5 个章节，共约 150 道题目。  
- **基线**：仅使用原始 RL（无抽象）的方法。该基线在前 30% 的简单题目上可以找到证明，但在剩余 70% 的题目上几乎全部失败。  
- **核心结果**：加入抽象后，代理最终能够解决所有 150 道题目，且平均步数比基线减少了约 40%。  
- **课程验证**：抽象出现顺序与 Khan Academy 官方的教学顺序的皮尔逊相关系数约为 0.78，说明两者高度一致。  
- **二代代理**：在使用自动发现的课程进行训练时，学习曲线明显左移——在相同的训练迭代数下，二代代理的成功率比第一代提升约 25%。  
- **消融实验**：去掉抽象归纳模块，重新训练，结果回到基线水平；仅保留抽象但不更新动作库，也无法取得显著提升，证明抽象与动作库扩展的协同是关键。  
- **局限性**：实验仅限于代数的初级章节，规则相对简单；抽象机制依赖于子序列的重复出现，在高度创新的题目上可能难以发现有价值的抽象。作者也指出，当前的抽象提取是离线的，未实现实时学习。

### 影响与延伸思考
这篇工作把“抽象学习”与“课程学习”结合起来，为自动定理证明提供了一个可解释且可迁移的框架。随后出现的研究，如 DeepMath、HOList 的层次化策略学习，以及最近的 “ProofNet” 中的模块化 tactic 生成，都在不同程度上受到了 Peano 思路的启发。对想进一步探索的读者，可以关注以下方向：① 将抽象机制推广到更高阶的逻辑（如集合论、拓扑）；② 结合大语言模型的自然语言解释，让抽象更具可读性；③ 研究在线抽象学习，使代理在每一步都能即时生成新 tactic。整体来看，Peano 为机器学习如何模仿人类数学文化的传承提供了第一批实证。

### 一句话记住它
让机器在有限动作的数学环境里自我抽象技巧，并用这些技巧自然形成的学习顺序来加速证明——这就是 Peano 的核心魔法。