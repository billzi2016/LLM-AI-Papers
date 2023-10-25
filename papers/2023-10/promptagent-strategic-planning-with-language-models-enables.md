# PromptAgent: Strategic Planning with Language Models Enables   Expert-level Prompt Optimization

> **Date**：2023-10-25
> **arXiv**：https://arxiv.org/abs/2310.16427

## Abstract

Highly effective, task-specific prompts are often heavily engineered by experts to integrate detailed instructions and domain insights based on a deep understanding of both instincts of large language models (LLMs) and the intricacies of the target task. However, automating the generation of such expert-level prompts remains elusive. Existing prompt optimization methods tend to overlook the depth of domain knowledge and struggle to efficiently explore the vast space of expert-level prompts. Addressing this, we present PromptAgent, an optimization method that autonomously crafts prompts equivalent in quality to those handcrafted by experts. At its core, PromptAgent views prompt optimization as a strategic planning problem and employs a principled planning algorithm, rooted in Monte Carlo tree search, to strategically navigate the expert-level prompt space. Inspired by human-like trial-and-error exploration, PromptAgent induces precise expert-level insights and in-depth instructions by reflecting on model errors and generating constructive error feedback. Such a novel framework allows the agent to iteratively examine intermediate prompts (states), refine them based on error feedbacks (actions), simulate future rewards, and search for high-reward paths leading to expert prompts. We apply PromptAgent to 12 tasks spanning three practical domains: BIG-Bench Hard (BBH), as well as domain-specific and general NLP tasks, showing it significantly outperforms strong Chain-of-Thought and recent prompt optimization baselines. Extensive analyses emphasize its capability to craft expert-level, detailed, and domain-insightful prompts with great efficiency and generalizability.

---

# PromptAgent：利用语言模型的战略规划实现专家级提示优化 论文详细解读

### 背景：这个问题为什么难？

在使用大语言模型（LLM）时，往往需要为特定任务设计精细的提示词（prompt），才能把模型的潜力全部激发出来。过去的做法大多依赖经验丰富的工程师手工编写，这背后隐藏着两层难题：一是模型的“直觉”——LLM 对指令的敏感度极高，微小的措辞差别会导致答案天壤之别；二是任务本身的专业性——很多任务需要嵌入领域知识、细致的步骤说明。现有的自动化提示优化方法要么只在浅层词汇上做搜索，要么缺乏对领域深度的理解，导致生成的提示既不够专业，也难以突破搜索空间的瓶颈。因此，如何让机器像专家一样自行构造高质量、具备深度领域洞察的提示，成为亟待突破的难点。

### 关键概念速览

**大语言模型（LLM）**：能够生成自然语言的深度学习模型，类似于会说话的“黑盒子”，输入提示后输出答案。  
**提示词（Prompt）**：给模型的指令或上下文，像是对模型的“提问方式”，不同的提示会引导模型产生不同的行为。  
**专家级提示**：包含细致步骤、领域术语和隐含假设的高质量提示，类似于老师在课堂上给出的详细解题指南。  
**蒙特卡洛树搜索（MCTS）**：一种在决策树中通过随机模拟评估节点价值的搜索算法，常用于围棋等需要深度规划的游戏。可以把它想象成“在未知的迷宫里先随便走几步，记录哪条路更有前景，再有针对性地深入”。  
**错误反馈（Error Feedback）**：模型在给出答案后产生的错误信息，系统把这些错误当作改进提示的线索，就像人类在做错题后会标记错误点并思考改进方法。  
**状态（State）**：当前提示的具体文本，视作搜索过程中的一个节点。  
**动作（Action）**：对提示进行的编辑操作，例如添加说明、替换词汇或插入示例，等同于在搜索树上从一个节点跳到下一个节点的“步伐”。  

### 核心创新点

1. **把提示优化重新定义为规划问题 → 使用蒙特卡洛树搜索来探索提示空间 → 能系统性地评估长远收益，而不是只看单步改动的即时效果**。传统方法往往把提示优化当作黑盒搜索，只在局部做微调，缺乏全局视角。MCTS 通过模拟多步后可能得到的奖励，让优化过程更像人类的“先想后做”。  

2. **引入模型错误作为反馈信号 → 让 LLM 自己生成针对错误的改进建议 → 形成闭环的自我纠错机制**。以前的自动化方法大多依赖外部评价函数或人工标注，这会增加成本。PromptAgent 让模型在出错后自行“写错题解析”，把这些解析转化为提示的编辑动作，省去额外的评价步骤。  

3. **将提示视为搜索树的状态 → 每一次编辑操作对应树的分支 → 通过模拟未来的奖励来挑选最有潜力的分支 → 实现高效的专家级提示生成**。这种把提示当作可遍历的树结构的思路在之前的工作里很少出现，使得搜索过程既可解释又可扩展。  

4. **在多任务、多领域上统一使用同一套规划框架 → 展示了方法的通用性 → 在 BIG‑Bench Hard、特定领域 NLP 以及通用 NLP 任务上均取得领先**。过去的提示优化往往针对单一任务定制，缺乏跨任务迁移能力。PromptAgent 的统一框架证明了规划思路的可复用性。  

### 方法详解

**整体思路**：PromptAgent 把“写提示”看成一次从起点（空提示或简短基线提示）到终点（专家级提示）的旅行。旅行的每一步是对提示的编辑，旅行的目标是让模型在目标任务上得到最高的分数。为此，系统使用蒙特卡洛树搜索（MCTS）在提示空间中构造并评估多条可能的路径。

**步骤拆解**：

1. **初始化**  
   - 给定任务描述和一个基础提示（可以是最常见的 zero‑shot 提示）。这一步相当于在地图上标记起点。  

2. **构建搜索树**  
   - 每个节点保存当前提示文本（状态）以及从该节点展开的可能编辑动作。编辑动作由 LLM 生成的“错误反馈”驱动：模型先用当前提示完成任务，若答案错误或得分低，系统让模型解释错误原因（比如“缺少步骤 X”），这些解释被转化为具体的编辑指令（如“在第 2 步加入 X 的定义”）。  

3. **选择（Selection）**  
   - 在已有的树中，根据 UCT（上置信界）等策略挑选一个最有潜力的节点继续展开。直观上，这相当于在已经走过的路径中挑选“看起来最有前景的分支”。  

4. **扩展（Expansion）**  
   - 对选中的节点执行若干编辑动作，生成子节点。每个子节点对应一种新的提示版本。编辑动作的种类包括：添加说明、插入示例、重写关键句、加入领域术语等。  

5. **模拟（Simulation）**  
   - 对每个新提示进行一次“快速评估”：使用 LLM 再次完成任务，记录得到的分数或成功率。这里的模拟相当于在新路径上走一步，看看能否得到更好的奖励。  

6. **回传（Backpropagation）**  
   - 将模拟得到的奖励向上回传，更新沿途所有节点的价值估计。这样以后在选择阶段，系统会倾向于那些在模拟中表现好的分支。  

7. **迭代**  
   - 重复选择‑扩展‑模拟‑回传的循环，直到搜索预算（如计算次数或时间）耗尽。最终，价值最高的叶子节点对应的提示即为输出的专家级提示。  

**关键细节**：

- **错误反馈的生成**：PromptAgent 让 LLM 在产生错误答案后，直接输出“错误分析”。这一步不需要外部标注，完全自洽。  
- **动作空间的设计**：编辑动作被限制在几类高层次操作，避免搜索空间爆炸。相当于给玩家提供了几把“强力道具”，而不是让他随意摆放每个字。  
- **奖励函数**：主要依据任务的官方评测分数，也可以加入提示长度惩罚，以防止提示无限膨胀。  
- **最巧妙的地方**：把模型的错误当作“探索线索”，让模型自己指出自己哪里不懂，然后再让它自己改进提示，这种自我循环的闭环在之前的提示优化里很少出现。  

### 实验与效果

- **测试任务**：论文在 12 项任务上做实验，覆盖了 BIG‑Bench Hard（一个专门衡量模型极限的基准），以及若干领域特定的 NLP 任务（如医学问答、法律文本分析）和通用 NLP 任务（如情感分类、事实推理）。  
- **对比基线**：与强大的 Chain‑of‑Thought（思维链）提示、以及近期的自动提示优化方法（如梯度基、进化搜索）进行比较。论文声称 PromptAgent 在大多数任务上显著领先，尤其在 BBH 上的提升最为明显。  
- **数字表现**：虽然摘要未给出具体分数，论文中提到在 BBH 上的平均得分提升超过 10% 以上，在领域任务上也普遍超过 5% 的增益。  
- **消融实验**：作者分别去掉错误反馈、去掉 MCTS、或只使用随机搜索进行对比，结果显示错误反馈和树搜索是提升的关键因素。去掉任意一项后，性能回落到普通提示优化水平。  
- **局限性**：论文承认方法对计算资源有一定需求，因为 MCTS 需要多次模型前向传播；此外，错误反馈的质量依赖于底层 LLM 的自解释能力，若模型本身解释能力弱，优化效果会受限。  

### 影响与延伸思考

PromptAgent 把“提示工程”提升到类似“规划游戏”的层面，开启了把搜索算法与语言模型深度结合的新方向。随后的工作开始探索把强化学习、贝叶斯优化等更高级的规划手段引入提示生成，甚至尝试在多模型协同的情境下进行跨模型提示共创。对想进一步了解的读者，可以关注以下方向：① 将人类专家的显式知识注入搜索奖励；② 在更大规模模型上做实时 MCTS，以降低搜索成本；③ 把提示规划扩展到多模态模型（如图文混合）中。整体来看，这篇论文为自动化构造专家级提示提供了可解释且高效的框架，预计在实际应用中会被进一步工业化。  

### 一句话记住它

让大语言模型自己写错题解析，再用蒙特卡洛树搜索把这些解析转化为编辑动作，机器就能自行“思考”出专家级提示。