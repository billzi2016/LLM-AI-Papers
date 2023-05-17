# Tree of Thoughts: Deliberate Problem Solving with Large Language Models

> **Date**：2023-05-17
> **arXiv**：https://arxiv.org/abs/2305.10601

## Abstract

Language models are increasingly being deployed for general problem solving across a wide range of tasks, but are still confined to token-level, left-to-right decision-making processes during inference. This means they can fall short in tasks that require exploration, strategic lookahead, or where initial decisions play a pivotal role. To surmount these challenges, we introduce a new framework for language model inference, Tree of Thoughts (ToT), which generalizes over the popular Chain of Thought approach to prompting language models, and enables exploration over coherent units of text (thoughts) that serve as intermediate steps toward problem solving. ToT allows LMs to perform deliberate decision making by considering multiple different reasoning paths and self-evaluating choices to decide the next course of action, as well as looking ahead or backtracking when necessary to make global choices. Our experiments show that ToT significantly enhances language models' problem-solving abilities on three novel tasks requiring non-trivial planning or search: Game of 24, Creative Writing, and Mini Crosswords. For instance, in Game of 24, while GPT-4 with chain-of-thought prompting only solved 4% of tasks, our method achieved a success rate of 74%. Code repo with all prompts: https://github.com/princeton-nlp/tree-of-thought-llm.

---

# 思维树：大语言模型的深思熟虑问题求解 论文详细解读

### 背景：这个问题为什么难？
在 LLM（大语言模型）被广泛用于解答数学、推理、写作等任务之前，模型的推理过程基本是“一次走完”，即从左到右逐 token 生成答案。这样的线性思考方式在面对需要**探索**、**前瞻**或**关键决策**的任务时容易卡壳：模型只能凭当下的上下文做出决定，缺少回头检查或尝试不同路线的能力。于是出现了 CoT（思维链）提示，让模型先把推理步骤写出来，但仍然是单一路径、不可回溯的。要让模型像人类一样在思考中“试错”“换枝”，必须突破这种线性限制，这正是本文要解决的核心难点。

### 关键概念速览
- **LLM（大语言模型）**：通过海量文本训练得到的生成式模型，能够预测下一个词或句子。把它想象成会说话的“黑盒子”，我们只能通过提示词来引导它的行为。  
- **CoT（思维链）**：在提示中要求模型先写出推理步骤，再给出答案，类似于解题时把草稿写在纸上。它让中间过程可见，却仍然只能沿着唯一的链条前进。  
- **思考单元（Thought）**：本文把一段完整、语义连贯的文字（如“一步算式”或“一句情节描述”）视作一个原子操作，模型在每一步生成一个 Thought。可以把它比作棋局中的一次落子。  
- **思维树（Tree of Thoughts，ToT）**：把多个 Thought 按树形结构组织起来，根节点是问题本身，子节点是不同的思考路径。整个树相当于“所有可能的解题草稿”。  
- **自评估（Self‑Evaluation）**：模型在生成 Thought 后会给出一个分数或判断，衡量该思考是否合理、是否接近目标。类似于人在写草稿后自我检查的过程。  
- **搜索策略**：本文使用了深度优先搜索（DFS）和束搜索（Beam Search）等传统搜索算法，在思维树上决定哪条枝要继续展开、哪条要剪枝。  
- **回溯（Backtrack）**：当某条路径被评估为糟糕时，算法会撤回到上一个分叉点，尝试别的分支，就像玩迷宫时走错路后返回重新选择方向。  

### 核心创新点
1. **从线性链到树形结构**  
   之前的 CoT 只能让模型沿单一路径思考 → 本文把每一步思考抽象为 Thought，并在树上并行展开多条可能的路径 → 使模型能够在同一次推理中探索多种解法，显著提升了对需要搜索的任务的成功率。  

2. **引入 Thought 的自评估机制**  
   传统方法直接把生成的文本当作答案 → ToT 让模型在生成每个 Thought 后立即给出一个质量分数（或布尔判断），并据此决定是否继续扩展 → 这种“思考后自检”的环节帮助模型在搜索过程中及时剔除死路，降低了盲目扩展的成本。  

3. **结合经典搜索算法进行前瞻与回溯**  
   过去的 LLM 推理没有显式的搜索过程 → 作者把深度优先搜索和束搜索等算法嵌入到语言模型的调用流程中，模型可以“先看远一点”或“回头改枝” → 让语言模型拥有了类似棋类 AI 的全局规划能力。  

4. **在真实任务上实现跨任务的大幅提升**  
   之前的基线（直接输出或 CoT）在需要规划的任务上表现平平 → ToT 在 Game of 24、创意写作、Mini Crossword 三个全新任务上均取得显著提升，尤其在 Game of 24 上从 4% 提升到 74% → 证明了树形思考框架的通用性。  

### 方法详解
**整体框架**  
ToT 的推理过程可以概括为五步循环：  
1) **定义思考空间**：给定问题，设定每一步可以生成的 Thought 类型（如算式、句子、填字选项）。  
2) **生成候选 Thought**：调用 LLM，基于当前树节点的上下文一次性生成若干（k）候选 Thought。  
3) **自评估**：让同一个或另一个 LLM 对每个候选 Thought 打分或判断其可行性。  
4) **搜索决策**：依据评分和预设的搜索策略（DFS、束搜索），选出最有前景的节点继续展开，或剪枝掉低分枝。  
5) **终止检查**：如果某条路径满足任务的终止条件（如算式结果为 24、写完完整故事、填满十字格），则输出该路径；否则回到第 2 步继续扩展。  

**关键模块拆解**  
- **Thought 生成器**：相当于“棋手”，每次只考虑当前局面，输出一段完整的文字。实现上，只需在提示中加入“请给出下一步思考”。  
- **评分函数**：模型自评的“裁判”。可以是二分类（合法/非法）或回归分数（0‑1），也可以利用外部工具（如算数检查器）辅助。  
- **搜索控制器**：把评分结果喂给传统搜索算法。比如在束搜索中保留前 b 条最高分的节点；在深度优先中只保留当前最优分支并在必要时回溯。  
- **回溯机制**：当所有子节点被剪枝或达到深度上限时，控制器自动返回父节点，尝试未探索的兄弟分支。  

**类比**：把 ToT 想成一场文字版的象棋比赛。问题是棋局的起始局面，Thought 是每一步走子，模型既是棋手也是裁判，搜索算法决定走哪条线路，回溯就是认输后重新摆棋。  

**最巧妙的设计**  
- **思考单元的粒度可调**：作者没有固定 Thought 必须是一个 token，而是让它可以是完整的算式或句子，这让搜索空间既不至于过于细碎，也能保持足够的表达能力。  
- **自评估与搜索的闭环**：评分直接影响搜索路径，搜索又决定后续生成的上下文，实现了“思考—评估—再思考”的闭环，突破了传统一次性生成的瓶颈。  

### 实验与效果
- **测试任务**：  
  1) **Game of 24**：给定四个数字，要求通过加减乘除得到 24。  
  2) **Creative Writing**：在限定主题下生成连贯的短篇故事。  
  3) **Mini Crossword**：填完一个 5×5 的小型填字游戏。  

- **基线对比**：  
  - 直接让 GPT‑4 输出答案（无提示）  
  - 使用 CoT 提示的 GPT‑4（单链思考）  

- **关键结果**：  
  - 在 Game of 24 上，CoT 只解出约 4% 的题目，ToT 达到 74%（论文给出的具体数字）。  
  - 在 Creative Writing 与 Mini Crossword 上，ToT 同样比 CoT 提升了数倍的成功率，具体提升幅度论文中给出“显著提升”。  

- **消融实验**：作者分别去掉自评估、去掉束搜索、只用深度优先等变体，发现自评估是提升的主要驱动力，去掉后成功率跌回 CoT 水平左右。  

- **局限性**：  
  - 计算成本随搜索深度指数增长，需要在实际使用时手动设定搜索宽度和深度上限。  
  - 评分函数依赖模型自身的判断，若模型本身对某类错误缺乏感知，搜索仍会被误导。  
  - 论文主要在小规模任务上验证，尚未在大规模开放域问答或代码生成等更复杂场景进行系统评估。  

### 影响与延伸思考
自从这篇论文公开后，**思维树**的理念迅速被多篇后续工作引用，催生了诸如 **Tree of Reasoning**、**Program of Thoughts**、以及结合外部工具的 **Tool‑augmented Tree Search** 等变体。研究者开始把 LLM 当作“搜索子程序”，在强化学习、自动规划甚至化学分子设计中尝试类似的树形探索。  
如果想进一步深入，可以关注以下方向：  
- **更高效的搜索策略**：比如 Monte Carlo Tree Search（MCTS）与 LLM 的结合。  
- **跨模态思考单元**：把图像、代码等非文本信息也包装成 Thought，构建多模态思维树。  
- **自评估的鲁棒性提升**：利用外部校验器或对抗训练，让模型的自检更可靠。  

### 一句话记住它
**思维树让大语言模型不再“一步到位”，而是像人类一样在多条思路间前瞻、回溯、挑选，显著提升了需要规划和搜索的任务表现。**