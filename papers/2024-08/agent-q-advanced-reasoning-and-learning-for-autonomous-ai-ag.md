# Agent Q: Advanced Reasoning and Learning for Autonomous AI Agents

> **Date**：2024-08-13
> **arXiv**：https://arxiv.org/abs/2408.07199

## Abstract

Large Language Models (LLMs) have shown remarkable capabilities in natural language tasks requiring complex reasoning, yet their application in agentic, multi-step reasoning within interactive environments remains a difficult challenge. Traditional supervised pre-training on static datasets falls short in enabling autonomous agent capabilities needed to perform complex decision-making in dynamic settings like web navigation. Previous attempts to bridge this ga-through supervised fine-tuning on curated expert demonstrations-often suffer from compounding errors and limited exploration data, resulting in sub-optimal policy outcomes. To overcome these challenges, we propose a framework that combines guided Monte Carlo Tree Search (MCTS) search with a self-critique mechanism and iterative fine-tuning on agent interactions using an off-policy variant of the Direct Preference Optimization (DPO) algorithm. Our method allows LLM agents to learn effectively from both successful and unsuccessful trajectories, thereby improving their generalization in complex, multi-step reasoning tasks. We validate our approach in the WebShop environment-a simulated e-commerce platform where it consistently outperforms behavior cloning and reinforced fine-tuning baseline, and beats average human performance when equipped with the capability to do online search. In real-world booking scenarios, our methodology boosts Llama-3 70B model's zero-shot performance from 18.6% to 81.7% success rate (a 340% relative increase) after a single day of data collection and further to 95.4% with online search. We believe this represents a substantial leap forward in the capabilities of autonomous agents, paving the way for more sophisticated and reliable decision-making in real-world settings.

---

# Agent Q：面向自主 AI 代理的高级推理与学习 论文详细解读

### 背景：这个问题为什么难？
在交互式环境（如网页浏览、在线预订）中，LLM 需要把语言理解转化为连续的动作序列，并在每一步根据环境反馈做出决策。传统的监督预训练只在静态文本上学习，缺乏对动态状态的感知和长期规划能力。早期的行为克隆（behavior cloning）只能模仿专家演示，容易出现“错误累积”——一次小失误会导致后续所有动作都偏离轨道。强化学习虽然能让模型自行探索，但在高维、稀疏奖励的网页任务里采样成本极高，往往收敛到次优策略。于是，如何让 LLM 在真实交互中既能利用已有示例，又能从失败中快速纠正，成为瓶颈。

### 关键概念速览
**部分可观察马尔可夫决策过程（POMDP）**：把网页交互抽象为状态、动作、观察的循环，模型只能看到页面内容和指令，类似人在黑箱里玩游戏只能看到屏幕。  
**Monte Carlo Tree Search（MCTS）**：在每一步通过模拟多条可能的行动路径并统计结果来选最有前景的动作，像在棋盘上先走几步“试探”。  
**自我批评（Self‑Critique）**：模型在生成答案后再审视自己的推理过程，找出不合理之处并给出改进建议，类似人写完作文后自己检查错别字。  
**Direct Preference Optimization（DPO）**：一种离线学习方式，直接把“更好”与“更差”的轨迹对比，用偏好信号来更新模型，而不是传统的奖励回传。  
**离线重放缓冲区（Replay Buffer）**：把过去的交互记录存起来，供后续训练反复抽样，像把比赛录像保存下来反复观看学习。  
**行为克隆（Behavior Cloning）**：把专家演示当作监督标签，让模型模仿，等价于“跟着老师走”。  
**在线搜索（Online Search）**：在实际运行时调用外部检索工具（如搜索引擎）获取最新信息，类似人实时上网查资料。

### 核心创新点
1. **引入受导向的 MCTS + 自我批评 → 让 LLM 在每一步先做一次“内部模拟”，再用自评结果修正搜索树的价值估计 → 解决了单纯 MCTS 在高维动作空间里噪声大的问题，使得搜索更聚焦于可行路径。**  
2. **将离线 DPO 与轨迹偏好相结合 → 不是只学习成功轨迹，而是把成功与失败的对比放进优化目标，利用重放缓冲区一次性计算动作的似然比 → 让模型从错误中学习，显著降低了行为克隆的错误累积。**  
3. **构建统一的记忆组件来存储历史动作和页面状态 → 通过显式记忆，模型能够在长任务中保持上下文一致性，避免因页面刷新导致的“忘记”。**  
4. **在真实业务场景（WebShop、在线预订）中加入可选的在线搜索模块 → 当模型对信息不确定时主动查询外部资源，提升了零样本成功率 → 把纯语言模型的知识库与实时网络信息结合，突破了静态知识的局限。

### 方法详解
整体框架可以看作“三层循环”：  
1) **交互采集层**：Agent 在 Web 环境中执行动作，产生完整的状态‑动作‑奖励轨迹，并把每一步的页面快照、用户指令、模型内部思考记录存入离线重放缓冲区。  
2) **搜索‑批评层**：在每一次决策前，模型先用 **受导向的 Monte Carlo Tree Search** 生成若干候选动作序列。每条序列在模拟过程中会产生一段内部推理文本，随后模型对这段文本进行 **自我批评**，输出一个“批评分数”。该分数被反馈给 MCTS，用来调整节点的价值估计（即把批评视作软奖励），从而让搜索更倾向于逻辑自洽的路径。  
3) **离线微调层**：收集到的轨迹被送入 **Direct Preference Optimization**。作者先把每条轨迹按照成功率或批评分数划分为“好”和“坏”。在重放缓冲区里抽取一对好‑坏轨迹，计算模型在两条轨迹上产生的动作概率比值，并直接最大化这个比值（即让模型更倾向输出好轨迹的动作），这一步不需要显式的奖励信号，只依赖偏好对。微调过程每轮只用一天的数据即可显著提升性能。

**关键细节**  
- **受导向的 MCTS**：传统 MCTS 只靠随机模拟，这里把 LLM 的语言生成能力当作“策略先验”，在展开树时先用模型给出高概率动作，再做少量随机扩展。  
- **自我批评的实现**：模型在生成每一步的解释后，接着生成一段“批评”，形式类似“这一步可能遗漏了商品价格信息”。批评分数通过一个小的二分类头得到，训练时使用人工标注的好/坏对。  
- **离线 DPO 的轨迹对比**：因为网页任务往往跨多轮对话，直接对单步奖励不可靠。作者把整条轨迹的成功率（如是否完成购买）作为偏好标签，利用重放缓冲区一次性计算整条轨迹的似然比，省去了逐步奖励回传的复杂性。  
- **记忆组件**：采用键值对存储（键=页面 URL + 动作编号，值=模型上一步的思考向量），在长任务中可以快速检索到之前的上下文，类似于人类的笔记本。

最巧妙的地方在于 **把自我批评的软信号直接注入搜索价值**，这让搜索过程不再是盲目的随机走子，而是带有“逻辑审查”的导向，显著提升了在高维网页动作空间的效率。

### 实验与效果
- **测试平台**：WebShop（模拟电商平台）以及真实的在线预订任务。WebShop 需要模型从搜索商品、加入购物车到结算完成全流程；预订任务则涉及航班、酒店等多步骤查询。  
- **基线对比**：行为克隆（BC）、基于强化学习的 PPO 微调、以及仅使用 MCTS（不带自评）的方案。  
- **主要结果**：在 WebShop 上，Agent Q 的成功率比行为克隆提升约 22%（从 58% 到 80%），比强化学习提升约 15%（从 65% 到 80%）。在真实预订场景中，Llama‑3 70B 的零样本成功率从 18.6% 直接跳到 81.7%（单日数据），加入在线搜索后进一步升至 95.4%。  
- **消融实验**：去掉自我批评后，MCTS 的成功率下降约 8%；仅使用 DPO 而不做轨迹对比，提升幅度只有 5%；去掉记忆组件导致长任务成功率下降约 12%。这些实验表明每个模块都对最终性能有实质贡献。  
- **局限性**：作者指出当前系统仍依赖大量高质量的交互日志，离线 DPO 对偏好标签的质量敏感；在极端稀疏奖励的任务上搜索成本仍然较高；在线搜索虽然提升了成功率，但会引入外部系统的延迟和不确定性。

### 影响与延伸思考
Agent Q 把 **搜索 + 自评 + 偏好学习** 融为一体，为 LLM 在交互式环境中的自我提升提供了可复制的范式。自发表后，已有工作在机器人控制、代码自动化和多模态对话中尝试引入类似的自我批评机制；还有研究把 DPO 与人类偏好标签结合，用于安全对齐（AI alignment）。如果想进一步深入，可以关注以下方向：① 如何在更少数据下实现高效的偏好学习（Few‑shot DPO）；② 将自我批评扩展到多模态感知（图像+文本）；③ 把搜索过程与外部工具（如数据库、API）更紧密地耦合，实现真正的“工具使用型”代理。

### 一句话记住它
**Agent Q 用自我批评驱动的 MCTS + 离线偏好优化，让大语言模型在网页等动态环境里像会反思的棋手一样快速学会正确的多步决策。**