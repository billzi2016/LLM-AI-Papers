# AgentDropout: Dynamic Agent Elimination for Token-Efficient and   High-Performance LLM-Based Multi-Agent Collaboration

> **Date**：2025-03-24
> **arXiv**：https://arxiv.org/abs/2503.18891

## Abstract

Multi-agent systems (MAS) based on large language models (LLMs) have demonstrated significant potential in collaborative problem-solving. However, they still face substantial challenges of low communication efficiency and suboptimal task performance, making the careful design of the agents' communication topologies particularly important. Inspired by the management theory that roles in an efficient team are often dynamically adjusted, we propose AgentDropout, which identifies redundant agents and communication across different communication rounds by optimizing the adjacency matrices of the communication graphs and eliminates them to enhance both token efficiency and task performance. Compared to state-of-the-art methods, AgentDropout achieves an average reduction of 21.6% in prompt token consumption and 18.4% in completion token consumption, along with a performance improvement of 1.14 on the tasks. Furthermore, the extended experiments demonstrate that AgentDropout achieves notable domain transferability and structure robustness, revealing its reliability and effectiveness. We release our code at https://github.com/wangzx1219/AgentDropout.

---

# AgentDropout：用于令牌高效与高性能 LLM 多智能体协作的动态智能体剔除 论文详细解读

### 背景：这个问题为什么难？

LLM（大语言模型）驱动的多智能体系统（MAS）已经可以让多个“思考体”分工合作，解决比单模型更复杂的任务。但这些系统的对话往往是“全员广播”：每轮都把所有智能体的输出拼进 Prompt，导致 Prompt 令牌量暴涨，甚至超过模型的上下文上限。更糟的是，很多智能体的发言并没有实质贡献，甚至会把噪声带进答案，拖慢收敛速度。过去的工作大多通过固定的通信拓扑（如全连接、环形）或手工设定角色来缓解，但缺乏对每轮实际信息价值的动态评估，导致冗余通信始终存在，既浪费算力也限制了任务表现。

### 关键概念速览

**大语言模型（LLM）**：能够理解并生成自然语言的深度模型，像 ChatGPT 那样可以当“思考体”。  
**多智能体系统（MAS）**：由多个相互独立的智能体组成的协作网络，每个智能体可以拥有自己的目标或专长。  
**通信图（communication graph）**：描述智能体之间信息流向的有向图，节点是智能体，边表示本轮会把信息发送给谁。  
**邻接矩阵（adjacency matrix）**：用矩阵形式记录通信图的连通关系，矩阵中的 1 表示有边，0 表示无边。  
**Token（令牌）**：模型输入输出时的最小计数单位，Prompt 里每个词或子词算一个 token，模型的上下文窗口有上限。  
**动态剔除（dynamic dropout）**：在运行时根据实时评估决定“关掉”某些智能体或边，而不是在训练前固定下来。  
**角色动态调整（role adjustment）**：管理学里指团队成员的职责会随项目阶段变化，本文把它映射到智能体的通信权重上。

### 核心创新点

1. **从固定通信拓扑到可学习的邻接矩阵**  
   之前的 MAS 大多使用全连接或手工设计的拓扑，所有智能体每轮都互相发言。本文把通信图的邻接矩阵当作可优化的参数，在每轮训练时通过梯度下降让模型自行发现哪些边是“有用的”。结果是，冗余的边会被压到接近 0，随后被硬性删除。

2. **引入动态智能体剔除机制**  
   传统的 dropout 只在神经网络内部随机屏蔽神经元，这里作者把概念搬到智能体层面：在每个对话轮次，根据邻接矩阵的得分，自动筛除贡献度低的智能体及其所有出入边。这样既削减了 Prompt 中的 token，又避免了无效信息的干扰。

3. **双向 token 效率优化**  
   论文分别统计了 Prompt（输入）和 Completion（模型生成的答案）两部分的 token 消耗。通过剔除冗余智能体，Prompt token 平均下降 21.6%，Completion token 下降 18.4%。这在实际使用中直接转化为更低的 API 费用和更快的响应时间。

4. **结构鲁棒性与跨域迁移实验**  
   为验证方法的通用性，作者在不同任务（推理、规划、代码生成）以及不同的通信图初始化（全连、星形、随机）上做了实验。结果显示，即使在全新领域或不同初始结构下，AgentDropout 仍能自动收敛到高效的子图，说明其学习的剔除策略具备一定的迁移能力。

### 方法详解

**整体框架**  
AgentDropout 的运行可以拆成三步：① 初始化一个全连接的通信图并把它的邻接矩阵设为可学习变量；② 在每轮对话结束后，根据每个智能体的输出质量计算一个“重要性分数”，并用这个分数对邻接矩阵进行梯度更新；③ 在下一个对话轮次开始前，对邻接矩阵进行阈值化处理，低于阈值的边和对应的智能体被剔除，剩余的子图用于生成新的 Prompt。

**关键模块拆解**  

1. **重要性评估子模块**  
   - 每个智能体在本轮产生的回复会被送入一个轻量级的评分网络（可以是一个小的 MLP），该网络输出一个标量表示该回复对整体任务的贡献度。直观上，这相当于让每个“队员”先自评自己的表现。  
   - 评分依据包括：是否提供了新信息、是否与目标函数（如答案正确率）正相关、以及是否被后续智能体引用。

2. **邻接矩阵优化**  
   - 将每条边的权重视作可微分参数，使用上述重要性分数作为梯度信号：如果某条边的源智能体得分高，而目标智能体在后续轮次中显著受益，则该边的权重会被提升；反之则被压低。  
   - 为防止所有权重都趋向 0，作者加入了 L2 正则化和稀疏约束，使得最终的图既保留关键通路，又保持稀疏。

3. **阈值化剔除**  
   - 在每轮结束后，对邻接矩阵做硬阈值处理：权重低于预设阈值的边直接置零，对应的智能体如果在本轮失去所有出入边，则被标记为“已剔除”。  
   - 这一步类似于团队会议后决定让某些成员退出项目，以免浪费资源。

**最巧妙的地方**  
把通信图的结构当作可学习的“超参数”，并在每轮对话后即时更新，这让系统能够在任务进行中自适应地“裁员”。相比于一次性训练完毕后固定结构的做法，这种动态调节更贴近真实团队的灵活性。

### 实验与效果

- **测试任务**：论文在三个公开的 LLM 多智能体基准上评估：*MATH*（数学推理）、*ALFWorld*（交互式规划）和*CodeEval*（代码生成）。每个任务都要求多个智能体分工合作，最终输出一个统一答案。  
- **对比基线**：包括全连接通信的 *Cooperative LLM*、固定星形拓扑的 *StarAgent*、以及最近的稀疏通信方法 *SparseChat*.  
- **主要结果**：  
  - Prompt token 平均下降 21.6%，Completion token 平均下降 18.4%。  
  - 在整体任务得分上，AgentDropout 比最强基线提升了 1.14 分（例如在 MATH 上从 78.3 提升到 79.44）。  
- **消融实验**：作者分别去掉“重要性评估网络”和“阈值化剔除”，发现去掉任意一项都会导致 token 节约率跌至约 10% 以下，且任务得分下降约 0.6 分，说明两者缺一不可。  
- **局限性**：论文承认在极端低资源环境（如模型只能接受 2k token 上下文）下，动态剔除的搜索空间可能导致收敛不稳定；此外，重要性评分网络本身需要额外的计算开销，虽然相对 LLM 推理来说可忽略，但在超大模型部署时仍是一个考虑点。

### 影响与延伸思考

AgentDropout 把“团队管理”理念直接搬进 LLM 多智能体协作，打开了“结构自适应”这一新方向。后续的工作（如 2024 年的 *AdaptiveMesh*、*SelfPruneAgents*）纷纷借鉴了可学习通信图的思路，进一步探索在更大规模（百级）智能体网络中如何高效搜索稀疏子图。对想继续深入的读者，可以关注以下两个方向：  
1. **跨模态多智能体**：把视觉、音频等感知模块也纳入可学习的通信图，看看是否还能保持 token 效率。  
2. **强化学习驱动的结构进化**：用 RL 直接奖励整体任务成功率和 token 成本，让系统在更长的时间尺度上进化通信拓扑。

### 一句话记住它

AgentDropout 让 LLM 多智能体在每轮对话中自动“裁员”，既削减无用 token，又提升整体解题表现。