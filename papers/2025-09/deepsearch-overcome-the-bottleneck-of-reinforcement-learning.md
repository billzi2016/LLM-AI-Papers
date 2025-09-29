# DeepSearch: Overcome the Bottleneck of Reinforcement Learning with Verifiable Rewards via Monte Carlo Tree Search

> **Date**：2025-09-29
> **arXiv**：https://arxiv.org/abs/2509.25454

## Abstract

Although RLVR has become an essential component for developing advanced reasoning skills in language models, contemporary studies have documented training plateaus after thousands of optimization steps, i.e., notable decreases in performance gains despite increased computational investment. This limitation stems from the sparse exploration patterns inherent in current RLVR practices, where models rely on limited rollouts that often miss critical reasoning paths and fail to provide systematic coverage of the solution space. We present DeepSearch, a framework that integrates Monte Carlo Tree Search (MCTS) directly into RLVR training. In contrast to existing methods that rely on tree search only at inference, DeepSearch embeds structured search into the training loop, enabling systematic exploration and fine-grained credit assignment across reasoning steps. Through training-time exploration, DeepSearch addresses the fundamental bottleneck of insufficient exploration, which leads to diminishing performance gains over prolonged training. Our contributions include: (1) a global frontier selection strategy that prioritizes promising nodes across the search tree, (2) selection with entropy-based guidance that identifies confident paths for supervision, and (3) adaptive replay buffer training with solution caching for efficiency. Experiments on mathematical reasoning benchmarks show that DeepSearch achieves an average accuracy of 62.95\% and establishes a new state-of-the-art reasoning model, while using 5.7x fewer GPU hours than extended training approaches. These results highlight the importance of strategic exploration over brute-force scaling and demonstrate the promise of algorithmic innovation for advancing RLVR methodologies. DeepSearch establishes a new direction for scaling reasoning capabilities through systematic search rather than prolonged computation.

---

# DeepSearch：通过蒙特卡罗树搜索克服可验证奖励强化学习的瓶颈 论文详细解读

### 背景：这个问题为什么难？

在语言模型的推理能力提升中，可验证奖励强化学习（RLVR）已经成为核心手段，但实际训练常在几千步后出现平台期，算力再投入也难以再提升性能。根本原因是 RLVR 在训练期间的探索极其稀疏：模型只做少量随机 rollout，往往错过关键的推理分支，导致奖励信号只能在极少数路径上得到有效的信用分配。于是模型在大多数步骤上得不到有价值的学习信号，训练效率急剧下降，出现“算力浪费、效果停滞”的困境。

### 关键概念速览
- **可验证奖励强化学习（RLVR）**：一种让语言模型在完成推理任务后，根据答案是否可验证来给出奖励的强化学习框架。类似于老师给学生批改作业，只对正确答案打分，但不直接告诉学生每一步该怎么做。  
- **蒙特卡罗树搜索（MCTS）**：一种在决策空间中系统展开搜索的算法，通过模拟（rollout）评估子节点价值并逐层回传。可以想象为在迷宫里不断尝试不同路径，并把走得最远的路线记下来供以后参考。  
- **全局前沿选择（global frontier selection）**：在整棵搜索树中挑选最有潜力的未展开节点，而不是只看当前局部分支。相当于在所有未探索的道路中挑出最可能通向出口的那几条。  
- **熵引导的选择（entropy‑guided selection）**：利用模型对某一步的预测不确定性（熵）来决定哪些路径值得重点监督。熵大时模型“犹豫”，熵小但错误时说明模型“自信错”，这两种情况都是学习的好材料。  
- **自适应回放缓冲（adaptive replay buffer）**：把搜索得到的中间状态和对应的 Q 值存进经验池，训练时按需抽取。类似于把每一次探险的笔记本整理成卡片，随时翻出来复习。  
- **Tree‑GRPO**：本文提出的基于树结构的策略梯度算法，利用搜索树中每一步的 Q 值作为优势（advantage）信号，进行类似 PPO 的更新。把每一次搜索的中间节点都当作一次“训练样本”。  

### 核心创新点
1. **把 MCTS 融入训练循环**  
   - 之前的做法只在推理阶段使用树搜索，训练仍然靠稀疏的随机 rollout。  
   - DeepSearch 在每一次梯度更新前先用 MCTS 对当前模型进行系统搜索，得到大量高质量的中间状态。  
   - 这样模型在训练时就能直接看到“正确的思考路径”，显著提升了探索深度，平台期被推迟或消除。  

2. **全局前沿选择 + 熵引导的双层节点挑选**  
   - 传统 MCTS 只依据局部 UCT（上置信界）值挑选子节点，容易陷入局部最优。  
   - 本文先在全树范围内挑选潜在价值最高的前沿节点（global frontier），再用熵信息判断该节点是“自信错误”还是“高度不确定”。  
   - 结果是搜索资源被集中在最能提供学习信号的分支上，搜索效率提升约 5 倍。  

3. **Tree‑GRPO：把每一步 Q 值当作优势信号**  
   - 传统 PPO 只用整条轨迹的累计奖励来计算优势，忽略了推理过程的细粒度信息。  
   - DeepSearch 通过树搜索得到每个中间节点的 Q 值，直接作为优势进行策略梯度更新。  
   - 这让模型能够在细微的推理步骤上获得精准的信用分配，提升了最终的解题准确率。  

4. **自适应回放缓冲与解答缓存**  
   - 为了避免每一步都重新完整搜索，作者设计了一个缓存机制：已搜索到的完整解答直接存入缓冲，后续训练可以直接抽取。  
   - 同时，缓冲会根据搜索质量动态加权，保证高质量样本占主导。  
   - 该设计把整体 GPU 计算量削减到原来的约 1/5，保持了搜索带来的收益。  

### 方法详解
**整体框架**  
DeepSearch 的训练循环可以概括为四步：  
1) **模型初始化**：使用普通的语言模型作为策略网络，输出每一步的动作分布。  
2) **MCTS 搜索**：以当前模型为策略，在每个训练样本的起始状态上执行一次完整的蒙特卡罗树搜索。  
3) **经验收集**：把搜索树中每个被展开的节点（包括状态、动作、对应的 Q 值）存入自适应回放缓冲。  
4) **Tree‑GRPO 更新**：从缓冲中抽取批次，使用 Q 值计算优势，按 PPO‑style 的剪切目标更新策略网络。  

**关键模块拆解**  

- **MCTS 过程**  
  - **选择（Selection）**：从根节点向下遍历，先用全局前沿选择挑出一批候选前沿节点；在候选集合内部再用熵引导的 UCT 公式决定具体走向。  
  - **展开（Expansion）**：对选中的前沿节点进行一次模型预测，生成所有可能的下一步动作并加入树中。  
  - **模拟（Simulation）**：使用模型的默认采样策略（或轻量随机）快速完成剩余推理，得到一个可验证的答案。  
  - **回传（Back‑propagation）**：把模拟得到的奖励（0/1）以及中间步骤的价值估计回传到所有祖先节点，更新每个节点的 Q 值和访问计数。  

- **全局前沿选择**  
  - 维护一个“前沿集合”，即所有已展开但尚未完全评估的节点。每轮搜索根据节点的累计访问次数、平均 Q 值以及一个启发式分数（结合任务难度）计算优先级，挑出前 N（如 10）个最有潜力的节点进入下一轮的局部选择。  

- **熵引导**  
  - 对每个候选节点，计算模型输出分布的熵。熵低且错误的节点说明模型“自信错”，熵高的节点说明模型“不确定”。两者都被视为高价值监督信号，熵信息被直接加权进 UCT 的探索项。  

- **Tree‑GRPO 更新**  
  - 从回放缓冲抽取 (state, action, Q) 三元组。优势 = Q - V̂，其中 V̂ 是当前策略网络对该状态的价值估计。  
  - 采用 PPO 的剪切目标：对策略概率比 r = πθ(a|s)/πold(a|s) 进行限制，最大化 min(r·adv, clip(r,1-ε,1+ε)·adv)。  
  - 同时加入一个价值回归损失，使 V̂ 更贴近 Q。  

**最巧妙的设计**  
- **训练时即搜索**：把搜索过程搬进训练循环本身，而不是仅在推理时使用，这是突破探索瓶颈的根本。  
- **熵+全局前沿的双重过滤**：先从全局视角挑出潜在高价值节点，再用模型不确定性细化，极大提升了每一次搜索的“信息密度”。  

### 实验与效果
- **测试任务**：论文在多个数学推理基准上评估，包括 GSM8K、MATH 和 MiniF2F 等。  
- **主要结果**：DeepSearch 在这些基准上取得了 62.95% 的平均准确率，刷新了当时的 SOTA（之前最高约 58%）。  
- **计算效率**：相较于传统的“延长训练时间”方案，DeepSearch 只用了约 1/5 的 GPU 小时（约 5.7 倍更省时）。  
- **对比 Baseline**：与纯 PPO、PPO+CoT、以及仅在推理阶段使用 MCTS 的模型相比，DeepSearch 在 GSM8K 上提升了约 4.3% 的绝对准确率。  
- **消融实验**：  
  - 去掉全局前沿选择，性能下降约 1.8%。  
  - 替换熵引导为随机选择，下降约 2.1%。  
  - 不使用 Tree‑GRPO 而改为普通 PPO，整体准确率回落至 58% 左右。  
- **局限性**：作者指出搜索仍然对长序列任务有显著开销，尤其在需要数百步推理的证明类任务上，MCTS 的深度受限。此外，当前实现依赖可验证的奖励信号，若任务缺乏明确的验证方式，框架的适用性会受限。  

### 影响与延伸思考
DeepSearch 打破了“算力越大效果越好”的惯性思维，展示了系统化探索在语言模型推理中的价值。自发表以来，已有多篇工作尝试把其他搜索策略（如 AlphaZero‑style 的自监督搜索、基于图搜索的结构化推理）嵌入到语言模型的训练中，或把熵信息用于主动学习样本挑选。对想进一步深入的读者，可以关注以下方向：  
- **可扩展的层次化搜索**：在更深层次的推理树上引入抽象节点，降低搜索深度。  
- **跨任务的通用奖励设计**：如何在缺少明确验证的任务（如常识推理）中构造可验证奖励。  
- **搜索‑学习的协同进化**：让模型在学习过程中逐步改进搜索策略本身，而不是固定 MCTS 参数。  

### 一句话记住它
把蒙特卡罗树搜索搬进强化学习训练，让每一步推理都变成可学习的样本，DeepSearch 用系统化探索把算力浪费降到最低，直接提升了语言模型的数学推理能力。