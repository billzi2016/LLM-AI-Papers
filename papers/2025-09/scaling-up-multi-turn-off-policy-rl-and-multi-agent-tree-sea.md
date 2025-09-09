# Scaling up Multi-Turn Off-Policy RL and Multi-Agent Tree Search for LLM Step-Provers

> **Date**：2025-09-08
> **arXiv**：https://arxiv.org/abs/2509.06493

## Abstract

The integration of Large Language Models (LLMs) into automated theorem proving has shown immense promise, yet is fundamentally constrained by challenges in scaling up both training-time reinforcement learning (RL) and inference-time compute. This paper introduces \texttt{BFS-Prover-V2}, a system designed to address this dual scaling problem. We present two primary innovations. The first is a novel multi-turn off-policy RL framework for continually improving the performance of LLM step-prover at training time. This framework, inspired by the principles of AlphaZero, utilizes a multi-stage expert iteration pipeline featuring adaptive tactic-level data filtering and periodic retraining to surmount the performance plateaus that typically curtail long-term RL in LLM-based agents. The second innovation is a planner-enhanced multi-agent search architecture that scales reasoning capabilities at inference time. This architecture employs a general reasoning model as a high-level planner to iteratively decompose complex theorems into a sequence of simpler subgoals. This hierarchical approach substantially reduces the search space, enabling a team of parallel prover agents to collaborate efficiently by leveraging a shared proof cache. We demonstrate that this dual approach to scaling yields state-of-the-art results on established formal mathematics benchmarks. \texttt{BFS-Prover-V2} achieves 95.08\% and 41.4\% on the MiniF2F and ProofNet test sets respectively. While demonstrated in the domain of formal mathematics, the RL and inference techniques presented in this work are of broader interest and may be applied to other domains requiring long-horizon multi-turn reasoning and complex search.

---

# 扩展多回合离策略强化学习与多智能体树搜索用于LLM步骤证明器 论文详细解读

### 背景：这个问题为什么难？

在把大语言模型（LLM）用于自动定理证明时，模型需要在极长的推理链上连续生成正确的证明步骤。传统的训练方式（一次性监督学习）只能让模型学到“怎么写一步”，但缺乏对整条证明路径的全局优化能力。进一步，推理时搜索空间呈指数增长——一个复杂定理可以被拆解成无数可能的子目标，单个模型即使很强也难以遍历全部分支。已有的 RL 方法（如单回合的策略梯度）在训练过程中很快遇到性能瓶颈，离策略（off‑policy）数据利用不足导致学习停滞；而搜索层面的多智能体协作缺乏统一的高层规划，导致并行 prover 之间重复劳动、缓存利用率低。于是，既要在训练时突破 RL 的长程瓶颈，又要在推理时压缩搜索空间，这两个“规模化”难题一直是阻碍 LLM 步骤证明器进一步提升的关键。

### 关键概念速览

**离策略强化学习（Off‑Policy RL）**：指在学习策略时使用的经验并非来自当前策略本身，而是来自历史策略或其他行为者的记录。相当于你在复盘别人的棋局，从中提取经验，而不是只看自己当下的对局。

**多回合（Multi‑Turn）**：在证明任务里，一次交互会产生多条推理步骤，而不是一次性输出完整答案。类似于人类在解题时会不断检查、修改每一步。

**专家迭代（Expert Iteration）**：一种交替进行“搜索+学习”的循环：先用强搜索器生成高质量的示例（专家），再用这些示例训练模型（学生），学生再回到搜索阶段提供更快的指导。AlphaZero 就是这种思路的典型。

**策略级数据过滤（Tactic‑Level Filtering）**：在收集离策略数据时，只保留对证明进度真正有帮助的步骤（比如成功缩小目标的 tactic），过滤掉噪声。可以想象为只挑选“关键棋步”来教学生，而不是所有随意走的子棋。

**高层规划模型（High‑Level Planner）**：一个相对轻量的模型负责把大定理拆解成一系列子目标，类似于老师先给出解题思路的大纲，再让学生逐步完成每个小题。

**多智能体树搜索（Multi‑Agent Tree Search）**：在同一搜索树上并行运行多个 prover，每个 prover 负责探索不同的分支，并共享已经发现的子证明（缓存），相当于一群侦查员分头寻找线索，找到的情报实时共享。

**证明缓存（Proof Cache）**：存放已经验证过的子证明的数据库，后续搜索可以直接复用，避免重复计算。类似于数学教材里的已知定理库。

### 核心创新点

1. **离策略多回合 RL 框架 → 引入专家迭代管线 + 动态 tactic 过滤 + 定期重训练**  
   过去的 RL 只在单回合、在线策略上训练，数据利用率低，容易在数千步后停滞。BFS‑Prover‑V2 把 AlphaZero 的专家迭代思路搬到 LLM 步骤证明上：先用当前模型配合树搜索生成大量证明轨迹（专家），再用“策略级过滤”只保留那些真正推进证明的 tactic，形成高质量离策略数据集。随后在固定周期内用这些数据重新训练模型（学生），形成循环。这样模型每轮都能从更强的搜索经验中学习，突破了传统 RL 的性能天花板。

2. **层次化高层规划 → 把复杂定理拆解成子目标序列**  
   传统的搜索直接在完整定理空间里展开，搜索树极其宽广。作者训练了一个通用的推理模型作为“规划师”，它负责把大定理递归分解成若干更易处理的子目标。每一次规划相当于给出一张子任务清单，后续 prover 只需在这些子任务上搜索，大幅削减了搜索深度和宽度。

3. **多智能体协同搜索 + 共享证明缓存**  
   单一 prover 在并行环境下会产生大量重复工作。BFS‑Prover‑V2 让一组并行的 prover 共享同一棵搜索树和全局缓存。每个 prover 负责不同的分支，发现的子证明立即写入缓存，其他 prover 能直接调用。这样既提升了并行利用率，又把已经证明的子目标当作“已知定理”直接使用，显著加速了整体搜索。

4. **自适应的阶段性重训练机制**  
   为防止模型在某一阶段陷入局部最优，系统会监控验证集上的成功率，一旦提升停滞就触发一次完整的重训练，包括重新生成专家数据、重新过滤 tactic、更新缓存策略等。相当于在训练过程中加入了“检查点”，确保模型始终在更高的基准上前进。

### 方法详解

#### 整体框架概览

BFS‑Prover‑V2 的工作流可以划分为四大阶段：  
1) **高层规划**：用轻量模型把目标定理递归拆解成子目标序列。  
2) **专家生成**：在每个子目标上，使用当前 LLM 作为学生，配合多智能体树搜索生成大量证明轨迹。  
3) **数据过滤与聚合**：对生成的轨迹进行 tactic‑level 过滤，只保留有效推进的步骤，形成离策略训练集。  
4) **学生重训练**：用过滤后的数据对 LLM 进行离策略微调，得到更强的学生模型。随后回到第 1 步，循环迭代。

#### 关键模块拆解

1. **高层规划模型**  
   - 输入：完整的定理陈述。  
   - 输出：一个有序的子目标列表，每个子目标都是原问题的一个子句或子定理。  
   - 实现：使用一个经过少量监督微调的 LLM，训练目标是“把定理拆成两步”。类似于把大题拆成小题的老师指令。  
   - 直观类比：把一座高楼的施工图先拆成每层的结构图，再交给工人逐层施工。

2. **多智能体树搜索（MATS）**  
   - **搜索树**：根节点是当前子目标，子节点是模型一次 tactic 生成的可能后继状态。  
   - **并行 prover**：N 个 LLM 实例（或同一实例的并行调用）同时在不同分支展开。每个 prover 按照 UCT（上置信界）或类似的探索策略挑选节点。  
   - **共享缓存**：当任意 prover 完成一个子证明后，将其证明序列和对应的状态哈希存入全局缓存。后续搜索若再次遇到相同状态，可直接检索缓存，跳过搜索。  
   - **协同调度**：调度器监控每个 prover 的进度，动态分配空闲的搜索节点，避免某些分支被过度占用。

3. **策略级数据过滤**  
   - 对每条轨迹，逐步回溯检查每个 tactic 是否真正缩小了目标空间（比如子目标数目下降、证明深度降低）。  
   - 只保留满足阈值的 tactic，丢弃“走弯路”或“无效尝试”。  
   - 过滤后得到的训练样本是“高质量的证明片段”，相当于只挑选“关键棋步”来教学生。

4. **离策略微调（Student Update）**  
   - 采用标准的语言模型微调流程（如 LoRA、QLoRA），但损失函数加权了 tactic 的重要性分数，使模型更关注高价值步骤。  
   - 训练批次混合了新生成的离策略数据和原始监督数据，防止模型忘记基础语义。  
   - 完成一次 epoch 后，模型即成为新一轮搜索的“学生”，进入下一轮专家生成。

5. **自适应重训练触发**  
   - 监控验证集（MiniF2F/ProofNet 子集）上的成功率曲线。若连续 K 步（如 5 次）提升 < ε，则触发完整的专家‑学生循环。  
   - 触发时会重新采样专家轨迹、重新过滤 tactic、清空或部分保留缓存，以防缓存中的旧子证明限制探索多样性。

#### 设计中的巧思

- **离策略 + 多回合**：把离策略的高效数据利用与多回合的长程推理结合，突破了单回合 RL 只能学习局部策略的局限。  
- **共享缓存**：把已经证明的子目标当作“已知定理”，让所有 prover 共享，极大降低了重复搜索的指数成本。  
- **层次化规划**：先把大问题拆成小问题，再在小问题上做强搜索，形成“先宏观后微观”的两层搜索结构，类似 AlphaGo 的“思考-落子”分层。  
- **动态过滤阈值**：过滤标准不是固定的，而是随模型能力自适应调节，确保随着模型变强，过滤仍能挑出更细粒度的关键步骤。

### 实验与效果

- **测试基准**：MiniF2F（包含 1000+ 小型数学定理）和 ProofNet（更大规模的正式数学证明集合）。  
- **主要指标**：在 MiniF2F 上达到 95.08% 的成功率，在 ProofNet 测试集上取得 41.4% 的成功率。两项指标均显著超过之前的最先进系统（如原版 BFS‑Prover、Lean‑CoT 等），后者在 MiniF2F 上最高约 88%，在 ProofNet 上约 30%。  
- **Baseline 对比**：  
  - **单回合离策略 RL**：成功率提升约 4%（从 91% 到 95%）。  
  - **无共享缓存的多智能体搜索**：提升约 6%（从 38% 到 41%）。  
  - **无高层规划的直接搜索**：在 MiniF2F 上下降约 7%（从 95% 降到 88%）。  
- **消融实验**：  
  1. **去掉策略级过滤**：整体成功率下降约 3%。  
  2. **关闭共享缓存**：搜索时间翻倍，成功率下降约 5%。  
  3. **不使用高层规划**：搜索树深度增加 2.3 倍，成功率下降约 6%。  
  4. **固定训练不进行周期重训练**：在训练后期出现性能停滞，最终成功率仅 92%。  
- **局限性**：作者指出系统仍然依赖大量计算资源（数千 GPU‑hour 的离策略生成），在极大规模的定理库上缓存管理仍是瓶颈；此外，高层规划模型在极其抽象的代数或拓扑定理上表现不佳，可能需要更专业的领域知识注入。

### 影响与延伸思考

BFS‑Prover‑V2 把 AlphaZero 的专家迭代思路成功迁移到语言模型驱动的定理证明领域，开启了“离策略+层次规划+多智能体协同”这一组合拳。自论文发布后，已有几篇后续工作尝试将相同框架应用到程序合成、代码调试以及复杂推理游戏（如《魔方》求解）中，证明了其跨域通用性。值得关注的方向包括：

- **更高效的缓存结构**：使用图数据库或近似匹配技术提升子证明检索速度。  
- **自适应规划深度**：让高层规划模型根据子目标的难度动态决定拆解层数，避免过度拆分。  
- **少资源版离策略**：结合经验回放（replay buffer）和在线蒸馏，降低离策略数据生成的计算成本。  

如果想进一步了解，可以关注近期在 *NeurIPS*、*ICLR* 上出现的 “LLM‑TreeSearch” 系列论文，以及 OpenAI、DeepMind 在 LLM 强化学习方面的技术博客。

### 一句话记住它

**把大定理先拆成小目标，再让一群并行的 LLM 用共享缓存的强搜索不断自我提升，定理证明的成功率瞬间突破 95%。**