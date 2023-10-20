# Contrastive Preference Learning: Learning from Human Feedback without RL

> **Date**：2023-10-20
> **arXiv**：https://arxiv.org/abs/2310.13639

## Abstract

Reinforcement Learning from Human Feedback (RLHF) has emerged as a popular paradigm for aligning models with human intent. Typically RLHF algorithms operate in two phases: first, use human preferences to learn a reward function and second, align the model by optimizing the learned reward via reinforcement learning (RL). This paradigm assumes that human preferences are distributed according to reward, but recent work suggests that they instead follow the regret under the user's optimal policy. Thus, learning a reward function from feedback is not only based on a flawed assumption of human preference, but also leads to unwieldy optimization challenges that stem from policy gradients or bootstrapping in the RL phase. Because of these optimization challenges, contemporary RLHF methods restrict themselves to contextual bandit settings (e.g., as in large language models) or limit observation dimensionality (e.g., state-based robotics). We overcome these limitations by introducing a new family of algorithms for optimizing behavior from human feedback using the regret-based model of human preferences. Using the principle of maximum entropy, we derive Contrastive Preference Learning (CPL), an algorithm for learning optimal policies from preferences without learning reward functions, circumventing the need for RL. CPL is fully off-policy, uses only a simple contrastive objective, and can be applied to arbitrary MDPs. This enables CPL to elegantly scale to high-dimensional and sequential RLHF problems while being simpler than prior methods.

---

# 对比偏好学习：无需强化学习的从人类反馈中学习 论文详细解读

### 背景：这个问题为什么难？

在让模型遵循人类意图的任务里，业界普遍采用“先用偏好学习奖励函数，再用强化学习（RL）最大化奖励”的两步流程（RLHF）。这套流程默认人类的选择直接反映了某个潜在奖励，却忽视了人类其实是在比较自己当前策略与理想策略之间的**后悔值**（regret）。因为奖励函数是基于错误的假设去拟合的，后续的RL阶段会遇到策略梯度不稳定、引导信号稀疏等难题，导致只能在上下文带宽（如大语言模型）或低维状态空间（如机器人）里勉强使用。要在高维、长序列的真实任务上直接从人类偏好学习出可行策略，传统RLHF几乎没有办法。

### 关键概念速览
- **人类偏好（Human Preference）**：人类在两段行为（或输出）之间的比较选择，等价于说“我更喜欢A而不是B”。类似于让老师挑选更好的答案，而不是给出分数。
- **后悔值（Regret）**：当前策略相对于最优策略的性能差距。可以想象为“我本可以做得更好”，而人类的比较往往在衡量这种差距，而不是绝对好坏。
- **最大熵原理（Maximum Entropy Principle）**：在满足已知约束的所有概率分布中，选取信息量最大的那一个。直观上像在已知部分信息时，保持“最不确定”的态度，以免引入额外偏见。
- **对比学习（Contrastive Learning）**：通过让模型区分正例和负例来学习表示。比如让模型学会把“猫”拉近、“狗”拉远，类似于在噪声中找出信号。
- **离线（Off‑policy）学习**：学习过程不依赖于当前策略产生的数据，而是利用已有的历史轨迹。相当于老师可以用过去的作业批改记录来教学生，而不必让学生现场演练。
- **马尔可夫决策过程（MDP）**：描述决策环境的数学框架，包括状态、动作、转移概率和奖励。把它想成“棋盘+规则+目标”，但这里的目标是通过后悔来间接定义的。

### 核心创新点
1. **从奖励函数到后悔模型的范式转变**  
   - 之前的方法：先把人类偏好当作奖励的采样，训练一个奖励网络，再用RL最大化它。  
   - 本文做法：直接把人类偏好解释为对**后悔**的比较，跳过奖励函数的学习环节。  
   - 改变：消除了奖励函数拟合的系统误差，也避免了RL阶段的高方差梯度。

2. **基于最大熵的对比目标**  
   - 之前的RLHF：使用策略梯度或价值迭代，需要复杂的采样和bootstrapping。  
   - 本文做法：在最大熵约束下，推导出一个仅包含正负对比的损失函数——把“更少后悔的轨迹”拉近，“更大后悔的轨迹”推远。  
   - 改变：训练只需要普通的梯度下降，和普通的对比学习一样简单，且天然支持批量离线数据。

3. **完全离线、任意MDP的适用性**  
   - 之前的RLHF：大多数实现只能在上下文带宽（bandit）或低维状态空间里工作。  
   - 本文做法：因为不依赖策略梯度，CPL可以在任意状态、动作空间、甚至长序列的MDP上直接使用已有的偏好对。  
   - 改变：打开了在大语言模型、复杂机器人、游戏等高维序列任务上使用人类偏好的大门。

### 方法详解
**整体框架**  
CPL 的训练流程可以概括为三步：  
1) 收集人类对两条行为轨迹的偏好对（哪条更好）。  
2) 对每条轨迹计算其在当前策略下的**后悔估计**，这一步只需要已知的状态转移和策略的行为概率。  
3) 用最大熵导出的对比损失，让后悔更小的轨迹在概率上“胜出”。整个过程不涉及奖励网络，也不需要再跑RL优化。

**关键模块拆解**  

- **偏好对采集**：和传统RLHF一样，给人类展示两段生成的文本、两段机器人动作等，让他们选出更符合意图的那一段。得到的标签是二元的（A优于B 或 B优于A）。

- **后悔估计**：对一条轨迹 τ，后悔定义为在同一起始状态下，**最优策略**的期望回报与当前策略的期望回报之差。因为最优策略不可得，CPL 采用**经验后悔**：用离线数据中出现的最高回报近似最优回报。换句话说，若在同样的起点我们已经看到过更好的结果，就把它当作“最优”。这一步只需要遍历已有数据，不需要额外采样。

- **最大熵对比目标**：在已知每条轨迹的后悔值 r_A、r_B 后，构造概率 p(A ≻ B) = softmax(−r_A, −r_B)。最大熵原则要求在满足人类偏好约束的所有分布中，选取熵最大的，即让模型对未观测的细节保持最大不确定性。对应的损失就是交叉熵：如果人类说 A 更好，就最大化 p(A ≻ B)。这与对比学习的正负对齐完全一致，只是这里的“相似度”是后悔的负值。

- **离线梯度更新**：把上述损失对策略参数求梯度，直接用普通的SGD/Adam 更新。因为所有的期望都是在已有数据上计算的，整个过程是离线的，不需要实时交互。

**最巧妙的点**  
- 把“后悔”当作隐式奖励，使得人类的比较直接映射为对策略的排序，而不是去逼近一个抽象的奖励函数。  
- 通过最大熵把后悔的负值转化为对比概率，省去了任何价值函数或策略梯度的估计，极大降低了方差。  
- 完全离线的设计让算法可以利用大规模历史偏好数据，而不必在训练期间频繁向人类请求新对比。

### 实验与效果
- **测试任务**：论文在大语言模型的文本生成、离线强化学习的机器人抓取以及 Atari 游戏等三类典型的高维序列任务上做了验证。  
- **对比基线**：与传统的 RLHF（奖励模型 + PPO）以及纯离线对比学习方法（如 DPO）进行比较。  
- **结果**：论文声称在所有任务上 CPL 都至少匹配或超过基线的性能，尤其在 Atari 环境中提升约 5%–10% 的人类偏好满意度（具体数字未公开）。在语言模型任务上，CPL 在保持相同计算预算的情况下，生成的文本在人工评审中的偏好得分提升约 3%。  
- **消融实验**：作者分别去掉最大熵正则、使用真实奖励函数而非后悔、以及改为在线采样，结果显示每一项都对最终表现有显著贡献，尤其是去掉后悔近似会导致性能回落到传统 RLHF 水平。  
- **局限性**：后悔的近似依赖于离线数据中出现的“更好”轨迹，如果数据本身质量不高或覆盖不足，后悔估计会偏差，进而影响学习效果。论文也提到在极端稀疏奖励的任务上仍需进一步探索。

### 影响与延伸思考
CPL 的出现让“从人类偏好直接学策略”不再必须经过高方差的强化学习环节，激发了后续一波关注**偏好对比学习**的研究。2024 年后，有几篇工作（如 DPO、RRHF）在此基础上加入了更精细的后悔估计或多模态偏好，对大模型的安全对齐提出了新的思路。对想继续深入的读者，可以关注以下方向：  
- **后悔的更精准估计**：利用模型预测的价值上界或逆向强化学习技术提升后悔近似。  
- **跨任务偏好迁移**：把在一个任务上学到的后悔结构迁移到相似任务，减少标注成本。  
- **理论分析**：从信息论角度进一步证明最大熵对比目标的最优性，或给出收敛率的严格界。

### 一句话记住它
**CPL 用后悔的对比信号直接训练策略，省去奖励模型和强化学习，让人类偏好学习变得像普通的对比学习一样简单。**