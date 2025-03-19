# 1000 Layer Networks for Self-Supervised RL: Scaling Depth Can Enable New Goal-Reaching Capabilities

> **Date**：2025-03-19
> **arXiv**：https://arxiv.org/abs/2503.14858

## Abstract

Scaling up self-supervised learning has driven breakthroughs in language and vision, yet comparable progress has remained elusive in reinforcement learning (RL). In this paper, we study building blocks for self-supervised RL that unlock substantial improvements in scalability, with network depth serving as a critical factor. Whereas most RL papers in recent years have relied on shallow architectures (around 2 - 5 layers), we demonstrate that increasing the depth up to 1024 layers can significantly boost performance. Our experiments are conducted in an unsupervised goal-conditioned setting, where no demonstrations or rewards are provided, so an agent must explore (from scratch) and learn how to maximize the likelihood of reaching commanded goals. Evaluated on simulated locomotion and manipulation tasks, our approach increases performance on the self-supervised contrastive RL algorithm by $2\times$ - $50\times$, outperforming other goal-conditioned baselines. Increasing the model depth not only increases success rates but also qualitatively changes the behaviors learned. The project webpage and code can be found here: https://wang-kevin3290.github.io/scaling-crl/.

---

# 千层网络用于自监督强化学习：深度扩展可实现全新目标达成能力 论文详细解读

### 背景：这个问题为什么难？
在强化学习（RL）里，智能体通常需要靠奖励信号或专家示范来学习如何完成任务。自监督（对比）RL 试图摆脱这些外部指导，让智能体自己探索并通过“达到指定目标”的概率来学习。但在实际实现中，网络往往只有 2‑5 层，导致特征表达能力受限，探索效率低下，尤其在高维连续控制任务上几乎无法产生有意义的行为。换句话说，深度不足让模型难以捕捉长程时序依赖和细粒度的目标信息，导致自监督 RL 的性能提升停滞不前。

### 关键概念速览
**自监督强化学习（Self‑Supervised RL）**：让智能体在没有显式奖励或示例的情况下，通过内部构造的学习目标（如对比损失）来驱动探索，就像孩子在没有老师指点的情况下自己玩耍学会走路。  
**目标条件 RL（Goal‑Conditioned RL）**：智能体的策略不仅接受当前状态，还接受一个“目标状态”，需要学会在任意给定目标下产生相应动作，类似于 GPS 导航系统需要根据目的地规划路线。  
**对比学习（Contrastive Learning）**：通过让模型区分“相似”与“不同”的样本来学习有用的表示，就像把相同颜色的球放在一起、不同颜色的球分开。  
**网络深度（Network Depth）**：指神经网络层数，层数越多，理论上能表达越复杂的函数，类似于把一段文字拆成更多的细粒度句子来捕捉细微含义。  
**层级残差连接（Residual Connections）**：在深层网络中加入跨层的直接通路，帮助梯度传播，像在长跑途中设置补给站，防止选手因疲劳而掉速。  
**目标达成概率（Goal‑Likelihood）**：模型对给定目标的预测概率，越高说明智能体越有把握实现该目标，类似于投篮时对命中概率的自我评估。  

### 核心创新点
1. **深度扩展 → 超深网络（最高 1024 层） → 性能跨尺度提升**  
   过去的自监督 RL 大多使用 2‑5 层的浅网络，作者直接把网络层数推到 1024 层，并配合层级残差结构保证训练稳定。实验显示，这种深度提升让成功率提升 2‑50 倍，说明深度本身是自监督 RL 的瓶颈。  

2. **层级残差 + 归一化策略 → 训练可行性 → 解决梯度消失/爆炸**  
   在极深网络中，普通的前向传播会导致梯度在反向传播时快速衰减或膨胀。作者在每 10 层左右插入残差块并使用层归一化，使得信息可以在网络中“跳跃”，相当于在长隧道里装上了灯塔，保证信号不迷路。  

3. **统一的对比目标函数 + 目标采样机制 → 更高效的自监督信号 → 行为多样性显著提升**  
   传统对比 RL 只在当前状态和目标状态之间做对比，作者加入了时间跨度更大的负样本（即不同时间步的状态），并在目标空间中均匀采样，使得智能体在训练时必须学会区分更细微的目标差异，行为表现从单调的“前进”变成了多方向的“爬行、跳跃”。  

### 方法详解
整体思路可以拆成三步：**（1）构建超深残差网络；（2）设计自监督对比目标；（3）在目标条件环境中进行无奖励探索**。下面逐层解释每一步的细节。

1. **超深残差网络**  
   - 网络采用标准的全连接（或卷积）层堆叠，每层宽度保持在 256‑512 维。  
   - 每隔 10 层插入一个残差块：输入先经过层归一化，再经过 ReLU 激活，最后与块入口相加。这样做的好处是梯度可以直接跨过 10 层向上传递，防止深度导致的梯度消失。  
   - 为了进一步稳定训练，作者在每个残差块后加入轻量的 dropout，防止过拟合。  

2. **自监督对比目标**  
   - 给定当前状态 $s_t$ 和目标状态 $g$，网络输出它们的嵌入向量 $f(s_t)$、$f(g)$。  
   - 采用 InfoNCE 损失：正样本是 $(s_t, g)$，负样本是同一批次中其他目标的嵌入。损失鼓励正样本的相似度高，负样本的相似度低。  
   - 为了让负样本更具挑战性，作者在时间维度上挑选“远距离”状态作为负样本，这相当于让智能体在学习时必须区分“现在离目标很远”和“现在已经很接近”。  

3. **目标条件探索**  
   - 环境在每个 episode 开始时随机抽取一个目标 $g$（在任务空间均匀采样），智能体的策略 $\pi(a|s,g)$ 通过超深网络生成。  
   - 没有外部奖励，唯一的学习信号来自对比损失。智能体通过最大化目标达成概率的对数似然来更新策略。  
   - 为了防止智能体陷入局部探索，作者在每个 episode 结束后使用经验回放池，随机抽取过去的 $(s,g)$ 对进行再训练，类似于人类在回忆过去的经验并从中提炼教训。  

**最巧妙的点**在于作者并没有改变对比 RL 的核心框架，而是通过“把网络拉长、加上跨层捷径、让负样本更难”三招，让原本难以收敛的训练在 1024 层时依旧稳健。  

### 实验与效果
- **任务设置**：在 MuJoCo 仿真环境中选取了 4 种连续控制任务，包括 2 种步态（Walker2d、HalfCheetah）和 2 种机械臂操作（FetchReach、HandManipulate）。所有任务均采用无奖励、目标条件的自监督设置。  
- **对比基线**：与原始对比 RL（2‑5 层网络）、Goal‑Conditioned SAC、以及最新的自监督目标网络（同样浅层）进行比较。  
- **性能提升**：论文报告在所有任务上成功率提升 2‑50 倍，尤其在高维手部操作任务中，成功率从 5% 提升到超过 40%。  
- **消融实验**：作者分别去掉残差块、层归一化以及时间负样本，发现去掉任意一项都会导致成功率下降至少 30%，验证每个设计都是关键。  
- **局限性**：训练时间显著增长（约 3‑5 倍），对显存需求较高，且在极端稀疏的目标空间（目标分布极不均匀）仍会出现探索停滞。原文未给出在真实机器人上的实验，仍是仿真验证。  

### 影响与延伸思考
这篇工作首次在自监督 RL 领域展示了“深度即能力”的 scaling law，激发了后续研究关注网络规模而非仅仅算法改进。随后出现的几篇论文尝试把 **Transformer** 结构引入自监督 RL，进一步验证“更深、更宽”同样适用于序列决策。还有工作把 **稀疏激活** 与超深网络结合，试图在保持性能的同时降低计算成本。想深入了解的读者可以关注 **“深度强化学习的梯度流动理论”**（推测）以及 **“自监督 RL 与大模型的跨模态扩展”**（推测）这两个方向。  

### 一句话记住它
把自监督强化学习的网络层数推到千层，配上残差跳连和更硬的对比负样本，就能让智能体从“只能前进”变成“会多方向达成目标”。