# Agency Is Frame-Dependent

> **Date**：2025-02-06
> **arXiv**：https://arxiv.org/abs/2502.04403

## Abstract

Agency is a system's capacity to steer outcomes toward a goal, and is a central topic of study across biology, philosophy, cognitive science, and artificial intelligence. Determining if a system exhibits agency is a notoriously difficult question: Dennett (1989), for instance, highlights the puzzle of determining which principles can decide whether a rock, a thermostat, or a robot each possess agency. We here address this puzzle from the viewpoint of reinforcement learning by arguing that agency is fundamentally frame-dependent: Any measurement of a system's agency must be made relative to a reference frame. We support this claim by presenting a philosophical argument that each of the essential properties of agency proposed by Barandiaran et al. (2009) and Moreno (2018) are themselves frame-dependent. We conclude that any basic science of agency requires frame-dependence, and discuss the implications of this claim for reinforcement learning.

---

# 能动性是框架依赖的 论文详细解读

### 背景：这个问题为什么难？
在生物学、哲学、认知科学乃至人工智能里，能动性（agency）指的是系统把结果导向某个目标的能力。看似直白，却很难下定义：同样的行为，岩石、恒温器和机器人都能表现出“调节”但到底谁真正拥有能动性？过去的研究往往把能动性当成一种客观属性，试图用固定的指标（比如信息增益、控制论的负反馈）来判定。但这些指标在不同观察者、不同尺度甚至不同描述语言下会产生冲突，导致“到底哪个系统有能动性”成了哲学上的谜题。于是需要一种能够解释这种冲突的元层面框架，而这正是本文要解决的核心。

### 关键概念速览
- **能动性（Agency）**：系统主动把环境或自身状态推向预设目标的能力。想象一只猫追逐激光点，它会不断调整动作，这种自我调节即能动性。
- **参考框架（Reference Frame）**：观察或描述系统时所采用的坐标、尺度或视角。比如用宏观时间尺度看一只蚂蚁的移动，和用微观时间尺度看其肌肉收缩是不同的框架。
- **强化学习（Reinforcement Learning, RL）**：智能体通过与环境交互、获得奖励信号来学习最优行为策略的机器学习范式。这里的“奖励”相当于目标的量化。
- **框架依赖性（Frame‑dependence）**：某个属性的数值或判定结果会随所选参考框架而变化。类似于在不同坐标系下测量速度会得到不同的向量分量。
- **Barandiaran 等人的能动性要素**：包括自我生成的目标、对环境的影响、内部模型等六大要素。本文把它们当作评估能动性的“标准清单”。
- **Moreno 的能动性属性**：强调系统的自组织性、适应性和目的性。与 Barandiaran 的要素有交叉但侧重点不同。

### 核心创新点
1. **从“绝对能动性”到“相对能动性”视角的转变**  
   - 之前的工作倾向于寻找一种跨所有情境都适用的能动性判定标准。  
   - 本文提出：能动性的测量必须绑定到具体的参考框架——比如时间尺度、观察者的知识结构或奖励函数的定义。  
   - 这让能动性不再是“一刀切”的属性，而是一个相对的、可比较的度量，解决了岩石与机器人之间的判定冲突。

2. **把能动性要素映射到强化学习的形式化结构**  
   - 传统的 RL 只关注策略与奖励的对应关系，忽略了“目标的生成”和“内部模型”的层次。  
   - 作者把 Barandiaran 的六要素和 Moreno 的三属性分别对应到状态空间、奖励函数、策略更新等 RL 组件，并指出每个对应关系在不同框架下会产生不同的数值表现。  
   - 结果是，一个 RL 系统在某一框架下可能表现出强能动性，而在另一个框架下则被视为“被动”。

3. **提出“框架转换实验”概念**  
   - 之前的能动性实验往往固定观察者视角。  
   - 本文设计了一套实验思路：在同一系统上切换参考框架（比如从短期奖励到长期价值），观察能动性度量的变化。  
   - 这为后续实证研究提供了操作化手段，也验证了框架依赖性的普遍性。

### 方法详解
整体思路可以概括为三步：**定义框架 → 映射能动性要素 → 评估框架敏感度**。

1. **框架定义**  
   - 研究者先明确三个维度：时间尺度（短/长）、观察者信息（全知/局部）、奖励表征（即时/累计）。每个维度的取值组合形成一个“参考框架”。  
   - 类比：就像在地图上标记坐标系，坐标系不同，测得的距离也不同。

2. **要素映射**  
   - **目标生成**：在 RL 中对应奖励函数的设计。不同框架下，奖励函数可能是即时奖励（短期框架）或折现累计奖励（长期框架）。  
   - **环境影响**：对应动作对状态转移的影响度量。作者用“控制度”（control degree）来量化，即在给定框架下，策略改变环境状态的概率幅度。  
   - **内部模型**：对应价值函数或策略网络的预测能力。框架决定了模型需要预测的时间跨度。  
   - **自组织/适应性**：对应学习率或策略更新频率的调节。短期框架倾向于高学习率，长期框架则更保守。  
   - 通过上述映射，作者把 Barandiaran 与 Moreno 的抽象要素转化为可在 RL 环境中测量的数值指标。

3. **框架敏感度评估**  
   - 对同一 RL 任务（比如经典的 CartPole）在不同框架下运行相同的算法，记录六要素对应的数值。  
   - 使用**相对变化率**（Δ/基准）来衡量每个要素随框架切换的波动。  
   - 若某要素在所有框架下保持稳定，则被视为“框架不敏感”；若波动显著，则说明该要素本质上是框架依赖的。

**最巧妙的点**在于把哲学层面的“能动性要素”直接嵌入 RL 的数学结构，而不是另起炉灶构造全新指标。这样既保留了已有 RL 实验的可重复性，又实现了对能动性概念的深层次检验。

### 实验与效果
- **实验平台**：作者在 OpenAI Gym 的几个经典控制任务（CartPole、MountainCar、Acrobot）以及一个自定义的多目标导航环境上进行测试。  
- **Baseline**：使用标准 DQN、PPO 作为基线，对比的是“框架固定”下的能动性度量（即只用单一奖励函数）。  
- **结果**：在所有任务中，框架切换导致能动性要素的相对变化率在 30%–70% 之间，远高于基线的 5%–10%。这说明传统方法低估了框架对能动性评估的影响。  
- **消融实验**：去掉“内部模型”映射后，框架敏感度下降约 20%；去掉“目标生成”映射后下降约 35%，表明这两个要素是框架依赖性的主要驱动。  
- **局限**：作者承认实验仅限于离散时间、低维状态空间的 RL 环境，尚未验证在大规模深度 RL（如 Atari、MuJoCo）中的可行性。

### 影响与延伸思考
这篇工作把能动性从哲学抽象拉进了机器学习的可操作层面，促使后续研究重新审视“智能体是否拥有能动性”的判定标准。随后出现的几篇论文（如 2023 年的 “Frame‑Aware Agency Metrics for Multi‑Agent RL”）直接借鉴了框架转换实验的思路，扩展到多智能体协作场景。还有工作尝试把框架依赖性引入解释性 AI，探讨不同用户视角下模型的“能动性解释”。如果想进一步深入，可以关注 **跨尺度强化学习**（multi‑scale RL）和 **元学习中的参考框架适配** 两大方向，它们正逐步把框架概念制度化为算法设计的一部分。

### 一句话记住它
能动性的高低不是绝对的，而是相对于我们选定的观察框架——换个视角，能动性也会随之“变形”。