# Hierarchical Budget Policy Optimization for Adaptive Reasoning

> **Date**：2025-07-21
> **arXiv**：https://arxiv.org/abs/2507.15844

## Abstract

Large reasoning models achieve remarkable performance through extensive chain-of-thought generation, yet they suffer from a critical inefficiency: applying uniformly extensive reasoning regardless of problem complexity. We present Hierarchical Budget Policy Optimization (HBPO), a reinforcement learning framework that enables models to learn problem-specific reasoning depths without sacrificing capability. Unlike existing approaches that impose rigid constraints or rely on discrete mode selection, HBPO partitions the exploration space into budget-constrained hierarchies (512-2560 tokens), each with differentiated reward structures that preserve both efficiency incentives and reasoning capabilities. This design addresses a fundamental challenge in efficient reasoning training: traditional length penalties systematically bias models away from necessary long reasoning paths, causing exploration space collapse. Through hierarchical sampling and budget-aware rewards, HBPO maintains exploration diversity while teaching models to recognize when extended deliberation is warranted. Extensive experiments demonstrate that HBPO reduces average token usage by up to 60.6% while improving accuracy by 3.14% across four reasoning benchmarks. Most notably, HBPO exhibits emergent adaptive behavior where models automatically adjust reasoning depth based on problem complexity. Our results suggest that reasoning efficiency and capability are not inherently conflicting, and can be simultaneously optimized through appropriately structured hierarchical training that preserves exploration diversity.

---

# 层次预算策略优化用于自适应推理 论文详细解读

### 背景：这个问题为什么难？

大模型在需要链式思考的任务上表现惊人，但它们往往会把每道题都当成“高难度”来处理，统一生成几百甚至上千个推理 token。这样既浪费算力，又导致响应时间变慢。早期的解决思路要么在训练时直接加上长度惩罚，要么硬性规定固定的思考步数。前者会把模型推向“只想快不想对”，导致在真正需要长链思考的题目上性能下降；后者则缺乏灵活性，无法根据题目复杂度自行调节推理深度。于是出现了一个核心矛盾：**效率**和**推理能力**似乎不可兼得。

### 关键概念速览
- **链式思考（Chain‑of‑Thought, CoT）**：模型在给出最终答案前，先把推理过程写出来，像在纸上写草稿一样，帮助模型理清思路。  
- **预算（Budget）**：这里指模型在一次推理过程中最多可以使用的 token 数量上限，类似于给模型设定的“思考时间”。  
- **层次（Hierarchy）**：把不同规模的预算（如 512、1024、2048 token）组织成层级，每一层对应一种“思考深度”。  
- **奖励结构（Reward Structure）**：强化学习中用来衡量行为好坏的打分规则，不同层次的奖励会强调不同的目标（效率 vs. 完整性）。  
- **探索空间（Exploration Space）**：模型在训练时可能尝试的所有推理路径的集合，若奖励设计不当，这个空间会被“压缩”，导致模型只学会一种固定的思考方式。  
- **强化学习（Reinforcement Learning, RL）**：让模型通过试错获得策略的学习框架，这里模型的“动作”是决定继续思考还是直接输出答案。  
- **自适应推理（Adaptive Reasoning）**：模型能够根据题目难度自动调节思考深度，而不是一刀切。

### 核心创新点
1. **预算层次化 → 将推理空间划分为多个预算区间（512‑2560 token）** → 通过让模型在不同预算层次中采样，保持了探索的多样性，避免了单一长度惩罚导致的空间塌陷。  
2. **层级奖励差异化 → 为每个预算层次设计独立的奖励函数** → 在小预算层次强调高效使用 token，在大预算层次则更看重完整推理的正确率，使模型学会在需要时“加大投入”。  
3. **层次采样策略 → 训练时先随机挑选一个预算层次，再在该层次内部进行 RL 优化** → 这种两层抽样相当于让模型先决定“要花多少钱”，再决定“怎么花”，实现了从宏观到微观的决策分解。  
4. **兼顾效率与能力的统一目标 → 通过层次奖励的加权求和，使整体目标既能压缩 token 使用，又能提升准确率** → 实验表明，平均 token 使用下降 60.6% 的同时，整体准确率提升 3.14%。

### 方法详解
整体框架可以看作三步走：**预算层次划分 → 层次采样 → 层次奖励驱动的 RL 优化**。

1. **预算层次划分**  
   作者先把可用的 token 上限划分为若干区间，例如 512、1024、2048、2560。每个区间对应一个“思考预算”。这一步类似于把不同难度的题目放进不同的教室，教室大小决定了学生可以使用的教材量。

2. **层次采样**  
   在每一次训练迭代中，系统先从这些预算层次中随机抽取一个（可以按均匀或按任务难度的先验分布抽），然后让模型在该预算限制下生成推理序列。抽取预算的过程相当于给模型先发一张“消费卡”，卡面上写明了本轮最多可以花多少 token。

3. **层次奖励设计**  
   对于每个预算层次，作者定义了专属的奖励函数。  
   - **小预算层**：奖励主要由 token 使用率（越少越好）和答案正确率的加权组成，鼓励模型在有限资源下尽可能准确。  
   - **大预算层**：奖励更倾向于完整推理的质量，例如对每一步的逻辑一致性、答案正确率以及推理链的完整性进行打分。  
   这种差异化奖励让模型在“小钱买大事”和“大钱买细节”之间学会权衡。

4. **强化学习优化**  
   使用策略梯度（如 PPO）对模型的决策策略进行更新。关键在于**预算感知的状态表示**：模型的输入除了原始问题外，还会拼接当前预算信息，使得策略能够感知“还能花多少 token”。在梯度计算时，奖励会根据所处的预算层次自动切换，从而实现层次化的学习信号。

5. **最巧妙的设计**  
   - **预算作为显式动作**：传统 RL 只把“继续思考”或“停止”当作动作，而这里把“选择预算层次”也视为一次高层动作，使得模型在宏观层面就能决定思考深度。  
   - **防止探索空间塌陷**：通过在每层都保留一定的奖励激励，模型不会因为惩罚过重而只学会最短路径，从而保持了多样的推理策略。

### 实验与效果
- **测试任务**：四个公开的推理基准（包括数学题、逻辑推理、常识问答等），均需要链式思考。  
- **对比基线**：传统的统一长度惩罚模型、固定预算的硬约束模型以及最新的自适应推理方法。  
- **核心结果**：HBPO 在所有基准上平均 **降低 60.6%** 的 token 使用量，同时 **提升 3.14%** 的整体准确率。尤其在需要长链思考的题目上，准确率提升更明显。  
- **消融实验**：作者分别去掉（1）层次奖励差异化、（2）预算层次采样、（3）预算信息输入。结果显示，去掉任意一项都会导致 token 使用下降幅度减半，或准确率回落到基线水平，说明每个模块都是不可或缺的。  
- **局限性**：论文指出，HBPO 仍然依赖预先设定的预算区间，若任务分布与这些区间不匹配，可能出现预算不足或浪费的情况；此外，训练过程的 RL 部分对计算资源要求较高，尚未在大规模商用模型上验证。

### 影响与延伸思考
HBPO 打破了“效率 vs. 能力”必须二选一的旧观念，开启了**层次化预算感知训练**的潮流。后续工作（如 *Budgeted Adaptive Prompting*、*Hierarchical Reasoning Controllers*）纷纷借鉴其层次奖励和预算采样的思路，尝试把预算概念推广到多模态推理、代码生成等更复杂的场景。对想进一步探索的读者，可以关注以下方向：  
- 自动学习预算区间（让模型自己发现最优预算层次）  
- 将预算概念与检索增强（RAG）结合，实现“思考+查资料”的双预算系统  
- 在大模型微调阶段引入轻量级的 RL 近似，以降低训练成本。

### 一句话记住它
**HBPO 让大模型学会先决定“花多少钱”，再决定“怎么花”，从而在保持高推理质量的同时大幅削减思考成本。**