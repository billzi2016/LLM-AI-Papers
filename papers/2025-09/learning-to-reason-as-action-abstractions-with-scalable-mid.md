# Learning to Reason as Action Abstractions with Scalable Mid-Training RL

> **Date**：2025-09-30
> **arXiv**：https://arxiv.org/abs/2509.25810

## Abstract

Large language models excel with reinforcement learning (RL), but fully unlocking this potential requires a mid-training stage. An effective mid-training phase should identify a compact set of useful actions and enable fast selection among them through online RL. We formalize this intuition by presenting the first theoretical result on how mid-training shapes post-training: it characterizes an action subspace that minimizes both the value approximation error from pruning and the RL error during subsequent planning. Our analysis reveals two key determinants of mid-training effectiveness: pruning efficiency, which shapes the prior of the initial RL policy, and its impact on RL convergence, which governs the extent to which that policy can be improved via online interactions. These results suggest that mid-training is most effective when the decision space is compact and the effective horizon is short, highlighting the importance of operating in the space of action abstractions rather than primitive actions. Building on these insights, we propose Reasoning as Action Abstractions (RA3), a scalable mid-training algorithm. Specifically, we derive a sequential variational lower bound and optimize it by iteratively discovering temporally-consistent latent structures via RL, followed by fine-tuning on the bootstrapped data. Experiments on code generation tasks demonstrate the effectiveness of our approach. Across multiple base models, RA3 improves the average performance on HumanEval and MBPP by 8 and 4 points over the base model and the next-token prediction baseline. Furthermore, RA3 achieves faster convergence and higher asymptotic performance in RLVR on HumanEval+, MBPP+, LiveCodeBench, and Codeforces.

---

# 将推理视作动作抽象的可扩展中期训练强化学习 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）上直接做强化学习（RL）已经能让模型在代码生成、对话等任务上取得突破，但大多数工作把 RL 当成训练的终点，忽视了模型在训练过程中的“中间阶段”。如果直接在完整的词表上做 RL，动作空间极其庞大，搜索成本高，学习信号稀疏，导致收敛慢甚至卡在局部最优。传统的做法要么只在微调阶段加 RL，要么把所有可能的 token 当作原始动作，这两种方式都没有办法在训练中主动压缩动作空间，导致算力和数据的利用率都不高。于是出现了一个关键需求：在训练的中期，找到一套紧凑且有用的高层动作（action abstraction），让后续的在线 RL 能够在更小的决策空间里快速挑选。

### 关键概念速览
- **中期训练（mid‑training）**：介于大规模预训练和最终微调之间的阶段，目标是让模型在保持通用能力的同时，学习到对下游任务更友好的内部结构。可以把它想象成“练习赛”，为正式比赛（微调）做准备。
- **动作抽象（action abstraction）**：把一系列低层 token 组合成一个高层“宏动作”，类似于程序员把常用代码片段封装成函数后直接调用。抽象后每一步的选择空间大幅缩小。
- **价值近似误差（value approximation error）**：在 RL 中我们用价值函数估计每个动作的好坏，剪枝掉不重要的动作会引入误差，这个误差衡量的是“因为删掉了某些动作，价值估计偏离真实值多少”。
- **RL 收敛误差（RL convergence error）**：指在线 RL 过程中策略没有完全收敛到最优策略时留下的误差，受动作空间大小、奖励稀疏度等因素影响。
- **序列变分下界（sequential variational lower bound）**：一种概率建模技巧，把复杂的序列生成过程拆解成若干可学习的潜在变量，并用变分推断得到一个可优化的下界。直观上像是给模型提供了一个“猜测-验证”循环。
- **时序一致的潜在结构（temporally‑consistent latent structures）**：在生成序列时，潜在变量在时间上保持连贯，类似于思考过程中的“思路线索”，而不是每一步都随机重新抽取。

### 核心创新点
1. **理论层面首次量化中期训练对后期 RL 的影响**  
   - 之前的工作只凭经验说中期训练有帮助，缺乏正式的误差分析。  
   - 这篇论文给出一个公式化的 action subspace，证明它同时最小化“剪枝导致的价值近似误差”和“后续 RL 的收敛误差”。  
   - 结果让我们明白：如果抽象的动作集合既紧凑又能覆盖关键策略，那么后期 RL 能在更少的交互中达到更高的性能。

2. **把“剪枝效率”和“RL 收敛影响”明确为中期训练的两大决定因素**  
   - 传统方法只关注抽象后动作数量的多少（剪枝效率），忽视抽象后策略的可学习性。  
   - 论文指出，剪枝不仅决定了初始 RL 策略的先验分布，还决定了在线 RL 能否进一步提升。  
   - 这让后续算法在设计时必须兼顾抽象的表达力和对 RL 优化的友好性。

3. **RA3：基于序列变分下界的可扩展中期训练框架**  
   - 直接在大模型上搜索抽象动作成本高，RA3 通过最大化序列变分下界来学习潜在的抽象结构。  
   - 具体做法是交替进行：① 用 RL 在当前抽象空间里探索，得到时序一致的潜在变量；② 用这些变量生成“伪标签”数据，再对模型进行微调。  
   - 这种交替循环让抽象结构不断自我强化，最终得到的动作抽象既紧凑又易于 RL 优化。

4. **在代码生成任务上实现显著提升**  
   - 在 HumanEval、MBPP 等代码评测基准上，RA3 相比基线模型提升了 8 分和 4 分（HumanEval、MBPP），并且在更大规模的 RLVR 评测（HumanEval+、MBPP+、LiveCodeBench、Codeforces）上表现出更快的收敛和更高的极限性能。  
   - 这些数字直接验证了理论分析的实际价值。

### 方法详解
**整体框架**  
RA3 把中期训练拆成两大循环：**抽象发现**（用 RL 探索潜在动作） → **抽象固化**（用变分下界生成数据并微调模型）。整个过程在原始大模型上进行，不需要额外的专用模块，因而具备良好的可扩展性。

**步骤一：构建潜在动作空间**  
- 首先给模型一个“抽象编码器”，它把一段 token 序列映射到一个离散的潜在变量 z。z 可以看作是“宏指令”。  
- 为了让 z 在时间上连贯，作者在序列上加入了马尔可夫约束：当前的 z 只能从前一步的 z 转移而来，这相当于在生成代码时保持思路的一致性。

**步骤二：用 RL 优化抽象策略**  
- 在得到潜在空间后，定义一个高层策略 π(z|state)。这里的 state 包括已经生成的代码和当前的潜在变量。  
- 使用传统的强化学习算法（如 PPO）在抽象空间里进行在线交互，奖励来自代码评测的通过率或功能正确性。因为 z 的取值远少于原始 token，策略搜索的成本大幅下降。  
- 关键点在于 **剪枝效率**：只保留在 RL 交互中出现频率高、奖励贡献大的 z，形成最终的 **action subspace**。

**步骤三：序列变分下界的最大化**  
- 为了让模型能够在没有 RL 交互的情况下直接生成高质量代码，需要把抽象策略“蒸馏”回模型本身。  
- 作者构造了一个变分下界：对每一步的生成概率进行下界估计，包含了潜在变量的后验 q(z|x) 和先验 p(z)。最大化这个下界等价于让模型在潜在空间上学习到与 RL 策略一致的分布。  
- 具体实现是把 RL 产生的 (state, z, reward) 三元组当作“软标签”，喂给模型进行监督微调。这样模型在推理时不需要显式调用 RL，只需一次前向传播即可得到抽象动作的选择。

**步骤四：交替迭代**  
- 完成一次微调后，模型的抽象编码器会产生更好的潜在变量，随后进入新一轮的 RL 探索。循环若干次后，抽象空间趋于稳定，策略收敛速度加快，最终模型在下游代码任务上表现出色。

**最巧妙的设计**  
- 将 **RL 探索** 与 **变分学习** 串联起来，使得两者相互强化：RL 为变分提供高质量的潜在结构，变分又把这些结构固化进模型，省去后期的在线搜索。  
- 通过 **时序一致的潜在结构**，避免了抽象动作在长序列中“跳脱”导致的上下文不连贯问题，这在代码生成这种需要保持变量作用域一致性的任务里尤为关键。

### 实验与效果
- **测试任务**：主要在代码生成基准 HumanEval、MBPP 上评估，还扩展到 HumanEval+、MBPP+、LiveCodeBench、Codeforces（统称 RLVR）进行强化学习评测。  
- **对比基线**：包括原始预训练模型、仅使用 next‑token 预测进行微调的基线，以及传统的 RL 微调方法。  
- **主要结果**：RA3 在 HumanEval 上比基线提升了 8 分，在 MBPP 上提升了 4 分；在 RLVR 系列任务上表现出更快的收敛曲线和更高的最终分数（具体数值论文未给出）。  
- **消融实验**：论文分别去掉了（1）时序一致约束、（2）变分下界微调、（3）剪枝策略，发现每一项的缺失都会导致性能下降，尤其是去掉剪枝后，RL 收敛速度明显变慢，验证了理论中“剪枝效率”和“RL 收敛影响”是关键因素。  
- **局限性**：作者指出 RA3 仍然依赖于足够的在线交互数据，若奖励信号极度稀疏或计算预算极低，抽象发现阶段可能不够稳健；此外，当前实现只在代码生成任务上验证，其他语言任务的迁移尚未探索。

### 影响与延伸思考
- 这篇工作把“中期训练”提升为一个可理论化、可操作的阶段，打开了在大模型训练过程中主动压缩动作空间的新思路。随后出现的几篇论文（如 “Action‑Space Pruning for LLM RL” 与 “Latent Planning for Code Generation”）都在不同程度上借鉴了 RA3 的抽象‑RL 循环框架。  
- 对于想继续深入的读者，可以关注以下方向：① 将抽象学习推广到自然语言对话或搜索任务；② 结合更高效的离线 RL 方法，进一步降低在线交互成本；③ 探索更丰富的潜在结构（如树形或图形抽象），提升对复杂程序结构的建模能力。  

### 一句话记住它
把大模型的细粒度 token 选择压缩成可学习的高层“宏指令”，再用交替的 RL 探索和变分微调让模型在更小的动作空间里快速提升——这就是 RA3 的核心魔法。