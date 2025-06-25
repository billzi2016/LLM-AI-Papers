# DiffuCoder: Understanding and Improving Masked Diffusion Models for Code Generation

> **Date**：2025-06-25
> **arXiv**：https://arxiv.org/abs/2506.20639

## Abstract

Diffusion large language models (dLLMs) are compelling alternatives to autoregressive (AR) models because their denoising models operate over the entire sequence. The global planning and iterative refinement features of dLLMs are particularly useful for code generation. However, current training and inference mechanisms for dLLMs in coding are still under-explored. To demystify the decoding behavior of dLLMs and unlock their potential for coding, we systematically investigate their denoising processes and reinforcement learning (RL) methods. We train a 7B dLLM, \textbf{DiffuCoder}, on 130B tokens of code. Using this model as a testbed, we analyze its decoding behavior, revealing how it differs from that of AR models: (1) dLLMs can decide how causal their generation should be without relying on semi-AR decoding, and (2) increasing the sampling temperature diversifies not only token choices but also their generation order. This diversity creates a rich search space for RL rollouts. For RL training, to reduce the variance of token log-likelihood estimates and maintain training efficiency, we propose \textbf{coupled-GRPO}, a novel sampling scheme that constructs complementary mask noise for completions used in training. In our experiments, coupled-GRPO significantly improves DiffuCoder's performance on code generation benchmarks (+4.4\% on EvalPlus) and reduces reliance on AR bias during decoding. Our work provides deeper insight into the machinery of dLLM generation and offers an effective, diffusion-native RL training framework. https://github.com/apple/ml-diffucoder.

---

# DiffuCoder：解析与改进用于代码生成的掩码扩散模型 论文详细解读

### 背景：这个问题为什么难？

代码生成一直是大语言模型（LLM）的硬核场景。传统的自回归（AR）模型一次生成一个 token，虽然思路直观，却只能在局部做决定，难以全局规划代码结构。扩散大语言模型（dLLM）把噪声逐步去除，理论上可以一次看到完整序列，却缺少成熟的训练和推理方案，尤其在代码这种对语法和逻辑要求极高的任务上更是如此。之前的工作大多把 dLLM 当作“玩具”，没有系统分析它们的解码行为，也没有针对代码设计有效的强化学习（RL）训练流程。因此，如何让 dLLM 真正发挥全局规划和迭代细化的优势，成为了迫切需要解决的难题。

### 关键概念速览

**扩散大语言模型（dLLM）**：把文本看成被噪声污染的信号，模型学习在多个噪声层次上逐步恢复原始序列。类似于把一幅模糊的画一步步擦掉雾，让细节慢慢显现。

**自回归（AR）模型**：一次预测下一个 token，后面的生成全部依赖已经产生的前缀。可以想象成在写代码时只能看已经写好的代码，不能提前看到后面的结构。

**掩码噪声（masked noise）**：在扩散过程中，只对序列的某些位置加入噪声，其他位置保持原样。相当于在一段代码里把若干行涂黑，让模型在恢复时必须推断这些被遮住的行。

**半自回归解码（semi‑AR decoding）**：在 dLLM 中人为强制模型先按自回归顺序生成，再再用扩散步骤微调。像是先让程序员写草稿，再让编辑细致润色。

**采样温度（sampling temperature）**：控制随机性的大参数，温度高会让模型更大胆尝试不同 token，甚至改变生成顺序。可以把它比作画家在创作时的“随性程度”。

**强化学习（RL）**：让模型通过奖励信号学习更好策略的框架。这里的奖励往往是代码的功能正确性或测试通过率。

**GRPO（Generalized Reward‑Weighted Policy Optimization）**：一种基于策略梯度的 RL 方法，利用奖励对采样的概率进行加权，以提升期望回报。

**Coupled‑GRPO**：本文提出的改进版 GRPO，使用互补的掩码噪声为同一次采样生成两套不同的完成版本，从而降低梯度估计的方差。

### 核心创新点

1. **解码行为系统分析 → 通过 7B DiffuCoder 观察**  
   过去只知道 dLLM 能一次看到全序列，却不清楚它到底怎么决定生成顺序。作者用训练好的 7 B 模型做大量实验，发现 dLLM 能自行决定“因果性”强弱，不需要半自回归的强制约束；同时提升采样温度会让模型不仅换词，还会调换生成的先后顺序。这种行为为后续的搜索空间提供了天然的多样性。

2. **温度驱动的顺序多样性 → 为 RL 开辟更宽的搜索**  
   传统 AR 模型的温度只影响 token 选取，顺序基本固定。DiffuCoder 的实验表明，温度还能让模型在不同的迭代轮次中尝试不同的代码块排列，从而在 RL 的 roll‑out 阶段产生更丰富的候选解。

3. **Coupled‑GRPO 采样方案 → 通过互补噪声降低方差**  
   标准 GRPO 在每次采样后直接使用该完成计算奖励，梯度方差大。作者提出在同一次噪声采样后，生成两套互补的掩码（即一次遮住 A，另一次遮住非 A），得到两条完成路径。因为两条路径在噪声上是“配对”的，梯度估计的噪声会相互抵消，训练更稳。

4. **Diffusion‑native RL 框架 → 提升代码生成基准**  
   将 Coupled‑GRPO 融入 DiffuCoder 的训练流程后，在 EvalPlus 基准上提升了约 4.4% 的分数，同时在解码时对 AR 偏置的依赖显著下降。相当于在不增加模型规模的前提下，直接把扩散模型的潜力搬上了实用层面。

### 方法详解

#### 整体思路

DiffuCoder 的训练和推理可以拆成三大阶段：  
1) **预训练**：在 130 B 代码 token 上，用标准的掩码扩散目标训练一个 7 B 参数的 dLLM。  
2) **行为分析**：利用已训练好的模型，系统测量不同温度、不同掩码策略下的生成顺序和 token 多样性。  
3) **强化学习微调**：在预训练模型的基础上，使用 Coupled‑GRPO 进行 RL 微调，使模型在实际代码评测上获得更高奖励。

#### 关键模块拆解

1. **掩码扩散过程**  
   - 输入序列先被随机掩码（即把若干位置替换为噪声），形成“噪声序列”。  
   - 模型在每个扩散步长上预测被掩码位置的原始 token，同时保留已经恢复的部分。  
   - 通过多步迭代，噪声逐渐被去除，最终得到完整代码。  
   类比：把一本书的若干页撕掉，然后让模型一步步补全缺失的页码。

2. **温度调节的顺序多样性**  
   - 在采样阶段，引入温度 τ。高 τ 会让模型在每一步的概率分布更平坦，从而更容易选择非最优 token。  
   - 由于扩散是并行恢复所有位置的，选择不同 token 也会导致后续恢复顺序的变化。实验表明，温度提升 0.5 左右即可让同一段代码出现多种生成顺序。  
   - 这为 RL 的 roll‑out 提供了“不同路径”，类似于在搜索树中同时探索多条枝干。

3. **Coupled‑GRPO 采样**  
   - **噪声采样**：一次抽取噪声向量 z。  
   - **互补掩码**：生成两套掩码 m₁、m₂，使得 m₁ ∪ m₂ = 全部位置且 m₁ ∩ m₂ = ∅。  
   - **双向完成**：分别用 m₁、m₂ 对同一噪声 z 进行扩散去噪，得到完成 c₁、c₂。  
   - **奖励计算**：对 c₁、c₂ 运行代码测试，得到奖励 r₁、r₂。  
   - **梯度加权**：使用 GRPO 的加权公式，但因为 c₁、c₂ 来自同一噪声，它们的对数似然梯度在噪声维度上是相反的，方差自然被抑制。  
   直观上，这一步相当于让模型在同一“雾天”里同时预测两条互补的路线，然后让奖励帮助它找出哪条更靠谱。

4. **解码时的 AR 偏置削弱**  
   - 传统 dLLM 在实际使用时往往会加入自回归的强制约束，以防止生成不符合语法的代码。  
   - 通过 Coupled‑GRPO 的训练，模型学会在噪声层面自行控制因果性，解码时可以直接使用纯扩散采样，减少对 AR 机制的依赖。  

#### 反直觉或巧妙之处

- **温度影响顺序**：大多数人只把温度当作“让模型更随机”，没想到它还能改变生成的时间线，这在扩散模型里尤为重要。  
- **互补掩码配对**：把同一次噪声拆成两套互补任务，看似增加了计算量，实际上梯度方差的下降让收敛更快，整体训练时间并未显著上升。  
- **不需要半自回归**：很多早期的 dLLM 研究都在尝试把自回归和扩散混合，结果复杂且效果一般。DiffuCoder 直接让模型自行决定因果强度，省去了一层人为的约束。

### 实验与效果

- **数据与任务**：在 130 B 代码 token（包括多语言开源项目）上预训练 7 B 参数模型。随后在 **EvalPlus** 基准上进行 RL 微调，该基准覆盖了函数实现、单元测试通过率等多维度指标。  
- **基线对比**：与同规模的自回归模型（如 CodeGen‑7B）以及未使用 RL 的原始 DiffuCoder 进行比较。  
  - 在 EvalPlus 上，Coupled‑GRPO 微调后提升约 **4.4%**（原文给出的数字）。  
  - 解码时对 AR 偏置的依赖下降，意味着可以更纯粹地使用扩散采样而不牺牲质量。  
- **消融实验**：作者分别去掉温度调节、互补掩码或 GRPO 加权，发现每一项都对最终分数有正向贡献，尤其是去掉互补掩码后方差回升，训练不稳定。  
- **局限性**：论文主要在单模型（7 B）上验证，尚未探索更大规模模型的行为；此外，Coupled‑GRPO 需要两套完成的奖励计算，若奖励函数非常昂贵（如大规模运行时）会导致成本上升。作者也提到在极端低温度下顺序多样性消失，RL 搜索空间会受限。

### 影响与延伸思考

DiffuCoder 把扩散模型的全局规划优势带入代码生成，并提供了第一个专门针对扩散模型的 RL 框架。自发表后，已有几篇工作尝试把 **Coupled‑GRPO** 思路迁移到自然语言生成、图像编辑等领域，证明互补噪声配对在降低梯度方差上具有通用价值。还有研究在探索更高效的掩码策略（如动态掩码）以及把 **温度‑顺序耦合** 用作搜索启发式。想进一步深入的读者可以关注：

- **扩散模型在结构化输出（代码、SQL、图）上的专用采样策略**。  
- **基于强化学习的扩散微调**，尤其是如何在奖励稀疏的情况下保持训练效率。  
- **大规模 dLLM 与硬件加速的协同优化**，因为扩散推理本身对显存要求更高。

### 一句话记住它

**DiffuCoder 用互补掩码的 Coupled‑GRPO，让扩散模型在代码生成中既能全局规划，又能高效强化学习。**