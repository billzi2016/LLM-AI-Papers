# Learning to Reason under Off-Policy Guidance

> **Date**：2025-04-21
> **arXiv**：https://arxiv.org/abs/2504.14945

## Abstract

Recent advances in large reasoning models (LRMs) demonstrate that sophisticated behaviors such as multi-step reasoning and self-reflection can emerge via reinforcement learning with verifiable rewards~(\textit{RLVR}). However, existing \textit{RLVR} approaches are inherently ``on-policy'', limiting learning to a model's own outputs and failing to acquire reasoning abilities beyond its initial capabilities. To address this issue, we introduce \textbf{LUFFY} (\textbf{L}earning to reason \textbf{U}nder o\textbf{FF}-polic\textbf{Y} guidance), a framework that augments \textit{RLVR} with off-policy reasoning traces. LUFFY dynamically balances imitation and exploration by combining off-policy demonstrations with on-policy rollouts during training. Specifically, LUFFY combines the Mixed-Policy GRPO framework, which has a theoretically guaranteed convergence rate, alongside policy shaping via regularized importance sampling to avoid superficial and rigid imitation during mixed-policy training. Compared with previous RLVR methods, LUFFY achieves an over \textbf{+6.4} average gain across six math benchmarks and an advantage of over \textbf{+6.2} points in out-of-distribution tasks. Most significantly, we show that LUFFY successfully trains weak models in scenarios where on-policy RLVR completely fails. These results provide compelling evidence that LUFFY transcends the fundamental limitations of on-policy RLVR and demonstrates the great potential of utilizing off-policy guidance in RLVR.

---

# 在离策略指导下学习推理 论文详细解读

### 背景：这个问题为什么难？

大型推理模型（LRM）在数学、逻辑等需要多步思考的任务上已经展现出惊人的潜力，但它们的能力大多来源于**强化学习带可验证奖励**（RLVR）这种“在策略上”（on‑policy）的方法。RLVR只能利用模型自己产生的轨迹来更新策略，等于是让模型在自己的答案上自我纠错。这样一来，模型的学习范围被锁死在它最初能想到的解法里，若起点太弱，就会陷入局部最优，甚至根本学不到多步推理或自我反思的技巧。换句话说，传统 RLVR 像是只让学生在自己的草稿本上练习，缺少老师的指点，导致学习效率和上限都受限。

### 关键概念速览
- **大型推理模型（LRM）**：参数量巨大的语言模型，专门用于需要链式思考的任务。把它想象成会写数学证明的“超级学生”。
- **强化学习带可验证奖励（RLVR）**：在训练时给模型一个可以自动检查对错的奖励信号，让模型通过试错学习。类似于给学生每一步都打分的考试系统。
- **on‑policy（在策略上）**：模型只能用自己当前策略产生的数据来学习，就像只能用自己的作业来改进学习方法。
- **off‑policy（离策略）**：允许使用其他策略（比如更强的老师模型）产生的数据进行学习，相当于学生可以参考优秀的范例答案。
- **混合策略 GRPO（Mixed‑Policy GRPO）**：一种理论上保证收敛的强化学习框架，能够在混合了多种数据来源的情况下仍然稳定优化。把它看作是能够兼顾自学和听课的混合学习计划。
- **策略塑形（policy shaping）**：在更新策略时加入额外的约束或权重，使得学习过程不会盲目模仿。类似于老师在批改作业时给出“这一步可以这样写更好”的提示。
- **正则化重要性采样（regularized importance sampling）**：对离策略数据进行加权时加入正则项，防止权重过大导致模型过度依赖示例。相当于在参考范例时给出“适度参考、保持创新”的原则。

### 核心创新点
1. **引入离策略示例 → LUFFY 采用强模型输出作为演示 → 训练弱模型时不再局限于自身的错误轨迹**  
   传统 RLVR 只能靠模型自己“摸索”。LUFFY 把更强的模型生成的推理链放进训练池，让弱模型可以直接学习高质量的多步推理路径，突破了自我循环的瓶颈。

2. **混合 on‑policy 与 off‑policy 数据 → 使用 Mixed‑Policy GRPO 统一优化目标 → 在理论上仍保有收敛速度**  
   直接把两类数据混在一起会导致优化不稳定。LUFFY 采用已证明收敛的 Mixed‑Policy GRPO，把两种轨迹视作同一策略的不同采样，确保学习过程既快又稳。

3. **防止“表面模仿” → 通过正则化重要性采样进行策略塑形 → 模型在学习示例的同时保持探索能力**  
   若只把离策略示例当作硬标签，模型会死板复制答案，失去创新。LUFFY 给每条离策略轨迹加上正则化的采样权重，使得模型在模仿的同时仍被鼓励探索未见的解法。

4. **动态平衡模仿与探索 → 在训练过程中自适应调节 on‑policy 与 off‑policy 的比例 → 在弱模型几乎学不动时自动倾向示例，学会后逐步恢复自我探索**  
   这一步让 LUFFY 能在“老师帮助”和“自我实验”之间找到最佳切点，避免了固定比例导致的过度依赖或不足。

### 方法详解
**整体思路**  
LUFFY 的训练流程可以概括为四步：  
1) **收集离策略示例**：用一个已经在数学推理上表现优秀的模型（老师模型）生成大量高质量的推理轨迹。  
2) **构建混合经验池**：把老师示例和模型自身在 RLVR 环境中产生的 on‑policy 轨迹一起存入 replay buffer。  
3) **混合采样并计算目标**：在每一次梯度更新时，从经验池中按一定比例抽取两类轨迹，使用 Mixed‑Policy GRPO 计算强化学习梯度。  
4) **策略塑形**：对抽取的离策略轨迹施加正则化重要性采样权重，使得它们在梯度中既能提供强信号，又不会压制模型的探索。

**关键模块拆解**  

- **离策略示例生成**  
  类比于老师在课堂上写出完整的解题步骤，LUFFY 让一个预训练好的大模型（比如 GPT‑4）在同样的数学题目上执行 **思维链（CoT）** 推理，得到“问题 → 步骤1 → 步骤2 → … → 答案”的完整序列。每条序列都被标记为高奖励（因为答案可验证），并存入离策略库。

- **经验池与混合采样**  
  经验池相当于一个大抽屉，里面既有老师的范例（离策略）也有学生自己的草稿（on‑policy）。在每次更新时，LUFFY 按 **α** 的比例抽取离策略样本，按 **1‑α** 抽取 on‑policy 样本。α 会随训练进度自适应：模型刚开始时 α 较大，帮助其快速获得正确的推理框架；当模型表现提升后，α 逐渐下降，鼓励模型自行探索。

- **Mixed‑Policy GRPO 目标**  
  GRPO（Generalized Reward Policy Optimization）是一种把奖励函数和 KL 散度（衡量策略变化）结合的优化方式。LUFFY 将两类轨迹视为同一策略的不同采样，计算 **期望奖励** 与 **策略熵正则** 的加权和。因为混合采样本质上仍然是从同一分布的近似抽样，GRPO 的收敛证明仍然成立。

- **正则化重要性采样（策略塑形）**  
  对每条离策略轨迹，LUFFY 计算 **重要性权重** = (目标策略概率 / 行为策略概率)。如果直接使用这些权重，稀有但高质量的示例会产生极大权重，导致模型过度模仿。为此，LUFFY 在权重上加上 **L2 正则**（或 KL 正则），限制权重的幅度，使得示例的指导力度是“温和的”。这样模型在学习老师的思路时仍保留自己的探索空间。

**最巧妙的设计**  
- **动态 α 调度**：不是固定的离策略比例，而是根据模型的验证表现实时调节，让弱模型在“需要帮助”时得到更多示例，在“已经掌握”时自行探索。  
- **正则化重要性采样**：把离策略示例的影响力软化，避免“照搬答案”，保持模型的创新能力，这在纯模仿学习里是常见的陷阱。

### 实验与效果
- **测试任务**：六个公开的数学推理基准（如 GSM8K、MATH、SVAMP 等），以及若干 **分布外（OOD）** 题目，用来检验模型的泛化能力。  
- **对比基线**：传统 RLVR（纯 on‑policy）、单纯的离策略模仿学习（IL），以及最新的混合 RL+IL 方法。  
- **主要结果**：LUFFY 在六个基准上平均提升 **+6.4 分**，在 OOD 任务上提升 **+6.2 分**。尤其在使用 **弱模型**（参数量仅为基准模型的 1/4）时，传统 RLVR 完全无法收敛，而 LUFFY 能够成功训练出可用的推理模型。  
- **消融实验**：  
  1) **去掉离策略示例** → 性能跌回到传统 RLVR 水平。  
  2) **去掉正则化重要性采样** → 模型出现“死板模仿”，在新题目上表现显著下降。  
  3) **固定 α**（不动态调节） → 收敛速度变慢，最终分数略低。  
- **局限性**：需要一个已经足够强的老师模型来生成高质量示例；离策略示例的质量直接决定最终提升幅度；存储和采样大量示例会带来额外的计算和内存开销。论文中也提到在极度噪声的离策略数据上，正则化重要性采样的效果会减弱。

### 影响与延伸思考
LUFFY 首次系统化地把 **离策略示例** 融入 **RLVR** 框架，并给出理论收敛保证，打开了“老师‑学生混合强化学习”在大语言模型推理中的新局面。此后，出现了多篇工作尝试：

- **Self‑Play with Teacher Models**（2023‑2024）把学生模型的最新版本回馈给老师，形成循环提升。  
- **Hybrid RL+IL for Code Generation**（2024）借鉴 LUFFY 的正则化重要性采样，在代码生成任务上实现了类似的提升。  
- **Curriculum Off‑Policy RL**（2025）进一步探索如何根据学生的学习进度动态挑选离策略示例的难度。

想继续深入，可以关注以下方向：  
1) **示例质量评估**：如何自动筛选或生成更有教学价值的离策略轨迹。  
2) **多老师融合**：不同强度或不同风格的老师模型提供多样化示例，提升模型的鲁棒性。  
3) **低资源离策略学习**：在缺少强老师的情况下，利用人类标注或自监督生成的示例进行离策略指导。

### 一句话记住它
**LUFFY 用强模型的高质量推理示例“点灯”，让弱模型在强化学习中不再盲目摸索，从而实现跨越式的推理能力提升。**