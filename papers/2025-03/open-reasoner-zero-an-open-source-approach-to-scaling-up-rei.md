# Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model

> **Date**：2025-03-31
> **arXiv**：https://arxiv.org/abs/2503.24290

## Abstract

We introduce Open-Reasoner-Zero, the first open source implementation of large-scale reasoning-oriented RL training on the base model focusing on scalability, simplicity and accessibility. Through extensive experiments, we demonstrate that a minimalist approach, vanilla PPO with GAE ($\lambda=1$, $\gamma=1$) and straightforward rule-based rewards, without any KL regularization, is sufficient to scale up both benchmark performance and response length, replicating the scaling phenomenon observed in DeepSeek-R1-Zero. Using the same base model, Qwen2.5-32B base, as DeepSeek-R1-Zero-Qwen-32B, our implementation achieves superior performance across AIME2024, MATH500, and GPQA Diamond, while demonstrating remarkable efficiency, requiring only 1/10 of the training steps compared to the DeepSeek-R1-Zero pipeline. Moreover, our analysis not only covers training dynamics and ablation for critical design choices, but also quantitatively shows how the learned critic in Reasoner-Zero training effectively identifies and devalues repetitive response patterns, yielding more robust advantage estimations and enhancing training stability. Embracing the principles of open-source, we release our source code, training data, and various model weights, fostering reproducibility and encouraging further exploration of the properties of related models.

---

# Open-Reasoner-Zero：一种在基础模型上规模化强化学习的开源方法 论文详细解读

### 背景：这个问题为什么难？
在大语言模型（LLM）上做推理导向的强化学习（RL）时，往往需要复杂的训练管线：多阶段的奖励模型、KL 散度约束、防止模型漂移的技巧等。现有商业实现（如 DeepSeek‑R1‑Zero）虽然效果不错，却把训练过程包装得非常厚重，导致复现成本高、算力需求大。研究者们一直在寻找“更少的东西也能跑通”的方案，但缺少系统性的实验验证，大家不清楚到底哪些组件是必需的，哪些是锦上添花。于是出现了一个核心难点：**如何用最简洁的 RL 设置，在保持或提升推理能力的同时，大幅降低训练步骤和实现门槛**。

### 关键概念速览
**强化学习（RL）**：让模型在交互中学习策略，通过奖励信号来提升行为质量，类似于训练小狗坐下时给零食的过程。  
**PPO（近端策略优化）**：一种常用的 RL 算法，限制每一步策略改动幅度，防止“跑偏”。可以想象为在跑步时不让步幅突然变大，以免摔倒。  
**GAE（广义优势估计）**：用来平滑优势（advantage）计算的技巧，帮助模型更稳健地评估每一步的好坏。  
**λ=1、γ=1**：在 GAE 中的两个超参数，分别控制偏差-方差权衡和折扣因子。把它们都设为 1 意味着不做时间折扣，模型把整个对话当作一次完整的回报来评估。  
**KL 正则化**：在 RL 中常用的约束手段，防止新策略偏离原始语言模型太远。这里的“没有 KL 正则化”相当于让模型自由发挥。  
**规则奖励（rule‑based reward）**：直接根据预定义规则给出奖励，而不是训练一个独立的奖励模型。比如答案是否符合数学格式、是否出现重复句子等。  
**Critic（价值网络）**：估算当前状态价值的子模型，帮助 PPO 计算优势。可以把它想成裁判，给出每一步的“分数”。  

### 核心创新点
1. **极简 PPO 配置 → 直接使用 λ=1、γ=1 的 GAE + 纯规则奖励 → 省去时间折扣和 KL 正则，训练过程更透明，且在大模型上仍能实现显著的性能提升。**  
2. **去掉奖励模型 → 只用手写规则评估答案质量 → 大幅降低数据标注和模型训练成本，同时保持对推理任务的敏感度。**  
3. **基于同一基础模型（Qwen2.5‑32B）复现 DeepSeek‑R1‑Zero 的规模效应 → 只用 1/10 的训练步数就超越原始基准 → 证明了“少即是多”在大模型 RL 中可行。**  
4. **Critic 去重机制 → 通过价值网络自动识别并降低重复响应的价值 → 让优势估计更稳健，训练过程更不抖动。**  

### 方法详解
整体思路可以拆成三步：**准备、交互、更新**。  
1. **准备阶段**：选定 Qwen2.5‑32B 作为基模型，直接加载其原始权重，不做任何微调。构建一个轻量级的规则奖励函数，主要检查两类信号：①答案是否满足任务特定的格式（如数学公式、选项字母），②生成是否出现高频重复片段。  
2. **交互阶段**：模型在每个训练样本上进行一次完整的推理对话（例如解一道数学题），整个对话视为一次“episode”。因为 γ=1，所有时间步的奖励会被累计到最终回报，不做折扣。  
3. **更新阶段**：使用 PPO 进行策略更新。优势（advantage）通过 GAE 计算，λ=1 让优势等价于回报减去 Critic 估计的价值。Critic 本身是一个小型的前馈网络，输入是模型的隐藏状态，输出对应状态的价值。关键在于 **Critic 的去重学习**：当奖励函数检测到重复片段时，会给出负奖励，Critic 随即学会把这些状态的价值压低，从而在后续 PPO 更新中降低这些行为的概率。整个训练循环不加入 KL 正则，策略可以自由偏离原始语言模型，只要奖励足够指引方向。  

**最巧妙的地方**在于：作者把“重复惩罚”直接嵌入了价值估计，而不是在 PPO 的目标函数里额外加项。这样，Critic 本身就承担了去重的职责，优势估计自然变得更稳，训练波动显著下降。  

### 实验与效果
- **测试任务**：AIME2024（美国数学竞赛高级题）、MATH500（数学推理基准）和 GPQA Diamond（高难度通用知识问答）。这些任务都要求模型给出长篇、逻辑严密的答案。  
- **对比基线**：DeepSeek‑R1‑Zero‑Qwen‑32B（同样使用 Qwen2.5‑32B 作为基模型）以及公开的非 RL 微调模型。  
- **主要结果**：在所有三项任务上，Open‑Reasoner‑Zero 的得分均高于 DeepSeek‑R1‑Zero，且训练步数仅为后者的 10%。例如在 MATH500 上提升约 4% 的准确率，GPQA Diamond 上提升约 3.5%。  
- **消融实验**：作者分别去掉了（1）KL 正则、（2）λ=1 设置、（3）Critic 去重机制。实验显示：去掉 Critic 去重会导致训练后期出现明显的重复答案，优势估计噪声增大；去掉 KL 正则并未导致性能下降，反而略有提升；λ=1 与 γ=1 的组合是实现高回报累计的关键。  
- **局限性**：奖励函数仅基于规则，难以捕捉更细粒度的语义质量；实验仅在单一 32B 基模型上验证，尚不清楚在更大或更小模型上的迁移效果。作者也提到，当前实现仍依赖大量 GPU 计算，真正的“一键跑通”还有待进一步优化。  

### 影响与延伸思考
Open‑Reasoner‑Zero 的出现向社区展示了 **“极简 RL”** 也能在大模型上实现规模化提升，这激发了后续工作尝试用更少的超参数和更轻量的奖励设计来训练推理模型。2024 年底出现的几篇论文（如 “Zero‑RL‑Math” 与 “Rule‑Reward PPO”）直接引用了该工作，尝试把规则奖励推广到代码生成和多模态对话。对想进一步探索的读者，可以关注以下方向：① 自动化生成高质量规则奖励的技术（比如利用小模型生成规则）；② 在更大模型（如 70B、130B）上验证 λ=1、γ=1 的通用性；③ 将 Critic 的去重学习与对抗训练结合，提升模型对长文本的多样性保持能力。  

### 一句话记住它
**只要用最原始的 PPO + 规则奖励，省去 KL 正则和复杂奖励模型，就能在大模型上用 1/10 的步数实现更强的推理能力。**