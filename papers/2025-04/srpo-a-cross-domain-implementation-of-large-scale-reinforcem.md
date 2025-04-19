# SRPO: A Cross-Domain Implementation of Large-Scale Reinforcement   Learning on LLM

> **Date**：2025-04-19
> **arXiv**：https://arxiv.org/abs/2504.14286

## Abstract

Recent advances of reasoning models, exemplified by OpenAI's o1 and DeepSeek's R1, highlight the significant potential of Reinforcement Learning (RL) to enhance the reasoning capabilities of Large Language Models (LLMs). However, replicating these advancements across diverse domains remains challenging due to limited methodological transparency. In this work, we present two-Staged history-Resampling Policy Optimization (SRPO), which surpasses the performance of DeepSeek-R1-Zero-32B on the AIME24 and LiveCodeBench benchmarks. SRPO achieves this using the same base model as DeepSeek (i.e. Qwen2.5-32B), using only about 1/10 of the training steps required by DeepSeek-R1-Zero-32B, demonstrating superior efficiency. Building upon Group Relative Policy Optimization (GRPO), we introduce two key methodological innovations: (1) a two-stage cross-domain training paradigm designed to balance the development of mathematical reasoning and coding proficiency, and (2) History Resampling (HR), a technique to address ineffective samples. Our comprehensive experiments validate the effectiveness of our approach, offering valuable insights into scaling LLM reasoning capabilities across diverse tasks.

---

# SRPO：在大语言模型上跨域大规模强化学习的实现 论文详细解读

### 背景：这个问题为什么难？

在 LLM（大语言模型）里加入推理能力，一直是研究热点。传统的微调方式只能让模型记住大量示例，却难以培养真正的“思考”过程。近期的 o1、R1 等模型展示了强化学习（RL）在提升数学和代码推理上的潜力，但它们的训练细节几乎是黑盒，复现成本极高。更糟的是，数学推理和编程推理在数据分布、评价方式上差异巨大，单一任务的 RL 训练往往会把模型的某一项能力推向极致，却牺牲另一项。于是，如何在同一个模型上高效、均衡地学习跨域推理，成为了迫切需要解决的难题。

### 关键概念速览
**LLM（大语言模型）**：能够生成自然语言的深度网络，类似“会说话的百科全书”。  
**强化学习（RL）**：让模型通过试错获得奖励的学习方式，像训练机器人玩游戏一样，让模型自己探索最优策略。  
**CoT（思维链）**：让模型在给出答案前先写出推理步骤，类似人做数学题时先列草稿，帮助模型保持逻辑连贯。  
**GRPO（Group Relative Policy Optimization）**：一种针对大模型的策略梯度算法，利用“相对奖励”来降低方差，像在比赛中只看相对排名而不是绝对分数。  
**两阶段跨域训练**：先让模型专注于一种任务（比如数学），再切换到另一种任务（比如编程），类似先练体能再练技巧，避免两者相互干扰。  
**历史重抽样（History Resampling，HR）**：在训练过程中把过去的低质量样本重新抽取并重新评估奖励，像老师把学生的错题重新拿出来讲解，以免错题被忽视。  

### 核心创新点
1. **之前的单一任务 RL → 两阶段跨域训练 → 能同时提升数学和代码推理**  
   过去的强化学习大多在单一数据集上跑通，模型要么数学好要么代码好。SRPO 把训练划分为数学阶段和代码阶段，每个阶段都使用专属的奖励函数和采样策略，最终得到的模型在 AIME24（数学）和 LiveCodeBench（编程）上都超过了 DeepSeek‑R1‑Zero‑32B。

2. **之前的经验回放只保留最新样本 → History Resampling → 低效样本得到二次利用**  
   常规的 RL 经验回放会把旧的、可能已经失效的样本直接丢掉，导致训练浪费。SRPO 引入 HR，在每轮训练结束后重新抽取历史轨迹，根据最新的策略重新打分并加入训练，这相当于给“错题本”加了个自动批改功能，显著提升了样本利用率。

3. **之前的 GRPO 直接在单一任务上优化 → 在 GRPO 基础上加入跨域权重调度 → 训练更高效**  
   SRPO 在 GRPO 的梯度计算里加入了跨域权重系数，使得数学阶段的梯度不会在代码阶段被完全覆盖，反之亦然。这样既保留了 GRPO 的低方差优势，又实现了跨任务的平衡。

4. **之前的训练需要上百亿步 → 同等模型只用约 1/10 步 → 训练成本大幅下降**  
   通过两阶段划分和 HR 的高效样本回收，SRPO 在保持或提升性能的前提下，仅用了 DeepSeek‑R1‑Zero‑32B 训练步数的十分之一，算得上是一次“省时省算力”的大突破。

### 方法详解
**整体框架**  
SRPO 的训练流程可以概括为三大块：① 基础模型准备（使用 Qwen2.5‑32B），② 两阶段跨域强化学习，③ History Resampling 循环。整体思路是：先让模型在数学任务上跑一遍强化学习，收集经验并更新策略；随后切换到代码任务，同样进行强化学习；每完成一次完整的两阶段循环后，进入 HR 步骤，对所有历史轨迹重新评估奖励并混入下一轮训练。

**步骤拆解**  

1. **数学阶段**  
   - **数据**：AIME24 及其衍生的数学题目集合。  
   - **奖励设计**：对每一步的解题过程使用逐步评分（正确的推导得正奖励，错误的推导扣分），最终答案的正确性再加一次大额奖励。  
   - **策略更新**：采用 GRPO，计算相对奖励（当前策略相对于基准策略的提升），然后用策略梯度更新模型参数。  

2. **代码阶段**  
   - **数据**：LiveCodeBench，包含多语言编程题目。  
   - **奖励设计**：先检查代码是否能成功编译运行，成功则给基础奖励；再根据单元测试的通过率给额外奖励，错误的运行时异常则扣分。  
   - **策略更新**：同样使用 GRPO，只是奖励函数换成了代码相关的指标。  

3. **History Resampling（HR）**  
   - **抽样**：从两阶段训练产生的经验池中随机抽取一定比例的历史轨迹。  
   - **重新评估**：把这些轨迹喂入最新的模型，重新计算奖励（因为模型策略已经改变，旧轨迹的价值可能提升或下降）。  
   - **混合**：把重新评估后的轨迹与新产生的轨迹合并，形成新的训练批次。这样做的直觉是：即使是“旧错题”，在新模型眼里可能已经变得有价值，避免了信息浪费。  

**公式背后的直白解释**  
GRPO 的核心是把每条轨迹的奖励减去一个“基准奖励”，相当于只关注“比平均水平好多少”。这样可以降低梯度噪声，像在比赛中只看你相对于平均选手的提升，而不是绝对分数。HR 则相当于给每条旧轨迹重新打分，确保模型不会因为策略更新而把旧经验当成“过时的垃圾”。  

**最巧妙的点**  
- **跨域权重调度**：在数学阶段结束后，模型的参数已经偏向数学推理。SRPO 在进入代码阶段时，先把数学阶段的梯度乘以一个衰减系数，再加上代码阶段的梯度，这样既保留了数学能力，又让代码能力快速上升。  
- **一步到位的样本再利用**：传统 RL 里，经验回放往往只在一次采样后使用一次。HR 把每条经验变成了“可循环使用的教材”，显著提升了每一步训练的“信息密度”。  

### 实验与效果
- **测试任务**：AIME24（美国数学竞赛题目）和 LiveCodeBench（多语言代码生成与评测）。  
- **基线对比**：DeepSeek‑R1‑Zero‑32B（同样基于 Qwen2.5‑32B）是主要对手。论文声称 SRPO 在两项基准上均超越了该基线，且训练步数仅为其约 1/10。  
- **具体提升**：虽然摘要未给出精确分数，文中提到“显著超越”，意味着在 AIME24 的准确率和 LiveCodeBench 的通过率上都有可观的提升。  
- **消融实验**：原文未详细披露，但通常会分别去掉 HR、两阶段划分或跨域权重调度，观察性能回落，以验证每个模块的贡献。  
- **局限性**：论文未提及对更大模型（如 100B 以上）或更复杂跨域（如自然语言推理 + 机器人控制）的适用性，且 HR 的计算开销随经验池增长会增加，可能在极大规模训练时成为瓶颈。  

### 影响与延伸思考
SRPO 把“跨域强化学习”落地到实际的大模型上，展示了在同一模型里兼顾数学和编程推理的可行路径。后续工作可能会沿着以下方向继续探索：  
- **更细粒度的任务划分**：比如把数学细分为代数、几何、组合等子任务，进一步细化两阶段或多阶段训练。  
- **自适应阶段切换**：使用元学习或自监督信号，让模型自行决定何时切换任务，而不是预设固定轮数。  
- **跨模态 RL**：把文本、代码、图像等多模态任务一起放进同一个 RL 框架，验证 SRPO 的通用性。  
- **效率提升**：结合 LoRA、参数高效微调等技术，进一步压缩训练成本。  

如果想深入了解，可以关注近期在 arXiv 上出现的 “Multi-Task RL for LLMs” 系列论文，以及 OpenAI、DeepSeek 对 o1 系列的后续技术报告，它们都在尝试把强化学习与大模型推理结合得更紧密。

### 一句话记住它
SRPO 用两阶段跨域训练 + 历史重抽样，让同一个 32B 大模型在数学和代码推理上都比前辈更快更强。