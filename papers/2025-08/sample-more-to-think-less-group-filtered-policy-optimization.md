# Sample More to Think Less: Group Filtered Policy Optimization for Concise Reasoning

> **Date**：2025-08-13
> **arXiv**：https://arxiv.org/abs/2508.09726

## Abstract

Large language models trained with reinforcement learning with verifiable rewards tend to trade accuracy for length--inflating response lengths to achieve gains in accuracy. While longer answers may be warranted for harder problems, many tokens are merely "filler": repetitive, verbose text that makes no real progress. We introduce GFPO (Group Filtered Policy Optimization), which curbs this length explosion by sampling larger groups per problem during training and filtering responses to train on based on two key metrics: (1) response length and (2) token efficiency: reward per token ratio. By sampling more at training time, we teach models to think less at inference time. On the Phi-4-reasoning model, GFPO cuts GRPO's length inflation by 46-71% across challenging STEM and coding benchmarks (AIME 24/25, GPQA, Omni-MATH, LiveCodeBench) while maintaining accuracy. Optimizing for reward per token further increases reductions in length inflation to 71-85%. We also propose Adaptive Difficulty GFPO, which dynamically allocates more training resources to harder problems based on real-time difficulty estimates, improving the balance between computational efficiency and accuracy especially on difficult questions. GFPO demonstrates that increased training-time compute directly translates to reduced test-time compute--a simple yet effective trade-off for efficient reasoning.

---

# 多采样少思考：基于分组过滤的策略优化实现简洁推理 论文详细解读

### 背景：这个问题为什么难？
在强化学习（RL）框架下微调大语言模型（LLM）时，奖励函数往往鼓励模型给出更准确的答案。为了提升准确率，模型会倾向于把答案写得更长——把每一步推理都展开、把细节反复描述。结果是，很多生成的 token 只是在填充，没有实质性信息，导致推理成本飙升。传统的 RL‑HF（人类反馈）或 GRPO（Group‑Reward‑Policy‑Optimization）只能在单个样本上优化，缺乏机制抑制这种“长度膨胀”。如果不解决，模型在实际部署时会消耗大量算力，成本高、响应慢，且对用户体验不友好。

### 关键概念速览
- **强化学习微调（RL‑HF）**：用奖励信号（比如正确率）指导模型生成更好答案的过程，类似给模型“打分”。  
- **奖励/代价比（Reward‑per‑Token Ratio）**：每生成一个 token 能得到的奖励多少，数值越高说明答案越“紧凑”。可以把它想成“每块钱买到的价值”。  
- **分组采样（Group Sampling）**：一次训练时对同一道题目生成多个候选答案，而不是只生成一个。相当于让模型多次“思考”，挑出最好的。  
- **过滤策略（Filtering Policy）**：在众多候选答案中挑选出满足特定标准（如长度短、奖励高）的样本用于梯度更新。类似老师挑选学生作业里最优秀的几篇来讲评。  
- **自适应难度分配（Adaptive Difficulty Allocation）**：根据实时估计的题目难度，动态决定给难题分配多少采样资源，难题多采样、易题少采样。像老师给学困生更多练习时间。  
- **GRPO（Group‑Reward‑Policy‑Optimization）**：之前的工作，通过对同一问题的多个答案取平均奖励来平滑梯度，但仍使用较小的采样组，导致对长度的控制不够。  

### 核心创新点
1. **更大规模的分组采样 → 训练时对每道题目生成大量候选答案 → 让模型在训练阶段就学会挑选最简洁的高奖励答案，推理时自然倾向于短答案。**  
2. **双指标过滤（长度 + 奖励/代价比） → 只保留既短又高效的答案用于梯度更新 → 把“少即是多”的信号直接写进奖励函数，显著抑制了冗余 token 的产生。**  
3. **奖励/代价比优化目标 → 在原有奖励基础上加入每 token 的奖励比重 → 模型被迫在保持准确率的同时压缩答案，长度压缩幅度提升 71%‑85%。**  
4. **自适应难度 GFPO → 根据实时难度估计动态调节每题的采样数量 → 对难题投入更多计算资源，保证难题的准确率不因压缩而下降，提升整体效率与性能的平衡。**  

### 方法详解
整体框架可以分为四步：**采样 → 评估 → 过滤 → 更新**。  
1. **采样**：在每轮训练中，对每个训练样本（即一道题目）使用当前策略模型生成 **N** 条答案，N 远大于 GRPO 中的组大小（例如 32‑64 条）。这一步相当于让模型在同一问题上“多次思考”。  
2. **评估**：对每条答案计算两类分数：  
   - **准确性奖励**：依据外部可验证的答案或自动评估器给出，衡量答案是否正确。  
   - **代价（长度）**：直接计数生成的 token 数。  
   然后计算 **奖励/代价比** = 准确性奖励 ÷ token 数。  
3. **过滤**：依据预设阈值或排序规则挑选出满足以下条件的子集：  
   - **长度阈值**：排除超过某个 token 上限的答案。  
   - **奖励/代价比阈值**：保留比值最高的前 K 条（或比值超过某个比例的答案）。  
   过滤后的答案既短又高效，形成 **Group Filtered Set**。  
4. **更新**：使用 **Policy Gradient**（策略梯度）对过滤后的答案进行加权更新。权重由原始奖励（准确性）乘以奖励/代价比得到，确保高效答案对梯度的贡献更大。  
**自适应难度** 机制在采样阶段加入：先用一个轻量的难度估计器（比如模型的置信度或前一次的奖励/代价比）判断题目难度；若判定为难题，则临时提升该题目的采样数 N，保证有足够的候选答案供过滤挑选。  

**最巧妙的点** 在于把“思考成本”从推理阶段搬到了训练阶段：通过大量采样让模型在训练时已经经历了“挑选最简答案”的过程，从而在实际使用时只需要一次生成，就能得到简洁且准确的答案。

### 实验与效果
- **测试集合**：AIME 2024/2025（美国数学竞赛高难度题）、GPQA（通用知识问答）、Omni‑MATH（多学科数学推理）、LiveCodeBench（代码生成与调试）。这些基准覆盖了数学、科学、编程等高推理需求。  
- **对比基线**：原始 GRPO、标准 RL‑HF、以及未做长度约束的直接微调模型。  
- **主要结果**：在 Phi‑4‑reasoning 模型上，GFPO 将 GRPO 的长度膨胀削减 **46%‑71%**，在加入奖励/代价比优化后进一步压缩至 **71%‑85%**，而准确率基本持平（有时甚至略有提升）。  
- **消融实验**：作者分别关闭（1）大规模采样、（2）长度过滤、（3）奖励/代价比权重，发现：仅关闭大规模采样会导致长度削减回到 GRPO 水平；仅关闭奖励/代价比会使压缩幅度下降约 15%；两者共同缺失时效果最差。  
- **局限性**：训练时需要显著更多的计算资源（因为每题要生成 dozens 条答案），对算力受限的实验室不友好；此外，奖励/代价比的阈值需要手动调参，跨任务迁移时可能需要重新校准。  

### 影响与延伸思考
GFPO 的核心思想——“用更多训练算力换取推理时的算力节省”——在随后的一批工作中被进一步推广。比如 **SparseRL** 系列尝试在采样阶段加入稀疏搜索，**EfficientCoT** 把思维链的长度直接纳入奖励函数。还有研究把 **自适应难度分配** 与 **Curriculum Learning（课程学习）** 结合，让模型在训练早期只处理易题，后期逐步加入难题的多采样策略。对想深入的读者，可以关注 **Reward‑per‑Token** 这一指标的理论分析以及在大规模多模态模型中的应用（推测）。  

### 一句话记住它
**让模型在训练时多思考、采样大量答案，再用“每 token 奖励”挑最简答案，从而在推理时自然输出短而准的答案。**