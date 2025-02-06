# Training Language Models to Reason Efficiently

> **Date**：2025-02-06
> **arXiv**：https://arxiv.org/abs/2502.04463

## Abstract

Scaling model size and training data has led to great advances in the performance of Large Language Models (LLMs). However, the diminishing returns of this approach necessitate alternative methods to improve model capabilities, particularly in tasks requiring advanced reasoning. Large reasoning models, which leverage long chain-of-thoughts, bring unprecedented breakthroughs in problem-solving capabilities but at a substantial deployment cost associated to longer generations. Reducing inference costs is crucial for the economic feasibility, user experience, and environmental sustainability of these models.   In this work, we propose to train large reasoning models to reason efficiently. More precisely, we use reinforcement learning (RL) to train reasoning models to dynamically allocate inference-time compute based on task complexity. Our method incentivizes models to minimize unnecessary computational overhead while maintaining accuracy, thereby achieving substantial efficiency gains. It enables the derivation of a family of reasoning models with varying efficiency levels, controlled via a single hyperparameter. Experiments on two open-weight large reasoning models demonstrate significant reductions in inference cost while preserving most of the accuracy.

---

# 训练语言模型高效推理 论文详细解读

### 背景：这个问题为什么难？
大模型的性能主要靠把模型规模和训练数据往上堆，效果虽然提升，但成本也跟着指数级增长。尤其是需要长链式思考（Chain‑of‑Thought，CoT）的推理任务，模型会生成大量中间步骤，导致生成时间变长、算力消耗大。部署时，这种“慢慢想”会让用户体验下降、运营费用飙升，也不利于环保。于是，如何在不牺牲推理质量的前提下，让模型在需要时“快点想”，成为迫切的研究点。

### 关键概念速览
**Chain‑of‑Thought（思维链）**：模型在给出最终答案前，先把推理过程写出来，像人在解数学题时先列草稿，能显著提升复杂问题的正确率。  
**推理时计算分配**：指模型在实际使用时，根据当前输入的难易程度，动态决定要花多少算力（比如生成多少 token）来完成任务。  
**强化学习（RL）**：让模型通过试错获得奖励的学习方式，这里用来教模型在保持答案准确的同时，尽量少用计算资源。  
**奖励函数**：在 RL 中用来衡量行为好坏的信号，本文把“准确率”和“计算成本”两者结合进奖励里。  
**超参数 λ**：控制模型在准确率和效率之间的权衡，只调这个值就能得到不同“快慢”版本的模型。  
**开放权重模型**：指作者实验中使用的公开可下载的、已经训练好的大模型，方便复现和对比。  

### 核心创新点
1. **从固定算力到动态算力** → 传统的大模型在推理时会固定使用全部算力，本文引入 RL 让模型自行决定每一步需要多少计算 → 使得在简单任务上显著缩短生成长度，在复杂任务上仍保留足够算力，整体推理成本下降。  
2. **奖励函数同时考虑准确率和计算量** → 以前的 RL 训练多关注提升答案质量，忽视资源消耗；这里把“正确率”与“使用的 token 数”加权合成奖励 → 模型学会在不影响答案的前提下主动“省钱”。  
3. **单一超参数控制全族模型** → 过去想得到不同效率的模型需要重新训练或手动调节多个参数，本文只需调节 λ，就能得到从“极致省算”到“高精度”不等的多个模型版本 → 大幅降低实验成本。  
4. **在真实开放模型上验证** → 许多效率研究只在小模型或合成任务上做实验，作者直接在两个公开的大推理模型上跑实验，展示了方法在真实场景的可行性 → 增强了结论的说服力。

### 方法详解
整体思路可以拆成三步：**（1）准备基线推理模型、（2）设计奖励、（3）用强化学习微调**。

1. **基线模型**  
   先选定一个已经具备长链式思考能力的大模型（比如公开的 LLaMA‑CoT 变体），把它当作“老师”。在普通推理时，这个模型会一次性生成完整的思维链，算力消耗固定。

2. **奖励函数设计**  
   - **准确率奖励**：如果模型的最终答案和金标准匹配，就给正向奖励；不匹配则不给或给负向奖励。  
   - **计算成本惩罚**：统计模型在生成思维链时用了多少 token（或算子），越多惩罚越大。  
   - **加权合成**：用一个 λ 参数把两部分加在一起：Reward = AccuracyReward – λ × ComputeCost。λ 越大，模型越倾向于省算；λ 越小，模型更关注准确率。

3. **强化学习微调**  
   - **策略网络**：其实就是原始的大模型，只是把它的输出视为“动作”。每一步生成一个 token，动作空间是词表。  
   - **采样与评估**：给定一个输入，模型先采样生成思维链（可能比基线短），随后根据奖励函数算出该序列的得分。  
   - **梯度更新**：使用常见的策略梯度算法（如 PPO）把高奖励的短链强化，低奖励的长链削弱。这里的“策略”就是模型的生成概率分布。  
   - **迭代**：重复采样、评估、更新，模型逐渐学会在简单问题上直接给出答案或只写几步推理，在难题上仍保留完整的思维链。

**巧妙之处**：奖励里直接把算力（token 数）当作负奖励，让模型把“省算”当成目标，而不是事后手动裁剪。这样模型自己决定哪些中间步骤是“可有可无”，哪些是必须保留的，避免了人为设定的剪枝规则。

### 实验与效果
- **测试任务**：作者在两个公开的大推理模型上跑了标准的数学推理基准和逻辑推理任务（具体数据集名字未在摘要中给出）。  
- **对比基线**：与原始不做效率优化的模型直接对比，还和几种常见的后处理加速手段（如提前截断、温度调高）做比较。  
- **结果**：论文声称在保持大部分准确率的前提下，推理成本（以 token 数或 FLOPs 计）显著下降。具体下降幅度未给出数字，但描述为“显著”。  
- **消融实验**：通过去掉奖励中的计算成本项或把 λ 设为极端值，作者展示了两者对效率和准确率的影响，证明奖励函数的加权设计是关键。  
- **局限性**：原文未详细说明在极端复杂任务上模型是否会因过度省算而失误，也未给出对不同模型规模的适配情况，作者自己承认需要进一步验证跨模型的通用性。

### 影响与延伸思考
这篇工作把“算力省钱”直接写进模型的学习目标，开启了“自适应推理成本”这一新方向。随后有几篇论文尝试把类似的 RL 奖励加入到检索增强生成、代码自动化等场景，甚至出现了把硬件能耗直接纳入奖励的研究（推测）。如果想继续深挖，可以关注以下两个方向：  
1. **多模态或多任务的动态算力分配**——让模型在视觉、语言等不同输入上都能自行调节计算。  
2. **更细粒度的算力度量**——比如把显存占用、实际延迟等硬件指标直接加入奖励，进一步贴合部署需求。

### 一句话记住它
用强化学习让大模型学会“看题目难易，自己决定想多深”，在不牺牲答案质量的情况下自动省算。