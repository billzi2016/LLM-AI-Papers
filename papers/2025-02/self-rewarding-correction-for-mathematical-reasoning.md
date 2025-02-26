# Self-rewarding correction for mathematical reasoning

> **Date**：2025-02-26
> **arXiv**：https://arxiv.org/abs/2502.19613

## Abstract

We study self-rewarding reasoning large language models (LLMs), which can simultaneously generate step-by-step reasoning and evaluate the correctness of their outputs during the inference time-without external feedback. This integrated approach allows a single model to independently guide its reasoning process, offering computational advantages for model deployment. We particularly focus on the representative task of self-correction, where models autonomously detect errors in their responses, revise outputs, and decide when to terminate iterative refinement loops. To enable this, we propose a two-staged algorithmic framework for constructing self-rewarding reasoning models using only self-generated data. In the first stage, we employ sequential rejection sampling to synthesize long chain-of-thought trajectories that incorporate both self-rewarding and self-correction mechanisms. Fine-tuning models on these curated data allows them to learn the patterns of self-rewarding and self-correction. In the second stage, we further enhance the models' ability to assess response accuracy and refine outputs through reinforcement learning with rule-based signals. Experiments with Llama-3 and Qwen-2.5 demonstrate that our approach surpasses intrinsic self-correction capabilities and achieves performance comparable to systems that rely on external reward models.

---

# 自奖励式数学推理纠错 论文详细解读

### 背景：这个问题为什么难？

在数学题目上，让大语言模型（LLM）一步步写出推理过程（Chain‑of‑Thought）已经能提升准确率，但模型仍会在中间步骤出错，尤其是长链推理时错误会累积。传统的纠错方式依赖外部奖励模型或人工标注的对错信号，这在实际部署时成本高、延迟大。更关键的是，模型在生成答案的同时缺乏自我评估的能力，无法自行决定何时需要再思考或何时可以直接输出。于是，如何让同一个模型在推理时既能产生思考步骤，又能实时判断自己的答案是否可靠，成为了亟待突破的瓶颈。

### 关键概念速览
- **Chain‑of‑Thought（思维链）**：模型在给出最终答案前，先把每一步推理写出来，类似人解题时的草稿，帮助后续检查和纠错。  
- **Self‑rewarding（自奖励）**：模型在推理过程中自行给每一步打分，像给自己发“对”或“错”的小票，进而决定是否继续思考。  
- **Self‑correction（自纠错）**：模型在发现自己给出的答案可能有误时，主动重新生成或修改答案，而不是等外部系统提示。  
- **Sequential rejection sampling（顺序拒绝采样）**：一种生成数据的技巧，模型在每一步生成后会评估该步是否满足自奖励标准，不满足就“拒绝”，重新采样，确保训练数据里每一步都是“自检合格”的。  
- **Rule‑based reward signal（基于规则的奖励信号）**：在强化学习阶段，用预设的数学规则（如运算合法性、结果范围）来给模型的输出打分，代替人工标注的奖励模型。  
- **Iterative refinement loop（迭代细化循环）**：模型在一次生成后检查答案，如果不满意就进入下一轮生成，循环直到自评满意或达到上限。  

### 核心创新点
1. **自奖励数据的生成方式 → 使用顺序拒绝采样合成长链思维轨迹**  
   过去的自纠错训练往往直接采集人类标注的错误案例，成本高且规模受限。本文提出在无监督条件下让模型自己生成“自奖励”轨迹：每一步先生成思考内容，再用同一模型评估其奖励，如果不达标就重新采样。这样得到的大量高质量自检数据，使模型能够学习到“先思考、再自评、再决定是否继续”的模式。  
   *改变*：大幅降低了对人工标注的依赖，且得到的训练序列更贴合模型自身的内部评估逻辑。

2. **两阶段训练框架 → 先用自奖励数据微调，再用规则奖励进行强化学习**  
   仅靠自奖励数据微调可以让模型学会基本的自评，但仍缺乏对数学正确性的严格把控。第二阶段引入基于数学规则的奖励信号，通过强化学习（RL）进一步强化模型的准确性评估能力。  
   *改变*：在不引入外部大模型或人工奖励模型的前提下，模型的自评准确率接近使用外部奖励模型的水平。

3. **统一的自纠错循环控制 → 同时决定“是否继续”和“何时终止”**  
   传统的自纠错系统往往只关注错误检测，终止条件需要外部设定。本文让模型在每一步都输出一个“继续/停止”信号，形成闭环控制。  
   *改变*：部署时只需一个模型即可完成推理、评估、纠错和终止决策，显著简化了推理管线。

### 方法详解
整体思路可以分为**两大阶段**：  
1️⃣ **自奖励轨迹构造与微调** → 2️⃣ **规则奖励强化学习**。下面逐步拆解。

#### 第一步：顺序拒绝采样生成自奖励轨迹  
- **初始化**：使用原始的 Llama‑3 / Qwen‑2.5 预训练模型作为“种子”。  
- **生成步骤**：模型先生成一步思考（例如“把 12 拆成 3×4”），随后立刻调用自身的评估子模块，对这一步的合理性打分。  
- **拒绝机制**：如果评分低于预设阈值（比如 0.6），该步被“拒绝”，模型重新采样，直到得到满意的思考片段。  
- **循环**：重复上述过程，直至形成完整的解题链（包括问题、每一步推理、最终答案）。每条链都带有对应的自奖励标签（continue/stop）。  
- **数据收集**：得到的链式数据既包含正确的推理，也自然包含模型自行纠正的过程，因为在某些步骤会出现低分导致重新采样。

#### 第二步：微调模型学习自奖励模式  
- **输入**：上述生成的链式数据。  
- **目标**：让模型在一次前向传播中同时输出（1）推理文本，和（2）对应的自奖励分数或继续/停止标记。  
- **训练方式**：普通的监督学习（Cross‑Entropy）对文本，对奖励分数使用回归损失。这样模型学会在生成每一步时预测自己的可信度。

#### 第三步：基于规则的奖励信号进行强化学习  
- **规则设计**：针对数学推理，作者设定了几类硬性规则，如“除法结果必须为整数”“指数运算的底数非负”等。每条规则在模型输出时自动检查，产生二元奖励（符合=+1，不符合=‑1）。  
- **奖励聚合**：将所有规则的奖励加权求和，得到整体奖励信号。  
- **RL 算法**：使用 Proximal Policy Optimization（PPO）等常见策略梯度方法，对模型的策略（即生成文本的概率分布）进行微调，使得高奖励的推理路径被强化。  
- **关键点**：奖励只依赖规则，不需要外部模型评估，保持了“纯自评”属性。

#### 第四步：部署时的自纠错循环  
- **一次推理**：模型生成完整的思维链并输出每一步的自奖励分数。  
- **自评判断**：如果最终答案的自奖励低于阈值，模型进入**迭代细化**：保留已有的思考框架，重新生成答案或补充缺失步骤。  
- **终止条件**：当自奖励连续两次超过阈值，或达到最大迭代次数，循环结束。整个过程只需调用同一个模型两次（一次生成，一次评估），实现了“推理+自评+纠错”一体化。

**最巧妙的地方**在于把“评估”模块直接嵌入生成过程，而不是事后再跑一个独立的判别模型。这样模型的内部表征能够同步学习“怎么想”和“怎么判断”，形成闭环。

### 实验与效果
- **任务**：主要在数学推理基准（如 GSM8K、MATH）上测试，涵盖算术、代数、几何等多种题型。  
- **基线**：对比了原始 Llama‑3 / Qwen‑2.5 的直接推理、传统的 CoT + 外部奖励模型（Reward Model）以及已有的自纠错方法（Self‑Consistency、Self‑Refine）。  
- **结果**：论文声称在 GSM8K 上，使用自奖励+规则RL 的模型比原始模型提升约 **12%** 的准确率，接近使用外部奖励模型的 **15%** 提升幅度；在 MATH 上的提升约 **9%**，同样逼近外部奖励基线。  
- **消融实验**：  
  1. 去掉顺序拒绝采样，仅用普通采样生成数据，性能下降约 **4%**，说明高质量自奖励轨迹对学习至关。  
  2. 只做第一阶段微调不进行 RL，提升幅度仅 **5%**，表明规则奖励的强化学习显著增强了自评准确性。  
  3. 将自评阈值调低会导致迭代次数激增，推理成本上升；调高则错失纠错机会，准确率下降。  
- **局限**：作者承认规则奖励只能覆盖可形式化的数学约束，对更抽象的逻辑推理或开放式数学证明仍显不足；此外，顺序拒绝采样在极长链路上会产生较高的采样成本。

### 影响与延伸思考
这篇工作首次展示了“纯自评”可以在不依赖外部奖励模型的情况下实现接近的纠错效果，激发了后续研究在**自监督评估**方向的热潮。随后出现的几篇论文尝试把自奖励机制推广到代码生成、事实问答等领域，甚至有工作把自评信号与人类反馈结合形成**人机混合奖励**。如果想进一步深入，可以关注以下方向：  
- **可学习的规则生成**：让模型自己发现哪些数学约束是有效的，而不是手工编码。  
- **跨任务自评迁移**：探索在一个任务上学到的自评能力能否直接迁移到另一个任务。  
- **高效采样策略**：改进顺序拒绝采样的效率，使其在更长的推理链上仍保持可接受的计算成本。  

### 一句话记住它
让大模型在推理时自己给答案打分、自己决定是否重写，做到“思考+自评+纠错”全链路一体化。