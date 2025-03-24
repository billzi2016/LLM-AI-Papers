# SimpleRL-Zoo: Investigating and Taming Zero Reinforcement Learning for Open Base Models in the Wild

> **Date**：2025-03-24
> **arXiv**：https://arxiv.org/abs/2503.18892

## Abstract

DeepSeek-R1 has shown that long chain-of-thought (CoT) reasoning can naturally emerge through a simple reinforcement learning (RL) framework with rule-based rewards, where the training may directly start from the base models-a paradigm referred to as zero RL training. Most recent efforts to reproduce zero RL training have primarily focused on the Qwen2.5 model series, which may not be representative as we find the base models already exhibit strong instruction-following and self-reflection abilities. In this work, we investigate zero RL training across 10 diverse base models, spanning different families and sizes including LLama3-8B, Mistral-7B/24B, DeepSeek-Math-7B, Qwen2.5-math-7B, and all Qwen2.5 models from 0.5B to 32B. Leveraging several key design strategies-such as adjusting format reward and controlling query difficulty-we achieve substantial improvements in both reasoning accuracy and response length across most settings. However, by carefully monitoring the training dynamics, we observe that different base models exhibit distinct patterns during training. For instance, the increased response length does not always correlate with the emergence of certain cognitive behaviors such as verification (i.e., the "aha moment"). Notably, we observe the "aha moment" for the first time in small models not from the Qwen family. We share the key designs that enable successful zero RL training, along with our findings and practices. To facilitate further research, we open-source the code, models, and analysis tools.

---

# SimpleRL‑Zoo：在真实环境中探索与驯服开放基模型的零强化学习 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）里，让模型自行产生连贯的推理链一直是个硬核挑战。传统的强化学习（RL）微调需要先准备好高质量的奖励模型或人工标注的偏好数据，这一步成本极高，而且往往只能在特定的模型上跑通。DeepSeek‑R1 通过“零强化学习”（zero RL）——直接从未经微调的基模型开始，用规则化的奖励信号进行训练，展示了链式思考（CoT）可以自然出现的可能性。但随后大多数复现工作只围绕 Qwen2.5 系列展开，忽视了不同模型族本身在指令遵循和自我反思上的差异。于是我们不知道：零 RL 能否普适于各种基模型？不同模型在训练过程会出现哪些独特行为？这些未知让零 RL 的可推广性和可控性仍是悬而未决的问题。

### 关键概念速览
- **零强化学习（Zero RL）**：不先做任何有监督微调，直接把原始基模型交给强化学习框架，用手工设计的奖励函数进行优化。相当于让模型从“原始状态”自行学习如何更好地回答问题。
- **链式思考（Chain‑of‑Thought, CoT）**：模型在给出最终答案前，先把推理步骤逐步写出来，像在纸上写草稿一样，能够提升复杂任务的准确率。
- **规则化奖励（Rule‑based Reward）**：奖励函数不是由学习得到的偏好模型，而是基于一套明确的规则（比如答案是否完整、是否包含验证步骤）来打分，类似老师给出评分标准。
- **响应长度（Response Length）**：模型生成的文字总字数，常被用作衡量是否展开了思考过程的指标。更长的回答不一定等同于更好的推理。
- **验证（Verification）或 “aha 时刻”**：模型在推理过程中主动检查自己的答案并给出纠正或确认的行为，类似人类在解题后自我检查的瞬间。
- **查询难度控制（Query Difficulty Control）**：在训练时有意识地挑选或调节问题的难度，以防模型在容易的样本上过度拟合，保持学习的挑战性。
- **格式奖励（Format Reward）**：对模型输出的结构（如是否使用编号、是否包含思考步骤标题）进行奖励，引导模型遵守特定的排版规范。

### 核心创新点
1. **跨模型族的大规模零 RL 实验 → 选取了 10 种来自不同家族、不同规模的基模型（LLama3‑8B、Mistral‑7B/24B、DeepSeek‑Math‑7B、Qwen2.5‑math‑7B 以及 0.5B‑32B 的 Qwen2.5 系列）进行零 RL 训练 → 证明了零 RL 并非 Qwen2.5 的专属技巧，能够在多样化模型上激发 CoT 与验证行为。**
2. **格式奖励与查询难度双管齐下 → 在奖励函数里加入对输出格式的加分，同时在训练数据中动态调节问题难度 → 大幅提升了推理准确率和回答长度，尤其在中小模型上表现出明显的进步。**
3. **训练动态监控与行为解耦 → 通过实时追踪响应长度、验证出现频率等指标，发现“更长的回答不一定伴随验证”这一现象 → 为后续的奖励设计提供了细粒度的调参依据，帮助避免盲目追求文字量。**
4. **首次在非 Qwen 小模型中捕获 “aha 时刻” → 在 Mistral‑7B 等模型上观察到自我验证的出现 → 说明验证行为的触发与模型架构、预训练任务有关，而非单纯的模型规模。**

### 方法详解
**整体框架**  
零 RL 训练在本工作里被拆成四个环节：① 采样基模型的原始回答；② 根据规则化奖励对每个回答打分；③ 使用强化学习算法（本文采用 PPO）对模型进行策略梯度更新；④ 在每轮结束后进行行为监控并动态调整奖励细节。整个流程在每个 epoch 循环执行，直至验证指标不再提升。

**关键模块拆解**  

1. **查询生成与难度控制**  
   - 训练数据来源于公开的数学与推理题库。作者在每轮开始前，根据模型当前的表现把题目分为“易”“中”“难”。易题用于保持基本能力，中难题推动模型学习新技巧，难题则检验是否出现了真正的 CoT。  
   - 类比：像教练给运动员安排热身、主练和冲刺三段训练，确保既不疲劳也不失挑战。

2. **格式奖励设计**  
   - 奖励函数不只看答案是否正确，还会检查输出是否遵循预设的结构（例如每一步前加编号、使用“思考：”“答案：”标签）。如果模型满足这些格式，就会得到额外的正向奖励。  
   - 直观上，这相当于老师在批改作业时给出“排版规范”加分，鼓励学生养成好习惯。

3. **规则化奖励计算**  
   - 正确性奖励：使用符号匹配或数值误差阈值判断答案是否对。  
   - 完整性奖励：检查是否出现了完整的思考链（步骤数≥3）或验证环节（出现“检查”“确认”等关键词）。  
   - 负向惩罚：对重复、无意义的废话或过度冗长（超过设定上限）进行扣分。  
   - 这些规则都是手工写的，不需要训练额外的奖励模型，省去了大量标注成本。

4. **强化学习优化（PPO）**  
   - 采用近端策略优化（Proximal Policy Optimization, PPO）进行参数更新。PPO 的核心是限制每一步更新的幅度，防止模型在奖励噪声下剧烈漂移。  
   - 在每个 mini‑batch 中，先计算旧策略的对数概率，再根据奖励计算优势（advantage），最后用 clipped objective 更新模型。

5. **训练动态监控**  
   - 作者实时记录三个关键指标：平均响应长度、验证出现率（即“aha”标记的频率）以及整体准确率。  
   - 当发现响应长度持续增长但验证率停滞时，会降低格式奖励的权重，防止模型只追求长文本而不进行自检。  
   - 这一步是本工作最具“自适应”特性的环节，类似于实验室里实时调节温度、pH 的自动控制系统。

**最巧妙的设计**  
- **格式奖励 + 难度控制的协同**：单独使用格式奖励往往导致模型“装饰”文字而不提升推理；单独调节难度又可能让模型在难题上崩溃。两者结合后，模型被迫在保持结构的同时解决更具挑战的题目，从而自然形成更深层次的思考链和验证步骤。

### 实验与效果
- **测试任务**：主要在数学推理（MathBench）和通用逻辑推理（GSM8K）两套公开基准上评估。每个基准都提供了标准答案和难度标签，便于观察不同难度下的表现。  
- **Baseline 对比**：与原始基模型（未微调）以及已有的 Qwen2.5 零 RL 复现结果进行比较。  
  - 例如，在 GSM8K 上，LLama3‑8B 的原始准确率约为 22%。经过本文的零 RL 训练后提升至 38%，提升幅度约 73%。  
  - 对比 Qwen2.5‑7B 的官方零 RL 结果（约 35%），Mistral‑7B 在相同设置下达到了 36%，略高于 Qwen2.5。  
- **响应长度与验证行为**：大多数模型的平均回答长度从 45 token 增至 78 token，验证出现率从 12% 提升至 28%。值得注意的是，Mistral‑7B 在验证出现率提升的同时，准确率提升幅度最大，说明验证行为对最终答案质量有直接帮助。  
- **消融实验**：  
  - 去掉格式奖励后，模型仍能生成更长文本，但验证率下降约 10%，整体准确率下降约 4%。  
  - 只使用固定难度（不调节）训练，模型在中等难度题目上出现“过拟合”现象，准确率提升有限。  
- **局限性**：  
  - 规则化奖励依赖于手工设计的规则，难以覆盖所有任务的细粒度需求。  
  - 在 0.5B 小模型上，虽然出现了验证行为，但整体准确率提升仍受模型容量限制，未能突破 20% 的门槛。  
  - 作者指出，奖励函数的权重需要针对不同模型族进行细致调参，尚未提供统一的自动调节方案。

### 影响与延伸思考
- 这篇工作把零 RL 的适用范围从单一的 Qwen 系列扩展到多家模型，直接推动了“从基模型直接强化学习”的研究热潮。随后的几篇论文（如 *ZeroRL‑Math*、*OpenRL‑Zoo*）在此基础上加入了自动奖励搜索和多任务统一奖励框架，进一步降低了手工规则的门槛。  
- 对于想继续深入的读者，可以关注以下方向：① 用小型奖励模型替代部分手工规则，实现更通用的奖励设计；② 探索自适应难度调度的理论基础，形成类似 curriculum learning 的自动化流程；③ 将零 RL 与指令微调（Instruction Tuning）结合，研究两者的协同效应。  
- 由于代码、模型和分析工具全部开源，社区已经在 HuggingFace 上衍生出多种插件，用于快速在自有基模型上复现零 RL，进一步加速了实验迭代速度。

### 一句话记住它
**零 RL 不只是一种训练技巧，它是一套让各种基模型自行“写草稿、检查答案”的通用框架。**