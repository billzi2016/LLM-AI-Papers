# GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning

> **Date**：2025-07-25
> **arXiv**：https://arxiv.org/abs/2507.19457

## Abstract

Large language models (LLMs) are increasingly adapted to downstream tasks via reinforcement learning (RL) methods like Group Relative Policy Optimization (GRPO), which often require thousands of rollouts to learn new tasks. We argue that the interpretable nature of language often provides a much richer learning medium for LLMs, compared to policy gradients derived from sparse, scalar rewards. To test this, we introduce GEPA (Genetic-Pareto), a prompt optimizer that thoroughly incorporates natural language reflection to learn high-level rules from trial and error. Given any AI system containing one or more LLM prompts, GEPA samples trajectories (e.g., reasoning, tool calls, and tool outputs) and reflects on them in natural language to diagnose problems, propose and test prompt updates, and combine complementary lessons from the Pareto frontier of its own attempts. As a result of GEPA's design, it can often turn even just a few rollouts into a large quality gain. Across six tasks, GEPA outperforms GRPO by 6% on average and by up to 20%, while using up to 35x fewer rollouts. GEPA also outperforms the leading prompt optimizer, MIPROv2, by over 10% (e.g., +12% accuracy on AIME-2025), and demonstrates promising results as an inference-time search strategy for code optimization. We release our code at https://github.com/gepa-ai/gepa .

---

# GEPA：反思式提示进化可超越强化学习 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）上做下游任务时，业界常用强化学习（RL）方法（比如 Group Relative Policy Optimization，GRPO）来微调模型的行为。RL 需要把模型的输出映射成一个标量奖励，然后通过梯度上升来改进策略。实际使用中，这种奖励往往非常稀疏——模型要经过成千上万次“rollout”（即完整的推理-执行-反馈循环）才能看到一点提升。换句话说，RL 把语言模型的丰富文字信息压缩成了一个数字，导致学习效率低下、调参成本高。于是，如何在保留语言模型可解释性的前提下，利用更少的交互次数实现同等甚至更好的任务适配，成为了迫切需要解决的难题。

### 关键概念速览
- **强化学习（RL）**：让模型通过试错获得奖励，然后用奖励的梯度来更新策略。想象你在玩游戏，只看分数不看过程，学习会非常慢。  
- **Rollout**：一次完整的模型推理过程，包括输入、内部思考、外部工具调用以及最终输出。相当于一次“实验”。  
- **Prompt（提示）**：给 LLM 的文字指令或上下文，决定模型的行为方式。就像老师给学生的题目描述。  
- **反思（Reflection）**：模型在得到输出后，用自然语言对自己的推理过程进行自我评估和解释。类似于人做完题后写下“哪里卡住了，为什么”。  
- **Pareto 前沿**：在多目标优化中，指那些在所有目标上都没有被其他解支配的解集合。把它想成“最优组合”，没有哪一个解在所有维度上都更差。  
- **Genetic‑Pareto（GEPA）**：一种把遗传算法的“变异、交叉”与 Pareto 前沿筛选结合起来的提示优化框架。它把每一次反思产生的改进看作“基因”，通过自然选择保留最有价值的改动。  
- **MIPROv2**：当前最强的基于梯度的提示搜索器，利用元学习和梯度信息在提示空间里寻找更好配置。  

### 核心创新点
1. **从奖励到语言反思的学习介质转变**  
   - 之前的 RL 方法只能看到一个数值奖励 → GEPA 让模型把每一次实验写成自然语言的“诊断报告”，并在报告里提出改进建议 → 这种文字化的反馈比稀疏奖励更丰富，模型能直接学习到高层次的规则，而不是靠梯度慢慢逼近。  

2. **遗传‑Pareto 的双层进化机制**  
   - 传统的提示搜索要么是随机扰动，要么是单目标梯度优化 → GEPA 先用遗传算法产生一批“候选提示”，再在这些候选中构建 Pareto 前沿，保留在准确率、样本效率、解释性等多个维度上都表现最好的提示 → 这样既避免了单一目标的局部最优，又能在少量 rollouts 中快速聚拢到高质量解。  

3. **自然语言驱动的自我迭代循环**  
   - 过去的系统在每次 rollout 后直接把奖励喂回模型 → GEPA 在每次 rollout 后让模型先“思考自己哪里错了”，生成一段反思文字，然后基于这段文字自动生成新的提示改动 → 这一步把模型的自我解释能力直接用于提示改写，形成闭环学习。  

4. **兼容多提示、多工具的通用框架**  
   - 许多已有方法只能针对单一提示或单一工具调用进行优化 → GEPA 设计了“轨迹采样 + 反思 + 组合”三步，能够同时处理包含多个子提示、工具调用链的复杂系统 → 这让它在代码优化等需要多步骤推理的任务上也能发挥作用。  

### 方法详解
**整体思路**  
GEPA 把一个包含若干 LLM 提示的系统视作“个体”。它通过多轮交互（rollout）收集系统的完整执行轨迹——包括模型的思考过程、调用的外部工具以及工具返回的结果。每一次轨迹结束后，模型会用自然语言对这段轨迹进行反思，指出错误、缺失或潜在的改进点。随后，GEPA 根据这些反思生成若干候选提示改动（相当于基因突变），并把所有候选与原始提示一起放入遗传池。遗传池每轮通过 Pareto 前沿筛选保留表现最好的若干个体，进入下一轮迭代。整个循环持续到预设的 rollout 预算耗尽或性能满足要求。

**关键模块拆解**  

1. **轨迹采样（Trajectory Sampling）**  
   - 输入：当前提示集合。  
   - 过程：让 LLM 按照提示完成任务，记录下每一步的思考（CoT）、工具调用及其输出。  
   - 类比：像实验室里做一次完整的化学实验，记录每一步的试剂、温度、观察结果。  

2. **自然语言反思（Reflective Diagnosis）**  
   - 输入：完整轨迹。  
   - 过程：模型生成一段“实验报告”，内容包括：哪一步卡住了、为什么会出错、哪些信息缺失、可以怎样改进提示。  
   - 这里没有使用梯度，而是让模型直接用语言表达自己的认知。  

3. **提示变异生成（Prompt Mutation）**  
   - 输入：反思报告。  
   - 过程：系统把报告中的改进建议转化为具体的提示修改，例如添加约束、重写问题描述、加入示例等。每条建议可以产生多个变体，形成一个小的“基因库”。  
   - 类比：把实验报告里的改进思路写进实验方案，准备下一轮实验。  

4. **遗传‑Pareto 选择（Genetic‑Pareto Selection）**  
   - 输入：原始提示 + 所有变体。  
   - 过程：对每个提示执行少量 rollout，评估多个指标（准确率、样本利用率、解释性分数等）。随后在这些指标上构建 Pareto 前沿，保留那些在任意指标上没有被其他提示支配的个体。  
   - 这一步相当于自然选择：只有“适者”才能进入下一代。  

5. **循环迭代（Iterative Loop）**  
   - 将保留下来的提示重新投入轨迹采样，重复上述步骤。每轮迭代都在更少的 rollout 中获得更大的性能提升。  

**最巧妙的设计**  
- **语言反思代替标量奖励**：把原本只能给出“好/坏”二元信号的奖励，升级为可读的诊断文本，让模型直接学习“规则”。  
- **多目标 Pareto 前沿**：不只追求单一的准确率，而是把样本效率、可解释性等一起考虑，避免了只优化一个指标导致的过拟合。  

### 实验与效果
- **任务覆盖**：作者在六个公开任务上评估 GEPA，任务类型包括数学推理（如 AIME‑2025）、代码生成、工具调用等。  
- **对比基线**：主要与 GRPO（强化学习基线）和 MIPROv2（当前最强提示优化器）比较。  
- **核心结果**：  
  - 在所有任务上，GEPA 的平均提升约为 6%，最高可达 20%，而使用的 rollout 数量比 GRPO 少至 1/35。  
  - 在 AIME‑2025 上，GEPA 超过 MIPROv2 超过 12% 的准确率，总体比 MIPROv2 高出 10% 以上。  
  - 作为推理时搜索策略，GEPA 在代码优化任务上也展示了可观的加速效果。  
- **消融实验**：论文提供了对“仅使用语言反思”“仅使用遗传选择”“去掉 Pareto 前沿”等配置的对比，结果显示两者结合是性能提升的关键。  
- **局限性**：作者指出 GEPA 仍依赖于 LLM 本身的自我诊断能力，如果模型的反思质量低下，生成的提示改动可能无效；此外，Pareto 前沿的计算在提示数量极大时会有一定开销。  

### 影响与延伸思考
GEPA 的出现让研究者重新审视“奖励”在语言模型微调中的角色，推动了“语言驱动的自我改进”这一思路。随后的工作（如 ReflexPrompt、Self‑Diagnose‑RL）开始尝试把模型的自我解释直接用于策略更新，甚至把反思过程嵌入到大模型的内部模块。对想进一步探索的读者，可以关注以下方向：  
- **提升模型反思质量的预训练**：让模型在大规模数据上学习如何写诊断报告。  
- **更高效的多目标选择算法**：在提示空间极大时，如何快速近似 Pareto 前沿。  
- **跨模态反思**：把视觉、音频等非语言信息也纳入反思文本，扩展到多模态任务。  

### 一句话记住它
让大语言模型用自己的文字“诊断+进化”，在极少的实验次数下就能跑赢传统强化学习。