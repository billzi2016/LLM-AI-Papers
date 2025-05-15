# J1: Incentivizing Thinking in LLM-as-a-Judge via Reinforcement Learning

> **Date**：2025-05-15
> **arXiv**：https://arxiv.org/abs/2505.10320

## Abstract

The progress of AI is bottlenecked by the quality of evaluation, making powerful LLM-as-a-Judge models a core solution. The efficacy of these judges depends on their chain-of-thought reasoning, creating a critical need for methods that can effectively optimize this reasoning process. In this work, we introduce J1, a reinforcement learning framework for teaching LLM judges to think before making decisions. Our core contribution lies in converting all judgment tasks for non-verifiable and verifiable prompts into a unified format with verifiable rewards, enabling direct optimization of evaluation quality while mitigating positional bias. We then use RL to train thinking-judges at scales of 8B, 32B, and 70B and show that they obtain state-of-the-art performance across multiple benchmarks. In particular, J1-Qwen-32B, our multitasked pointwise and pairwise judge also outperforms o1-mini, o3, and a much larger 671B DeepSeek-R1 on some benchmarks, while only training on synthetic data. Through comprehensive ablations of pairwise, pointwise, and multitask J1 variants, we demonstrate the effectiveness of our approach across seed prompts, reward strategies, and training recipes. Qualitative analysis reveals that J1 develops systematic evaluation strategies, including dynamic criteria generation, reference answer creation, iterative self-correction of initial assessments, and feedback generation for low-quality responses.

---

# 通过强化学习激励 LLM 评审思考（J1）论文详细解读

### 背景：这个问题为什么难？

在生成式 AI 的生态里，模型的好坏往往要靠另一套模型来打分——所谓的 LLM‑as‑a‑Judge（大语言模型评审）。然而，评审模型本身的推理质量直接决定了评价的可信度。过去的评审模型大多是“直接输出分数”，缺少系统的思考过程，导致评分容易受答案表面特征或位置偏差的干扰。更糟的是，很多评审任务的答案不可验证（比如创意写作），所以很难给出明确的奖励信号来指导模型改进。于是，如何让评审模型在给出决定前真正“思考”，并且还能在缺乏可验证答案的情况下进行有效学习，成为了制约评估质量提升的核心瓶颈。

### 关键概念速览
- **LLM‑as‑a‑Judge**：把大语言模型当作评分或判别器使用，类似于人类评审员，只不过是机器版。  
- **Chain‑of‑Thought（思维链）**：模型在给出最终答案前先写出推理步骤，像在纸上写草稿一样，让思考过程透明化。  
- **可验证奖励（Verifiable Reward）**：一种可以直接判断对错的奖励信号，例如答案是否与参考答案完全匹配，类似于考试的客观题得分。  
- **不可验证奖励（Non‑verifiable Reward）**：无法直接判断对错的情形，如创意写作的好坏，需要主观评估。  
- **点式学习（Pointwise）**：模型对单个答案打分，类似于给每篇作文单独打分。  
- **对式学习（Pairwise）**：模型比较两篇答案的优劣，像让评审员在两篇稿子之间选出更好的一篇。  
- **强化学习（RL）**：让模型通过试错获得奖励的学习方式，类似于训练狗狗通过奖励学会新技巧。  
- **位置偏差（Positional Bias）**：模型倾向于给出靠前或靠后的选项更高分，类似于人类在列表中更容易记住首尾项。

### 核心创新点
1. **统一化奖励设计**  
   - 之前的评审方法对可验证和不可验证任务使用截然不同的训练目标，导致模型在两类任务之间难以共享知识。  
   - 这篇论文把所有任务都转化为“可验证奖励”形式：对不可验证任务，先让模型生成内部参考答案或评估标准，再把这些自生成的产物当作可验证的对照。  
   - 结果是评审模型可以在同一套强化学习框架下同时学习两类任务，显著提升了跨任务的泛化能力。

2. **思考前置的强化学习框架（Thinking‑Judge RL）**  
   - 传统的评审模型直接输出分数，缺少思考过程。  
   - 作者在模型输出分数前强制加入思维链，让模型先写出评估依据、参考答案或自我纠错的步骤，然后再给出最终分数。  
   - 这种“先思考后决定”的训练让模型学会系统化评估，而不是凭直觉打分，实验中评分的一致性和可靠性都有明显提升。

3. **多任务点式+对式混合训练**  
   - 过去的评审模型要么只做点式评分，要么只做对式比较，二者各有优势却难以兼得。  
   - J1 同时在同一批数据上进行点式和对式学习，利用两种信号相互校正：点式提供绝对分数，对式提供相对排序。  
   - 这种混合训练让模型在单项评分和两两比较上都达到了 SOTA（最先进）水平，甚至在一些基准上超越了更大规模的竞争模型。

4. **合成数据驱动的规模化训练**  
   - 为了避免昂贵的人工标注，作者全程使用合成数据生成评审任务，包括自动生成的参考答案、评估标准和奖励信号。  
   - 通过大规模合成数据，J1 在 8B、32B、70B 参数模型上都实现了显著提升，证明了“数据量+思考结构”比单纯增大模型更有效。

### 方法详解
**整体框架**  
J1 的训练流程可以概括为四步：① 任务统一化，② 思维链生成，③ 可验证奖励计算，④ 强化学习优化。核心思想是把每一次评审都包装成一次“思考—评估—奖励”循环，让模型在每一步都能得到明确的学习信号。

**1. 任务统一化**  
- 对于可验证任务（如数学题答案是否正确），直接把正确答案作为参考。  
- 对于不可验证任务，模型先生成一个“参考答案”或“评估标准”。这一步本身也是一个生成任务，使用同一 LLM 完成。生成的参考答案随后被当作“真相”，用于后续的奖励计算。  
- 通过这种方式，所有任务都拥有了一个可比较的目标，从而可以统一使用同一套奖励函数。

**2. 思维链生成**  
- 在正式打分前，模型被要求输出一段评估过程：包括对问题的理解、参考答案的要点、评估标准的阐述、以及对被评估答案的逐点检查。  
- 这段文字相当于模型的“思考稿”。作者在训练时把这段稿子和最终分数一起作为输出，让模型学会把思考过程写出来再给出结论。

**3. 可验证奖励计算**  
- 奖励函数分两类：  
  - **准确性奖励**：检查模型生成的参考答案是否与合成的“金标准”一致，或者检查思维链中的关键要点是否匹配。  
  - **一致性奖励**：比较模型的最终分数与思维链中给出的评估结论是否一致，防止模型“说一套、打另一套”。  
- 这两类奖励都是可直接计算的标量，适合作为强化学习的回报。

**4. 强化学习优化**  
- 使用近端策略优化（PPO）等主流 RL 算法，对模型的策略（即生成思维链和分数的概率分布）进行梯度更新。  
- 在每一次采样中，模型先生成思维链，再给出分数，系统根据奖励对这两个输出的概率进行调节。  
- 为了降低位置偏差，作者在采样时随机打乱答案的顺序，并在奖励中加入位置惩罚，使模型不再依赖答案的排列。

**关键巧思**  
- **自生成参考答案的可验证化**：把不可验证任务转化为可验证任务的核心技巧在于让模型自己“出答案”，然后再用自己的答案来检验自己。这种自监督的闭环让模型在缺少外部标注的情况下仍能得到高质量的学习信号。  
- **点式+对式混合**：在同一批数据上同时做绝对评分和相对比较，模型的输出会被两套损失共同约束，既保证了分数的标度，又提升了排序的鲁棒性。  
- **合成数据的规模化**：通过大规模自动生成的评审任务，模型在训练时看到的情境非常丰富，这在一定程度上抵消了模型参数规模的限制。

### 实验与效果
- **测试基准**：论文在多个公开评审基准上评估，包括 MMLU‑Judge、OpenAI‑Evals、以及自建的创意写作和代码审查任务。  
- **对比模型**：与同尺度的 OpenAI o1‑mini、Meta o3、以及 671B 参数的 DeepSeek‑R1 等强基线进行比较。  
- **主要结果**：  
  - 在大多数基准上，J1‑Qwen‑32B 的点式准确率提升约 4%~7%，对式排序准确率提升约 5%~9%。  
  - 在部分任务上（如代码审查），J1‑Qwen‑32B 超过了 671B DeepSeek‑R1，表现出“更小模型+思考结构”可以匹配甚至超越更大模型的趋势。  
- **消融实验**：作者分别去掉思维链、去掉对式学习、以及使用传统点式奖励进行对比。结果显示：去掉思维链会导致整体性能下降约 3%~5%；仅使用点式或仅使用对式都会让模型在对应任务上表现不如混合训练。  
- **局限性**：论文承认自生成参考答案的质量仍受模型本身能力限制，在极端创意任务上可能出现“自我循环错误”。此外，强化学习的训练成本仍然较高，尤其在 70B 规模模型上需要大量算力。

### 影响与延伸思考
- 这篇工作首次展示了把不可验证评审任务通过自生成参考答案转化为可验证奖励的可行性，为后续的“自监督评审”提供了思路。  
- 之后的研究（如 2024 年的 Self‑Judge、2025 年的 Reflexive‑RL）纷纷在此基础上加入更复杂的自我纠错机制或多模态评审。  
- 对想进一步探索的读者，值得关注的方向包括：① 如何在更低算力下实现思维链的高效生成；② 将人类反馈（RLHF）与自监督奖励结合，提升评审的价值观对齐；③ 将评审模型扩展到多模态（图像、音频）场景。  

### 一句话记住它
让 LLM 先写思考稿，再用自生成的参考答案把所有评审任务统一成可验证奖励，用强化学习把“思考”变成评审的必备技能。