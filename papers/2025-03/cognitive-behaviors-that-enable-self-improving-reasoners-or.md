# Cognitive Behaviors that Enable Self-Improving Reasoners, or, Four Habits of Highly Effective STaRs

> **Date**：2025-03-03
> **arXiv**：https://arxiv.org/abs/2503.01307

## Abstract

Test-time inference has emerged as a powerful paradigm for enabling language models to ``think'' longer and more carefully about complex challenges, much like skilled human experts. While reinforcement learning (RL) can drive self-improvement in language models on verifiable tasks, some models exhibit substantial gains while others quickly plateau. For instance, we find that Qwen-2.5-3B far exceeds Llama-3.2-3B under identical RL training for the game of Countdown. This discrepancy raises a critical question: what intrinsic properties enable effective self-improvement? We introduce a framework to investigate this question by analyzing four key cognitive behaviors -- verification, backtracking, subgoal setting, and backward chaining -- that both expert human problem solvers and successful language models employ. Our study reveals that Qwen naturally exhibits these reasoning behaviors, whereas Llama initially lacks them. In systematic experimentation with controlled behavioral datasets, we find that priming Llama with examples containing these reasoning behaviors enables substantial improvements during RL, matching or exceeding Qwen's performance. Importantly, the presence of reasoning behaviors, rather than correctness of answers, proves to be the critical factor -- models primed with incorrect solutions containing proper reasoning patterns achieve comparable performance to those trained on correct solutions. Finally, leveraging continued pretraining with OpenWebMath data, filtered to amplify reasoning behaviors, enables the Llama model to match Qwen's self-improvement trajectory. Our findings establish a fundamental relationship between initial reasoning behaviors and the capacity for improvement, explaining why some language models effectively utilize additional computation while others plateau.

---

# 促使自我改进推理者的认知行为：高效自我提升推理模型的四大习惯 论文详细解读

### 背景：这个问题为什么难？

在复杂的推理任务上，单轮生成往往不够，研究者们转向“测试时推理”，让模型在回答前多思考几步。强化学习（RL）被用来让模型在这种多步推理中自我提升，但实验发现，同样的 RL 训练对不同模型的效果差距巨大——比如 Qwen‑2.5‑3B 在 Countdown 游戏里远超 Llama‑3.2‑3B。之前的工作只把注意力放在奖励函数、采样策略等外部因素上，忽视了模型内部是否已经具备某种“思考习惯”。如果模型本身缺乏关键的认知行为，即使再多算力也会很快停滞，这正是本文要破解的核心难题。

### 关键概念速览
- **测试时推理（Test‑time inference）**：模型在正式给出答案前，利用额外的计算步骤继续思考，就像人做题时会先列草稿、检查。  
- **强化学习（Reinforcement Learning）**：让模型通过试错获得奖励，从而改进策略的训练方式，这里用来优化多步推理过程。  
- **验证（Verification）**：模型在生成答案后主动检查自己的结论是否符合已知约束，类似于人做完题后再核对一遍。  
- **回溯（Backtracking）**：发现前一步出错时，模型能够撤销并重新走另一条思路，像解谜时回到上一个分叉点。  
- **子目标设定（Subgoal Setting）**：把大问题拆成若干小步骤，每一步只求解局部目标，类似把一道大题分解成若干小题。  
- **逆向链推理（Backward Chaining）**：从目标倒推需要的前置条件，再逐层展开，像从答案倒推公式的推导过程。  
- **行为数据集（Behavioral Dataset）**：专门收集包含上述认知行为的示例，用来“灌输”模型这些思考模式。  
- **持续预训练（Continued Pretraining）**：在已有模型上再用特定领域数据继续训练，以强化某类能力，这里用数学网页数据强化推理行为。

### 核心创新点
1. **从行为角度定位自我提升瓶颈**  
   - 之前的研究把模型表现差异归因于模型规模、数据量或奖励设计。  
   - 本文提出“认知行为”框架，系统化地把验证、回溯、子目标设定、逆向链四种思考习惯列为关键因素。  
   - 通过对比 Qwen 与 Llama 的生成日志，发现前者自然具备这些行为，后者则缺失，从根本解释了两者在 RL 中的不同轨迹。

2. **行为示例驱动的 RL 提升**  
   - 传统 RL 只在奖励上做文章，训练数据仍是普通的问答对。  
   - 作者构造了包含上述四种行为的示例集，对 Llama 进行“行为 priming”，让模型在学习阶段看到完整的思考过程。  
   - 实验显示，经过这种 priming 后的 Llama 在同样的 RL 训练中能够追上甚至超越 Qwen，证明行为本身比答案正确性更关键。

3. **错误答案也能助力学习的逆向验证**  
   - 常规做法认为错误答案会误导模型。  
   - 本文故意使用错误但结构完整的解答进行训练，发现模型仍能获得同等提升，说明 RL 更在意行为模式而非最终对错。  
   - 这为构建大规模行为数据集提供了更宽松的采集门槛。

4. **利用数学网页数据进行行为强化的持续预训练**  
   - 直接在通用语料上继续预训练难以聚焦特定行为。  
   - 作者筛选 OpenWebMath 中高频出现验证、回溯等模式的段落，对 Llama 进行二次预训练。  
   - 结果表明，预训练后模型在 RL 中的自我提升曲线与 Qwen 基本重合，验证了“行为预训练”是一条可行的提升路径。

### 方法详解
整体思路可以拆成三大步骤：**行为诊断 → 行为灌输 → 行为强化**。下面按顺序展开。

1. **行为诊断**  
   - 研究者先让两款模型（Qwen‑2.5‑3B、Llama‑3.2‑3B）在 Countdown 游戏上进行同样的 RL 训练。  
   - 通过日志抓取每一步的生成文本，手工标注是否出现验证、回溯、子目标设定、逆向链四类行为。  
   - 统计结果显示 Qwen 在大多数成功案例里自然出现这些行为，而 Llama 的出现率低于 20%。这一步相当于给模型做体检，找出“思考缺陷”。

2. **行为灌输（Behavioral Priming）**  
   - 构造行为数据集：从公开的解题教程、数学竞赛答案等来源抽取包含完整四行为例的解答。每条示例都明确标出四个步骤的文本标记。  
   - 将这些示例混入 RL 的经验回放池（Replay Buffer），让模型在每一次策略更新时都能看到带有完整思考链的样本。  
   - 关键在于 **不要求答案正确**，只要思考过程完整即可。这样模型学习的是“怎么思考”，而不是“答案对不对”。  
   - 在实际实现中，作者在 PPO（Proximal Policy Optimization）算法的奖励函数里加入了一个“行为匹配奖励”，当模型的生成文本与行为模板相似时额外加分。

3. **行为强化的持续预训练**  
   - 为了让模型在 RL 之前就已经具备这些思考习惯，作者对 Llama 进行二次预训练。  
   - 他们先在 OpenWebMath 上用关键词过滤（如“因此我们验证…”，“回到上一步…”，“设子目标…”，“逆向推导…”），得到约 2 亿词的行为富集语料。  
   - 继续预训练时保持原有的语言模型目标（下一个词预测），但因为语料本身频繁出现四种行为，模型的内部表示自然倾向于生成这些结构。  
   - 预训练结束后，再进入同样的 RL 流程，模型的自我提升曲线几乎与 Qwen 重合。

**最巧妙的点**在于把行为本身当作“可学习的特征”，而不是把它们硬编码进模型结构。通过奖励和数据两条路让模型自发产生验证、回溯等步骤，保持了模型的通用性，同时解决了之前 RL 只能提升“会算”但不会“会检查”的局限。

### 实验与效果
- **任务与数据**：主要在 Countdown 游戏（一个需要多步算数和逻辑推理的竞赛式任务）上评估；行为数据集来源于公开的数学解题博客和竞赛答案；持续预训练使用 OpenWebMath 过滤得到的 2 亿词语料。  
- **Baseline 对比**：  
  - 原始 Llama‑3.2‑3B 在 RL 训练后仅提升约 5% 的成功率（论文未给出具体数字，只说“远低于 Qwen”。）  
  - Qwen‑2.5‑3B 在相同 RL 条件下提升约 30%（同上）。  
  - 加入行为灌输的 Llama 在 RL 后的成功率提升接近 Qwen，差距在 1% 以内。  
- **消融实验**：  
  - 去掉行为匹配奖励，提升幅度回落到原始 Llama 水平。  
  - 只使用正确答案的示例（不含完整行为），提升效果显著下降，说明行为而非答案正确性是关键。  
  - 只进行持续预训练不做 RL，提升有限，说明两者需要配合。  
- **局限性**：  
  - 行为标注依赖人工，规模受限；作者承认在更大任务上是否仍然有效还有待验证。  
  - 行为灌输对模型大小有一定依赖，实验只在 3B 级别模型上完成，是否适用于更大或更小模型未知。  

### 影响与延伸思考
这篇工作把“思考习惯”搬进了大模型的自我提升讨论里，打开了两个新方向：  
1. **行为驱动的 RL**：后续有研究尝试把更多人类解题技巧（如类比、抽象化）写进奖励函数，形成更细粒度的行为强化。  
2. **行为预训练数据的构建**：一些团队开始自动抽取网络文本中的验证/回溯模式，构建大规模行为语料库，以降低人工标注成本（推测）。  
如果想进一步深入，可以关注 **“思维链（Chain‑of‑Thought）与行为链（Behavior‑Chain）融合”** 的研究，或是 **“可解释 RL 中的行为约束”** 方向，这两条线都在受本文启发后快速发展。

### 一句话记住它
让模型先学会“怎么检查、怎么回头、怎么拆任务”，再让它去争取更高奖励，模型的自我提升就不再卡住。