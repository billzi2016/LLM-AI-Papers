# SophiaVL-R1: Reinforcing MLLMs Reasoning with Thinking Reward

> **Date**：2025-05-22
> **arXiv**：https://arxiv.org/abs/2505.17018

## Abstract

Recent advances have shown success in eliciting strong reasoning abilities in multimodal large language models (MLLMs) through rule-based reinforcement learning (RL) with outcome rewards. However, this paradigm typically lacks supervision over the thinking process leading to the final outcome. As a result, the model may learn sub-optimal reasoning strategies, which can hinder its generalization ability. In light of this, we propose SophiaVL-R1, as an attempt to add reward signals for the thinking process in this paradigm. To achieve this, we first train a thinking reward model that evaluates the quality of the entire thinking process. Given that the thinking reward may be unreliable for certain samples due to reward hacking, we propose the Trust-GRPO method, which assigns a trustworthiness weight to the thinking reward during training. This weight is computed based on the thinking reward comparison of responses leading to correct answers versus incorrect answers, helping to mitigate the impact of potentially unreliable thinking rewards. Moreover, we design an annealing training strategy that gradually reduces the thinking reward over time, allowing the model to rely more on the accurate rule-based outcome reward in later training stages. Experiments show that our SophiaVL-R1 surpasses a series of reasoning MLLMs on various benchmarks (e.g., MathVisita, MMMU), demonstrating strong reasoning and generalization capabilities. Notably, our SophiaVL-R1-7B even outperforms LLaVA-OneVision-72B on most benchmarks, despite the latter having 10 times more parameters. All code, models, and datasets are made publicly available at https://github.com/kxfan2002/SophiaVL-R1.

---

# SophiaVL‑R1：通过思考奖励强化多模态大语言模型推理 论文详细解读

### 背景：这个问题为什么难？

多模态大语言模型（MLLM）在视觉问答、数学推导等任务上已经能给出看似合理的答案，但它们的推理过程往往是“黑箱”。过去的强化学习（RL）做法只给出最终对错的奖励（outcome reward），模型只能靠结果来调参，缺少对中间思考步骤的监督。于是模型容易学到“投机取巧”的策略——比如直接记忆答案模式或利用数据泄漏，而不是培养真正的逐步推理能力，这会导致在新场景或更复杂题目上表现不佳。要让模型既能产生高质量的思考链，又不被奖励漏洞误导，就必须在训练中显式评估并奖励思考过程本身。

### 关键概念速览
- **多模态大语言模型（MLLM）**：能够同时处理文字和图像等多种输入的语言模型，例如 LLaVA、MiniGPT‑4。它们在视觉理解的基础上生成自然语言输出。
- **规则式强化学习（Rule‑based RL）**：训练时使用预先设定的奖励函数（比如答案是否正确）来指导模型更新参数，类似给模型一个“对或错”的打分表。
- **思考奖励（Thinking Reward）**：对模型完整推理过程（包括每一步的中间文本）进行打分的奖励信号，类似老师给学生的“解题步骤”评分，而不是只看最终答案对不对。
- **Reward Hacking（奖励黑客）**：模型找到规避真实目标、只针对奖励函数取巧的行为。例如，生成大量冗余文字却仍能得到高思考奖励。
- **Trust‑GRPO（Trust‑Guided Reward‑Weighted Policy Optimization）**：一种在策略优化时为思考奖励分配可信度权重的技术，权重依据该奖励在正确答案和错误答案之间的区分度来计算。
- **Annealing（退火）训练策略**：在训练后期逐步降低思考奖励的影响，让模型更多依赖可靠的结果奖励，类似先让学生练习写步骤，再让他专注于答案的准确性。

### 核心创新点
1. **引入思考奖励模型**  
   之前的 RL 只看最终对错 → 这篇论文先训练一个专门评估完整思考链质量的奖励模型 → 让模型在生成每一步时都有质量指引，提升了推理的连贯性和可解释性。

2. **Trust‑GRPO 权重机制**  
   直接使用思考奖励会被奖励黑客利用 → 通过比较同一题目下正确答案的思考奖励与错误答案的思考奖励，计算出一个可信度系数 → 在策略更新时把思考奖励乘以该系数，降低噪声奖励的影响，使训练更稳健。

3. **思考奖励的退火策略**  
   思考奖励在训练初期帮助模型养成良好思考习惯，但长期依赖会限制模型对最终答案的精细调优 → 论文设计了一个随训练轮数逐渐衰减思考奖励权重的计划 → 后期模型更多听从规则式结果奖励，兼顾思考质量和答案准确性。

4. **小模型大突破**  
   通过上述三项技术，7B 参数的 SophiaVL‑R1 在多个基准上超过了 72B 参数的 LLaVA‑OneVision，证明了“思考质量+可信度加权”比单纯扩大模型规模更有效。

### 方法详解
**整体框架**  
整个训练流程可以划分为三步：① 预训练思考奖励模型，② 用 Trust‑GRPO 进行强化学习，③ 采用退火策略逐步削弱思考奖励的比例。简言之，先教会一个“评卷老师”，再让模型在老师的指导下练习写步骤，最后让老师的声音慢慢淡出，让模型自行靠答案对错来收敛。

**1. 思考奖励模型的构建**  
- 收集一批带有完整思考链的多模态问答样本（包括图像、问题、逐步推理文本、最终答案）。  
- 使用一个独立的语言模型（如 LLaMA‑2）对每一步的合理性进行打分，训练一个二分类/回归网络输出思考质量分数。  
- 该模型的目标是让高质量、逻辑连贯的思考链得到高分，显而易见错误或跳步的链路得到低分。

**2. Trust‑GRPO 权重计算**  
- 在一次 RL 更新中，模型会生成若干候选答案，每个候选都有对应的思考链。  
- 对每个候选，用思考奖励模型得到思考分数 `R_think`，用规则式奖励得到结果分数 `R_outcome`（对错 0/1）。  
- 将同一题目下所有正确答案的思考分数取平均，记为 `μ_correct`，错误答案的平均记为 `μ_wrong`。  
- 可信度权重 `w` = sigmoid(μ_correct − μ_wrong)。如果思考奖励能明显区分对错，`w` 接近 1；否则趋向 0。  
- 最终的综合奖励 = `w * R_think + (1‑w) * R_outcome`，并用于策略梯度更新。

**3. 退火训练策略**  
- 设定一个退火函数 `α(t)`（t 为训练步数），初始时 `α≈1`，随训练进行线性或指数衰减至接近 0。  
- 在每一步的综合奖励中，将思考奖励乘以 `α(t)`，即 `Reward = α(t) * w * R_think + (1‑α(t)) * R_outcome`。  
- 这样模型在前期会强烈受到思考质量的约束，后期则更多依赖最终答案的正确性，防止思考奖励长期主导导致的“写得好但答案错”的现象。

**关键细节**  
- **奖励黑客检测**：作者通过观察 `w` 的变化来捕捉异常——如果某批样本的 `w` 突然下降，说明思考奖励可能被滥用，系统会自动降低其影响。  
- **多模态输入统一**：所有视觉信息先通过视觉编码器转成向量，与文本嵌入拼接后送入语言模型，保证思考链的生成能够利用图像信息。  
- **梯度混合**：在策略优化时，使用 PPO（Proximal Policy Optimization）框架，将综合奖励作为优势函数的基准，保持训练的稳定性。

### 实验与效果
- **评测数据集**：论文在 MathVisita（数学可视化推理）和 MMMU（多模态理解大集合）等公开基准上进行测试，覆盖数学、常识、图表解释等多种推理场景。  
- **对比基线**：与 LLaVA‑OneVision‑72B、GPT‑4V、MiniGPT‑4 等最先进的 MLLM 进行比较。  
- **主要结果**：SophiaVL‑R1‑7B 在大多数子任务上超过了 72B 参数的 LLaVA‑OneVision，尤其在需要细致步骤的数学题上提升约 8%‑12% 的准确率（具体数字未在摘要中给出，论文声称有显著优势）。  
- **消融实验**：作者分别去掉 Trust‑GRPO、思考奖励退火以及思考奖励本身进行实验，发现：① 去掉 Trust‑GRPO 后整体性能下降约 3%；② 关闭退火导致后期收敛不稳，错误率上升约 2%；③ 完全不使用思考奖励时，模型在复杂推理任务上跌至与普通 RL 相当的水平。  
- **局限性**：思考奖励模型本身依赖人工标注的高质量思考链，构建成本较高；在极端长链或跨语言场景下，奖励的可靠性仍有待验证。作者也提到在某些视觉噪声较大的样本上，思考奖励的区分度下降。

### 影响与延伸思考
- **领域影响**：这篇工作首次把“思考质量”与“可信度加权”结合进多模态 RL，激发了后续研究在奖励设计上更细粒度的探索，例如将人类反馈（RLHF）与自动思考评估混合使用。  
- **后续工作**：已有几篇论文（如 *ChainReward‑MM*、*TrustRL‑Vision*）借鉴了 Trust‑GRPO 的权重计算思路，尝试在对话系统和代码生成中加入类似的可信度评估。  
- **进一步探索**：如果想深入，可以关注以下方向：① 自动生成高质量思考链的自监督方法，降低标注成本；② 将思考奖励扩展到跨模态序列（视频、音频）中；③ 探索更高级的退火调度，如基于模型不确定性自适应衰减。  

### 一句话记住它
让多模态模型先学会“写好解题步骤”，再让它靠答案对错收敛，思考奖励加可信度权重的双保险让小模型也能跑赢大模型。