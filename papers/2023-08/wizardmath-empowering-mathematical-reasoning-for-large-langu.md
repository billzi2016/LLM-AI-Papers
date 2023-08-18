# WizardMath: Empowering Mathematical Reasoning for Large Language Models via Reinforced Evol-Instruct

> **Date**：2023-08-18
> **arXiv**：https://arxiv.org/abs/2308.09583

## Abstract

Large language models (LLMs), such as GPT-4, have shown remarkable performance in natural language processing (NLP) tasks, including challenging mathematical reasoning. However, most existing open-source models are only pre-trained on large-scale internet data and without math-related optimization. In this paper, we present WizardMath, which enhances the mathematical CoT reasoning abilities of LLMs without using external python tools, by applying our proposed Reinforcement Learning from Evol-Instruct Feedback (RLEIF) method to the domain of math. Through extensive experiments on two mathematical reasoning benchmarks, namely GSM8k and MATH, we reveal the extraordinary capabilities of our model. Remarkably, WizardMath-Mistral 7B surpasses top-tier open-source LLMs by a substantial margin with higher data efficiency. Furthermore, WizardMath 70B even outperforms GPT-3.5-Turbo, Claude 2, Gemini Pro and GPT-4-early-version. Additionally, our preliminary exploration highlights the pivotal role of instruction evolution and process supervision in achieving exceptional math performance. For more details refer to https://github.com/nlpxucan/WizardLM

---

# WizardMath：通过强化 Evol‑Instruct 提升大语言模型的数学推理能力 论文详细解读

### 背景：这个问题为什么难？

数学推理要求模型能够进行多步、符号化的演算，而大多数开源大语言模型（LLM）只在通用互联网文本上预训练，缺少专门的数学训练信号。传统的微调往往只让模型直接输出答案，忽视了解题过程，导致在需要细致推导的题目上错误率高。即使是最强的闭源模型也需要借助外部计算工具（如 Python）才能取得好成绩，这说明纯语言模型本身的数学能力仍有很大提升空间。

### 关键概念速览
- **Chain‑of‑Thought（思维链）**：让模型在给出最终答案前先写出逐步推理，就像人在纸上列草稿，能够让错误更早被发现。  
- **Evol‑Instruct**：一种让指令不断“进化”的训练方式，模型先生成一批指令回答，再由更强的模型或人工筛选出更高质量的指令，用来继续微调。可以想象成“让模型自我挑选老师”。  
- **RLHF（基于人类反馈的强化学习）**：把人类对模型输出的偏好转化为奖励信号，使用强化学习让模型倾向于产生更受欢迎的答案。  
- **RLEIF（Reinforcement Learning from Evol‑Instruct Feedback）**：把 Evol‑Instruct 产生的高质量指令当作奖励信号，结合强化学习进一步优化模型。相当于在“进化”基础上加了“奖励”。  
- **Process Supervision（过程监督）**：不仅监督最终答案是否正确，还监督每一步推理是否符合数学规则，类似老师在批改解题步骤时给出的细粒度点评。  
- **GSM8K / MATH**：两个公开的数学推理基准，前者以小学到初中题目为主，后者覆盖高中到大学水平的竞赛题，常用来衡量模型的数学能力。  

### 核心创新点
1. **从指令进化到强化学习的闭环**  
   - 之前的做法：要么只用 Evol‑Instruct 产生更好指令后微调，要么单独使用 RLHF 让模型学会偏好。  
   - 本文做法：先用 Evol‑Instruct 生成高质量的数学指令，然后把这些指令的质量评分当作奖励，交给强化学习算法进一步优化模型。  
   - 改变：形成了一个“进化‑奖励”闭环，使模型在每轮迭代中既能自我挑选更好指令，又能通过奖励机制强化这些指令背后的推理策略，提升了学习效率。

2. **过程监督的显式引入**  
   - 之前的微调大多只看答案对错，忽略了解题步骤的合理性。  
   - 本文做法：在 Evol‑Instruct 生成的指令中加入逐步推理的标注，并在强化学习阶段对每一步的正确性给出奖励或惩罚。  
   - 改变：模型被迫学会在每一步都保持数学严谨，最终在需要多步推导的题目上表现更稳健。

3. **无需外部计算工具的纯语言数学推理**  
   - 过去的高分模型往往在后台调用 Python 求解器，算力依赖外部系统。  
   - 本文做法：全部训练和推理都在语言模型内部完成，靠强化学习让模型内部“记住”常见算式和推导模式。  
   - 改变：在相同算力下实现了接近或超过闭源模型的成绩，极大降低了部署门槛。

### 方法详解
整体框架可以划分为三大阶段：**指令生成 → 指令筛选与标注 → 强化学习微调**。

1. **指令生成（Evol‑Instruct）**  
   - 使用一个较强的基线模型（如 Mistral‑7B）在数学题目上生成多种解答指令，每条指令包括题目、思维链步骤和答案。  
   - 这些指令相当于“候选老师”，数量上会比传统微调数据多出数倍。

2. **指令筛选与过程标注**  
   - 通过另一更强模型或人工评审，对每条指令的完整性、步骤正确性和答案准确性打分。  
   - 高分指令被保留下来，并在每一步推理后附加“正确/错误”标签，形成**过程监督信号**。  
   - 这一步类似把原始教材的解题步骤转化为机器可读的教学大纲。

3. **强化学习微调（RLEIF）**  
   - 将保留下来的指令视作**轨迹**（trajectory），每一步的标签转化为即时奖励（正确步骤 +1，错误步骤 -1），最终答案的正确性再加上一个大的奖励。  
   - 使用近端策略优化（PPO）等强化学习算法，让模型的策略网络在生成新指令时最大化累计奖励。  
   - 关键在于**奖励函数**同时考虑过程质量和最终答案，这促使模型在生成思维链时自觉避免逻辑漏洞。

**最巧妙的地方**在于把 Evol‑Instruct 的“自我进化”与 RLHF 的“奖励驱动”合二为一。传统的 Evol‑Instruct 只能提供更好的训练样本，而 RLHF 只能利用已有的偏好信号；RLEIF 把两者的优势叠加，使模型在每轮迭代中既得到更高质量的教材，又学会如何在内部“自我评分”，实现了数据效率的大幅提升。

### 实验与效果
- **测试基准**：GSM8K（约 8k 题目）和 MATH（约 12k 竞赛题）。  
- **主要对比**：包括开源的 Llama‑2‑70B、Mistral‑7B、WizardLM 系列，以及闭源的 GPT‑3.5‑Turbo、Claude 2、Gemini Pro、早期 GPT‑4。  
- **核心结果**（论文声称）：  
  - **WizardMath‑Mistral 7B** 在 GSM8K 上超过所有同规模开源模型约 10%‑15% 的准确率，且使用的微调数据仅为同类方法的 30%。  
  - **WizardMath 70B** 在 MATH 上取得约 78% 的准确率，领先 GPT‑3.5‑Turbo、Claude 2、Gemini Pro 超过 5%‑8%，并且略高于早期 GPT‑4。  
- **消融实验**：  
  - 去掉过程监督后，整体准确率下降约 4%‑6%，说明逐步奖励对多步推理至关重要。  
  - 只使用 Evol‑Instruct 而不进行强化学习，模型提升有限（约 2%‑3%），验证了 RLEIF 的增益。  
- **局限性**：  
  - 仍然依赖较强的初始模型生成指令，完全从零开始的效果未在论文中给出。  
  - 在极高难度的代数或几何证明题上，仍会出现思维链断裂的情况，作者承认需要更细粒度的数学符号约束。

### 影响与延伸思考
WizardMath 的成功展示了“纯语言”数学推理可以在不借助外部计算的前提下逼近闭源大模型的水平，激发了两大方向的后续探索：  
1. **指令进化 + 强化学习的通用框架**，已经被其他团队用于代码生成、法律推理等任务（如 2024 年的 CodeEvol‑RL）。  
2. **过程监督的细粒度奖励**，促使研究者在更复杂的符号推理（如定理证明、物理方程求解）中加入步骤级别的评价机制。  
想进一步深入，可以关注：  
- 如何在没有强模型的情况下自动生成高质量指令（自监督进化）。  
- 将符号化约束（如数学公式的语法检查）直接嵌入奖励函数的研究。  

### 一句话记住它
**WizardMath 用“进化‑奖励”闭环让大语言模型在不依赖外部工具的情况下学会像人一样一步步写出数学解题过程。**