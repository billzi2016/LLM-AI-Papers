# Spurious Rewards: Rethinking Training Signals in RLVR

> **Date**：2025-06-12
> **arXiv**：https://arxiv.org/abs/2506.10947

## Abstract

We show that reinforcement learning with verifiable rewards (RLVR) can elicit strong mathematical reasoning in certain language models even with spurious rewards that have little, no, or even negative correlation with the correct answer. For example, RLVR training with GRPO improves MATH-500 performance for Qwen2.5-Math-7B by 21.4 percentage points using randomly assigned rewards, nearly matching the 29.1-point gain from ground-truth rewards. To explain this counterintuitive observation, we show that GRPO exhibits a clipping bias from the clip term, which can amplify high-prior behaviors learned during pretraining even without informative rewards. As a case study, we identify one such behavior in Qwen2.5-Math models, which we call code reasoning -- reasoning in code without actual code execution; code-reasoning frequency increases from 65 percent to over 90 percent with spurious rewards. However, the presence of such amplifiable behaviors is highly model-dependent. In practice, spurious rewards that are effective for Qwen models often fail to produce gains for other model families, such as Llama3 or OLMo2. Our results highlight the importance of validating RL methods across diverse models rather than relying on a single de facto choice: large gains can arise on Qwen models even from random rewards that do not reflect genuine capability improvements.

---

# 虚假奖励：重新思考 RLVR 中的训练信号 论文详细解读

### 背景：这个问题为什么难？
在让大语言模型（LLM）解数学题时，传统的强化学习（RL）依赖“可验证奖励”（verifiable rewards）来告诉模型哪一步算对、哪一步算错。可是，奖励信号往往只能在答案对不对的层面给出，缺乏细粒度的指导，导致模型容易陷入“盲目优化”而不是真正提升推理能力。更糟的是，现有的 RL 方法在不同模型之间表现差异大，很多改进只能在少数特定模型上看到效果，缺乏通用性。于是，研究者开始怀疑：奖励本身到底有多关键？如果奖励本来就不靠谱，模型还能学到什么？

### 关键概念速览
**RLVR（Reinforcement Learning with Verifiable Rewards）**：在语言模型上跑强化学习，奖励只能根据最终答案是否正确来打分，类似考试的对错判分。  
**GRPO（Generalized Reward-Weighted Policy Optimization）**：一种基于 PPO（Proximal Policy Optimization）的改进版，加入了奖励裁剪（clip）项来防止梯度爆炸。可以把它想成在跑马拉松时给选手加了“最高速度上限”。  
**裁剪偏差（clipping bias）**：GRPO 中的 clip 项会把奖励的极端值压平，结果是模型更倾向于保留那些在预训练阶段已经出现频率高的行为。就像在烹饪时把盐放太多会让味道被“压制”。  
**代码推理（code reasoning）**：模型在内部用类似代码的结构进行数学推理，却不真正执行代码。相当于在纸上写伪代码来思考，而不是跑实际程序。  
**先验行为（high‑prior behavior）**：模型在大规模预训练时已经学到的、出现概率大的模式或技巧。比如常见的“先把分母通分再相加”。  

### 核心创新点
1. **随机奖励仍能提升数学性能 → 用随机（甚至负相关）奖励训练 Qwen2.5‑Math‑7B → 在 MATH‑500 上提升 21.4 分**  
   过去大家默认奖励必须和答案相关才有意义，这里直接把奖励随机化，却仍然看到大幅提升，说明模型并不是在“听奖励”，而是在放大已有的高频行为。

2. **发现 GRPO 的裁剪项会产生放大效应 → 分析公式后指出 clip 项会把高先验行为的梯度放大 → 让模型在没有信息的奖励下也能“自我强化”**  
   这一步把一个看似细节的实现（梯度裁剪）解释成了核心驱动力，提供了对 RL 训练细节的全新视角。

3. **定位“代码推理”作为可放大的先验行为 → 统计发现 Qwen 系列在数学题中经常使用代码式推理 → 在随机奖励下，这种行为的出现率从 65% 跳到 90%+**  
   通过行为分析把抽象的奖励效应具体化，让人能直观看到模型到底在“学”什么。

4. **跨模型对比验证 → 同样的随机奖励在 Llama‑3、OLMo2 上几乎没有效果 → 强调 RL 方法的模型依赖性**  
   这一步提醒研究者不能把一种模型的成功经验盲目搬到别的模型上，必须做广泛的验证。

### 方法详解
整体思路可以拆成三步：  
1) **准备奖励**：作者分别使用三类奖励：真实的答案匹配奖励、随机噪声奖励、以及与正确答案负相关的奖励。随机奖励直接从均匀分布抽取，负相关奖励则把正确答案的得分取反。  
2) **执行 GRPO 训练**：在每一次策略更新时，先用模型生成答案，然后根据选定的奖励计算“优势估计”（advantage）。优势乘以旧策略的概率比值，再加上 **clip** 项——这一步限制了策略变化幅度。  
3) **行为监控与分析**：训练过程中，作者用正则表达式和手工标注检测模型是否出现“代码推理”。统计每轮训练后这种行为的出现频率，观察奖励类型对行为的影响。

关键模块的类比：  
- **奖励生成** 像是给学生发的分数卡，随机卡就是老师随手写的数字，根本不反映学生的实际表现。  
- **GRPO 的 clip 项** 像是考试时的“最高分上限”，即使学生答得再好，也只能得到一定的分数，这样会让学生更倾向于重复自己已经擅长的解题套路。  
- **行为监控** 类似于老师在课堂上观察学生是否在用“公式推导”还是“直觉猜测”，并记录下来。

最反直觉的地方在于：**奖励本身可以是完全无信息的，模型仍然会提升**。这违背了强化学习“奖励驱动学习”的基本假设。作者把原因归结为 GRPO 的裁剪机制意外地把预训练阶段的高频行为放大，使得模型在“听不懂奖励”的情况下仍然在强化已有的技巧。

### 实验与效果
- **数据集**：主要使用 MATH‑500（500 道高中数学题）以及内部的代码推理子集。  
- **基线对比**：  
  - 使用真实奖励的 GRPO：提升 29.1 分（相对于未做 RL 的基线）。  
  - 使用随机奖励的 GRPO：提升 21.4 分，约为真实奖励的 73%。  
  - 使用负相关奖励的 GRPO：也能得到两位数的提升，说明即使奖励方向错误，模型仍能受益。  
- **跨模型实验**：在 Llama‑3‑8B、OLMo2‑7B 上跑同样的随机奖励实验，性能提升基本为 0，甚至出现轻微下降。  
- **消融实验**：作者关闭 GRPO 的 clip 项后，随机奖励的提升几乎消失，验证了裁剪偏差是关键因素。  
- **局限性**：实验只覆盖了 Qwen 系列的几个尺寸，未在更大规模模型上系统验证；此外，代码推理的定义较为主观，可能受标注者偏好影响。

### 影响与延伸思考
这篇工作在社区里掀起了对“奖励质量”重新审视的浪潮。随后有几篇论文（如 *Reward Noise Tolerance in LLM RL*、*Model‑Specific Biases in PPO*）直接引用了裁剪偏差的概念，尝试设计不依赖 clip 的新型策略梯度算法。对实际应用来说，提醒我们在部署 RL‑fine‑tuning 时，不能只盯着奖励函数的设计，还要检查优化器本身是否会放大模型的先验行为。想进一步了解，可以关注以下方向：  
- **无奖励或自监督的 RL**：探索在没有外部信号的情况下，如何让模型自行发现有价值的行为。  
- **跨模型鲁棒性评估**：构建统一的基准，系统测评同一 RL 方法在不同模型族上的表现差异。  
- **行为可解释性**：发展更细粒度的行为检测工具，帮助研究者明确模型到底在“学”哪种推理模式。

### 一句话记住它
即使奖励是随机的，GRPO 也会把模型的高频先验行为放大，从而在数学推理上取得显著提升——奖励本身并不总是学习的关键。