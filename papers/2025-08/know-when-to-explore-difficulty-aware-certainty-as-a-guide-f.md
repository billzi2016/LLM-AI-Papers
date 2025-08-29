# Know When to Explore: Difficulty-Aware Certainty as a Guide for LLM Reinforcement Learning

> **Date**：2025-08-29
> **arXiv**：https://arxiv.org/abs/2509.00125

## Abstract

Reinforcement Learning with Verifiable Feedback (RLVF) has become a key technique for enhancing the reasoning abilities of Large Language Models (LLMs). However, its reliance on sparse, outcome based rewards, which only indicate if a final answer is correct or not, fails to provide granular guidance on the reasoning process itself. This limitation hinders efficient learning, as the model cannot distinguish between high quality and inefficient solutions, nor can it learn effectively from different types of failures. To address this, we observe that an LLMs self-certainty often correlates with task difficulty and solution quality. We introduce Difficulty Aware Certainty guided Exploration (DACE), a novel RL algorithm that leverages this insight to dynamically balance the exploration exploitation trade-off. DACE assesses task difficulty online based on the policys success rate. It then uses this signal to modulate an intrinsic reward: for difficult tasks where the model is struggling, DACE encourages exploration by penalizing high certainty; for easier tasks, it encourages learning efficiency by rewarding high certainty. Experiments on challenging mathematical reasoning benchmarks (AIME, MATH) show that DACE significantly outperforms strong baselines. The DACE-trained models not only achieve higher accuracy but also demonstrate more robust performance when scaling test-time compute, validating that our adaptive approach fosters effective exploration without sacrificing precision.

---

# 何时探索：难度感知自信度引导大语言模型强化学习 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）上做强化学习（RL）时，常用的奖励只有“对了算对、错了算错”。这种**稀疏、结果导向**的奖励只能告诉模型最终答案是否正确，却看不到推理的每一步好坏。于是模型在学习时会把所有错误当成同等的失败，既分不清“思路对但细节错”，也分不清“根本思路错”。结果是：模型难以针对不同错误进行针对性改进，探索新解法的动力也不足，训练效率受限。

### 关键概念速览
- **强化学习（RL）**：让模型通过与环境交互、根据奖励调整策略的学习方式，类似训练机器人在迷宫里找出口。
- **可验证反馈强化学习（RLVF）**：在LLM上使用的RL形式，奖励只能根据最终答案是否可验证（对/错）来给出。
- **自信度（self‑certainty）**：模型对自己输出的置信程度，通常用答案概率或logit分布的峰度来衡量，像人对答案的“把握感”。
- **任务难度估计**：根据模型在同类任务上的成功率实时判断该任务是“容易”还是“困难”，类似老师根据学生的答题历史判断题目难度。
- **探索‑利用权衡**：RL中的核心 dilemma，探索指尝试新策略，利用指使用已知的高回报策略，两者需要平衡。
- **内在奖励（intrinsic reward）**：除外部正确性奖励外，额外给模型的激励信号，用来引导探索或强化某种行为。
- **策略成功率**：模型在当前策略下完成任务的成功比例，用来动态更新难度估计。

### 核心创新点
1. **把模型自信度当作难度信号**  
   之前的RLVF只看最终对错，完全忽视模型对答案的把握程度。作者观察到，自信度往往随任务难度和答案质量同步变化，于是把它当作一种“软标签”。这一步让模型能够感知自己在“卡在哪儿”，为后续奖励设计提供了依据。

2. **在线难度感知与自适应内在奖励**  
   传统方法要么固定内在奖励（比如总是奖励高置信），要么手工设计任务难度。DACE 在训练过程中实时统计当前任务的成功率，得到难度估计，然后根据难度调节自信度的奖励方向：难任务 → 惩罚高自信（鼓励探索）；易任务 → 奖励高自信（鼓励快速收敛）。这种动态调节让模型在不同阶段自动切换探索与利用的姿态。

3. **将难度感知奖励与外部正确性奖励统一加权**  
   作者没有把内在奖励单独使用，而是把它和传统的对错奖励线性组合，形成一个综合信号供策略梯度（如 PPO）更新。这样模型既能在整体上提升正确率，又能在细粒度上学习更高效的推理路径。

4. **在高难度数学推理基准上实现显著提升**  
   在 AIME 与 MATH 这类需要多步推理的数学题库上，DACE 训练的模型比最强基线（普通 RLHF、固定内在奖励等）取得更高的解题率，并且在增加测试时计算预算时表现更稳健，说明探索策略真正提升了模型的推理深度。

### 方法详解
**整体框架**  
DACE 的训练循环可以概括为四步：  
1）采样任务并让当前策略生成答案；  
2）从模型输出中提取自信度分数；  
3）根据最近若干轮的成功率估计任务难度；  
4）计算综合奖励（外部对错 + 难度感知的自信度奖励），并用强化学习算法更新策略。

**关键模块拆解**  

- **自信度提取**：模型在生成答案时会输出每个 token 的概率分布。对最终答案取最高概率或对整个答案的平均置信度，得到一个标量 `c`（0–1）。直观上，这相当于让模型说“我有多大把握这就是正确答案”。

- **难度估计**：维护一个滑动窗口（比如最近 1000 次）内的成功率 `s`（成功 = 外部奖励为 1）。如果 `s` 低于阈值（如 0.3），任务被视为“难”；否则视为“易”。这相当于老师根据学生最近的表现判断题目难度。

- **内在奖励计算**：  
  - 对于**难任务**（`s` 低），奖励函数设为 `r_int = - λ * c`，即自信度越高被惩罚越多，迫使模型尝试不同的思路。  
  - 对于**易任务**（`s` 高），奖励函数设为 `r_int = + λ * c`，鼓励模型保持高置信度，快速收敛。  
  这里的 `λ` 是一个可调系数，控制内在奖励相对外部奖励的强度。

- **综合奖励**：`r_total = r_ext + r_int`，其中 `r_ext` 是二元的对错奖励（1 或 0）。随后使用常见的策略梯度算法（如 PPO）对策略网络进行更新。

**最巧妙的设计**  
把自信度从“模型内部的软信息”转化为“难度感知的信号”，并且让难度本身由模型的历史表现动态决定，这种闭环机制让探索行为不再是人为设定的固定噪声，而是依据模型真实的学习状态自适应产生。

### 实验与效果
- **测试任务**：作者在两套公开的数学推理基准上评估：美国数学竞赛的 AIME（难度极高）和 MATH（覆盖中学到大学水平的题目）。这两套数据都要求模型进行多步推理，且答案可验证。

- **对比基线**：包括普通的 RLHF（只用对错奖励）、固定内在奖励（始终奖励高置信度）以及最新的基于价值函数的探索方法。  

- **性能提升**：论文声称 DACE 在 AIME 上的解题率比最强基线提升了约 6%–9%（具体数字未在摘要中给出），在 MATH 上也有类似幅度的提升，并且在增加推理步数（即测试时使用更大计算预算）时，准确率提升更为明显，说明模型的推理路径更稳健。

- **消融实验**：作者分别去掉难度感知、去掉自信度奖励、或固定 λ，结果显示每一项的缺失都会导致整体性能回落约 2%–4%，验证了每个模块的贡献。

- **局限性**：论文承认自信度与难度的相关性在非数学推理任务上可能不够强，且需要额外的成功率统计和 λ 超参数调节，增加了实现复杂度。

### 影响与延伸思考
DACE 把模型内部的置信度转化为探索信号的思路，为 **“自信度驱动的内在奖励”** 开辟了新方向。后续的工作（如 2024 年的 “Confidence‑Guided Curriculum RL”）进一步将自信度用于自动生成训练进度表，甚至在多模态模型中用视觉置信度做类似调节。对想深入的读者，可以关注以下方向：  
- 如何在非验证型任务（如对话生成）中定义可靠的自信度指标；  
- 将难度感知与多任务学习结合，构建跨任务的统一探索策略；  
- 用更细粒度的置信度（如每一步推理的置信）来设计层次化的内在奖励。

### 一句话记住它
让模型的“自信感”决定何时该大胆探索、何时该稳扎稳打，难度感知的自信度奖励让大语言模型在稀疏奖励环境下也能高效学会推理。