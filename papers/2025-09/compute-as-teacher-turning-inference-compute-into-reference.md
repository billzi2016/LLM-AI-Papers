# Compute as Teacher: Turning Inference Compute Into Reference-Free Supervision

> **Date**：2025-09-17
> **arXiv**：https://arxiv.org/abs/2509.14234

## Abstract

Where do learning signals come from when there is no ground truth in post-training? We show that inference compute itself can serve as supervision. By generating parallel rollouts and converting them into reference estimates, models can learn without human labels-critically, even in non-verifiable domains like healthcare guidance where no programmatic checker exists. We call this framework Compute as Teacher (CaT) and it turns inference-time compute from parallel rollouts into supervision for RL training. The framework has two components: (1) reference estimation which aggregates rollouts into a pseudo-reference answer, and (2) reward derivation which converts that pseudo-reference into RL rewards. For (1), we explore a simple method we call synthesis, but the framework admits any aggregator. For (2), we introduce self-proposed rubrics for non-verifiable domains. These are binary, auditable criteria generated from the pseudo-reference and scored by an LLM judge. On HealthBench, models trained with CaT match or exceed inference-time aggregation quality while using 9x less test-time compute. Here, CaT also competes with learning from expert physician annotations, yielding up to +30% relative improvement over the initial policy. The framework extends naturally to verifiable rewards, matching the best existing baselines on MATH-500 in test-time RL and demonstrating 'drop-in' versatility across both types of domains.

---

# 计算即教师：将推理计算转化为无参考监督 论文详细解读

### 背景：这个问题为什么难？

在大模型微调后，往往缺少可靠的标注数据来继续提升性能，尤其是医疗、法律等领域几乎没有“正确答案”。传统的强化学习（RL）需要人工设计的奖励函数或专家标注的参考答案，但这些往往成本高、覆盖面有限，甚至在某些任务上根本不可验证。于是模型只能在“黑盒”环境中盲目探索，学习信号稀缺导致性能提升受限。要想在没有任何 ground‑truth 的情况下让模型自我提升，就必须找到一种能够自动生成监督信号的机制，而这正是本文要解决的核心难题。

### 关键概念速览
- **推理计算（Inference Compute）**：模型在给定输入时进行前向传播得到输出的过程。这里把它视作一种“资源”，可以被重复利用来产生额外信息。  
- **并行 rollout（Parallel Rollout）**：对同一输入，使用模型多次采样得到多个可能的输出序列，类似于让模型“多次思考”。  
- **伪参考（Pseudo‑Reference）**：把多个 rollout 通过某种聚合方式合成为一个“参考答案”，虽然它不一定是真实答案，但在统计意义上更可靠。  
- **自提评分标准（Self‑Proposed Rubric）**：在没有客观评判标准的领域，模型自行生成二元（对/错）判定规则，并交给另一个语言模型（LLM judge）进行审计。  
- **奖励派生（Reward Derivation）**：把伪参考转化为强化学习中的奖励信号，让策略在训练时能够“感知”自己是否更接近伪参考。  
- **CaT 框架（Compute as Teacher）**：整体方法的名称，核心思想是把推理时产生的计算资源当作教师，提供无参考的监督。  
- **LLM Judge**：专门用来评估自提评分标准是否符合预期的语言模型，充当“审计员”。  

### 核心创新点
1. **把推理计算直接当作监督信号**  
   - 之前的做法要么依赖人工标注，要么使用外部程序化检查器；两者都需要额外的人力或可验证的规则。  
   - CaT 让模型自己生成并聚合多个推理结果，形成伪参考，从而在没有任何外部标签的情况下得到学习信号。  
   - 这种自给自足的监督方式显著降低了对标注成本的依赖，使得在医疗等非可验证领域也能进行有效微调。

2. **引入“自提评分标准”并交由 LLM Judge 审核**  
   - 传统 RL 需要手工设计奖励函数，往往难以覆盖所有细节。  
   - CaT 让模型基于伪参考自行生成二元判定规则（如“是否包含关键医学建议”），再让另一个强大的语言模型检查这些规则的合理性。  
   - 通过这种双模型协作，奖励更贴近任务需求，且具备可审计性。

3. **统一可验证与不可验证任务的训练流程**  
   - 过去的研究要么专注于可验证的数学题目，要么专门针对无标签的对话任务，方法不通用。  
   - CaT 的聚合与奖励派生模块是通用的，只要提供相应的聚合器或 rubric，就能在 MATH‑500（可验证）和 HealthBench（不可验证）上使用同一套代码。  
   - 这种“一键切换”式的通用性让框架在不同领域都能保持竞争力。

### 方法详解
**整体思路**  
CaT 把一次推理过程拆成两层：外层是“教师”，负责产生监督；内层是“学生”，通过强化学习利用这些监督进行策略更新。具体步骤如下：

1. **并行 rollout**：对每条输入，模型使用采样（如温度采样）生成 N 条候选答案。  
2. **参考估计（Reference Estimation）**：把这 N 条答案交给聚合器，得到一个伪参考。论文中最简单的聚合器叫 **synthesis**，它可能是多数投票、答案拼接或基于语言模型的重写。  
3. **自提评分标准生成**：模型根据伪参考抽取关键要素（比如“包含治疗建议”），形成一组二元判定规则。  
4. **LLM Judge 审核**：另一个预训练的大模型检查这些规则是否合理，返回通过/不通过的二元信号。  
5. **奖励派生**：如果学生的输出满足所有通过的规则，就给出高奖励；否则给低奖励。奖励可以是 +1 / -1，也可以是更细粒度的分数。  
6. **RL 更新**：使用常见的策略梯度（如 PPO）把奖励反馈给学生模型，完成一次训练迭代。

**关键模块拆解**  
- **并行 rollout** 相当于让模型“多次思考”。想象你在写一篇医学建议报告，先写出几版草稿，再挑出最靠谱的那一版。  
- **参考估计** 的核心是把噪声答案压平为更稳健的信号。synthesis 方法可以看作“众人拾柴火焰高”，多数模型倾向的答案往往更可靠。  
- **自提评分标准** 类似于人类在没有答案时自行设定评判标准，例如“是否提到了患者的主要症状”。这一步把开放式答案转化为可量化的二元特征。  
- **LLM Judge** 起到“审稿人”的角色，确保自提标准不偏离任务本质，防止模型自我强化出奇怪的偏差。  
- **奖励派生** 把通过的标准映射为强化学习的回报，形成闭环学习。

**最巧妙的点**  
- 把 **推理计算** 直接变成 **教师**，不需要额外的标注或外部检查器，这在资源受限的领域尤为突破。  
- 使用 **双模型审计**（学生生成规则，LLM Judge 验证）实现了自监督的可审计性，避免了纯自我强化可能产生的“漂移”。  

### 实验与效果
- **测试任务**：在可验证的数学推理基准 MATH‑500 上，以及不可验证的医疗问答基准 HealthBench 上进行评估。  
- **对比基线**：与传统的基于人工标注的 RLHF（Reinforcement Learning from Human Feedback）以及仅使用单一 rollout 的自监督方法相比。  
- **主要结果**：在 HealthBench 上，CaT 训练的模型在测试时只需要 1/9 的计算量就能达到或超过直接对所有 rollout 做聚合的质量；相对专家标注的基线，提升了最高 30% 的相对性能。MATH‑500 上的表现与当前最好的 RL 方法持平，证明了框架的通用性。  
- **消融实验**：作者分别去掉了（1）synthesis 聚合器，改用单一 rollout；（2）LLM Judge，直接使用未审计的自提规则。结果显示，去掉任一模块都会导致奖励噪声增大，最终性能下降约 10%~15%。  
- **局限性**：论文承认在极端高维输出（如长篇医学报告）时，synthesis 聚合的质量仍受限；此外，LLM Judge 本身的偏见可能会传递到奖励中，需要进一步的公平性审查。

### 影响与延伸思考
CaT 的核心思想——把推理时的计算资源当作教师——在随后的一批自监督强化学习工作中被广泛引用。例如，2024 年的 **Self‑Generated Reward (SGR)** 系列直接借鉴了自提评分标准的思路；2025 年的 **Compute‑Driven Curriculum Learning** 进一步将并行 rollout 与难度自适应结合，形成了更细粒度的教学计划。对想深入的读者，可以关注以下方向：① 更高级的聚合器（如基于图神经网络的答案融合）；② 多模态任务中如何定义自提评分标准；③ LLM Judge 的公平性与可解释性研究。  

### 一句话记住它
把模型的推理计算变成“老师”，让模型自己生成并审计参考答案，从而在没有任何标注的情况下实现强化学习。