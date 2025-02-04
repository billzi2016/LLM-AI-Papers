# Satori: Reinforcement Learning with Chain-of-Action-Thought Enhances LLM Reasoning via Autoregressive Search

> **Date**：2025-02-04
> **arXiv**：https://arxiv.org/abs/2502.02508

## Abstract

Large language models (LLMs) have demonstrated remarkable reasoning capabilities across diverse domains. Recent studies have shown that increasing test-time computation enhances LLMs' reasoning capabilities. This typically involves extensive sampling at inference time guided by an external LLM verifier, resulting in a two-player system. Despite external guidance, the effectiveness of this system demonstrates the potential of a single LLM to tackle complex tasks. Thus, we pose a new research problem: Can we internalize the searching capabilities to fundamentally enhance the reasoning abilities of a single LLM? This work explores an orthogonal direction focusing on post-training LLMs for autoregressive searching (i.e., an extended reasoning process with self-reflection and self-exploration of new strategies). To achieve this, we propose the Chain-of-Action-Thought (COAT) reasoning and a two-stage training paradigm: 1) a small-scale format tuning stage to internalize the COAT reasoning format and 2) a large-scale self-improvement stage leveraging reinforcement learning. Our approach results in Satori, a 7B LLM trained on open-source models and data. Extensive empirical evaluations demonstrate that Satori achieves state-of-the-art performance on mathematical reasoning benchmarks while exhibits strong generalization to out-of-domain tasks. Code, data, and models are fully open-sourced.

---

# Satori：通过链式行动思维的强化学习提升大语言模型推理的自回归搜索 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在一次性输出答案时已经能解决不少任务，但面对需要多步推理的数学或逻辑题，它们常常会在中间一步卡住，导致最终错误。已有的提升手段大多依赖在推理时进行大量采样，然后交给另一个外部模型做“验证”，相当于把单个模型拆成两个人合作。这种两阶段搜索虽然能提升准确率，却把搜索能力外包给了额外的模型，增加了算力开销，也没有真正让原始模型自己学会“思考”。因此，如何在不引入外部验证器的情况下，让单个 LLM 内部拥有类似搜索的自我探索能力，成为了一个亟待突破的难点。

### 关键概念速览
- **Chain-of-Action-Thought（COAT）**：一种让模型在推理时显式标记“继续推理”“自我反思”“探索替代方案”等元动作的格式，就像在解题纸上写下“尝试另一种方法”或“检查上一步是否合理”的提示。
- **自回归搜索**：模型在生成下一个 token 时，不仅考虑已有文本，还把自己过去的搜索轨迹当作上下文，像在玩“走迷宫”时不断回看已经走过的路径再决定下一步。
- **格式调优（format tuning）**：在小规模数据上让模型熟悉 COAT 的写法，相当于教它使用一种新语言的语法规则。
- **自我改进阶段（self‑improvement）**：利用强化学习让模型在大量自生成的任务上尝试不同的行动序列，并根据最终答案的正确性给出奖励，促使它学会挑选更有效的搜索策略。
- **强化学习（RL）**：一种让智能体通过试错获得奖励的训练方式，这里把模型本身当作智能体，奖励函数基于答案对错和搜索效率。
- **数学推理基准**：常用的评测集合，如 GSM8K、MATH 等，专门测量模型在多步计算和逻辑推理上的表现。

### 核心创新点
1. **把搜索动作内化 → COAT 格式 → 单模型即可自行展开多轮搜索**  
   过去的搜索依赖外部 verifier，这篇论文把“继续、反思、探索”等动作写进模型的输出序列，让模型在同一次前向传播中自行决定是否继续推理或换条路。结果是模型不再需要外部伙伴，算力开销大幅下降。

2. **两阶段训练流程 → 小规模格式调优 + 大规模 RL 自我改进 → 兼顾快速适配和深度策略学习**  
   首先在少量标注好的 COAT 示例上让模型学会使用元动作标记，随后在海量自生成任务上用强化学习优化这些动作的选择。这样既避免了大规模标注成本，又让模型在真实任务中磨练搜索技巧。

3. **奖励函数设计 → 正确性 + 搜索成本双目标 → 鼓励高效而非盲目迭代**  
   与仅奖励最终答案正确的传统 RL 不同，Satori 的奖励同时考虑答案是否对以及搜索过程用了多少步骤。模型因此学会在“继续思考”和“直接给答案”之间做权衡，提升了推理速度。

4. **开源 7B 规模模型 → 在算力受限的环境下也能实现 SOTA**  
   通过上述技巧，作者把一个只有 7 B 参数的模型训练到在数学基准上超过更大模型的水平，证明了方法的算力友好性。

### 方法详解
整体思路可以拆成四个环节：  
1) **COAT 格式定义** → 设定几类元动作标签（`<continue>`、`<reflect>`、`<explore>`），并在训练数据中示例它们的使用方式。  
2) **格式调优** → 用几千条人工编写的 COAT 示例，对基座模型进行轻量微调，使其学会在生成答案前先输出这些标签。  
3) **自我生成任务** → 让已经掌握 COAT 的模型自行生成大量推理题目（比如随机算式、逻辑谜题），并在每一步记录下它选择的元动作。  
4) **强化学习优化** → 采用近端策略优化（PPO）等 RL 算法，对模型的策略网络进行更新。奖励函数由两部分组成：①答案正确给正奖励，错误给负奖励；②每使用一次 `<continue>` 或 `<explore>` 都扣除少量分数，以惩罚无效的长链。

**关键模块细化**  
- **元动作解码器**：在每一步生成时，模型先输出一个元动作标记，再紧跟对应的思考文本。可以把它想象成“先说我要干什么，再说怎么干”。  
- **搜索轨迹缓存**：所有已生成的动作-思考对会被存入上下文缓存，后续步骤的输入会把这些缓存拼接进去，形成自回归的搜索链。  
- **奖励评估器**：在每轮任务结束后，系统会运行一个答案校验器（如数学求值器），判断最终答案是否正确，并统计使用的元动作次数，进而计算总奖励。  
- **策略更新**：使用 PPO 时，旧策略的输出概率与新策略的输出概率比值会被用于裁剪，防止一次更新改变太大，保持训练的稳定性。

**最巧妙的设计**  
- 把“是否继续思考”这种元决策直接写进语言模型的输出空间，而不是在外部控制器里做二次判断。这样模型的所有决策都在同一条自回归链上，既保持了语言模型的统一性，又让搜索过程可解释。  
- 奖励函数同时考虑正确性和搜索成本，使模型学会“省时省力”，避免了传统 RL 中常见的“无限循环搜索”现象。

### 实验与效果
- **评测任务**：主要在 GSM8K、MATH、MathQA 等数学推理基准上测试，还包括一些逻辑推理和常识问答的跨域任务，以验证通用性。  
- **对比基线**：与传统 CoT（思维链）模型、Self‑Consistency（自洽采样）以及外部 verifier 系统（如 ReAct+Verifier）进行比较。  
- **成绩表现**：论文声称 Satori 在 GSM8K 上的准确率超过了同等规模的基线约 5%~7%，在 MATH 上也实现了领先的分数，且在跨域任务上保持了 2%~3% 的提升。  
- **消融实验**：去掉 `<reflect>` 或 `<explore>` 标签会导致整体准确率下降约 2%，仅保留格式调优而不做 RL 自我改进则提升幅度几乎消失，说明两阶段训练缺一不可。  
- **局限性**：作者指出在极端长链任务（超过 30 步）时搜索成本仍然偏高，奖励函数对搜索深度的惩罚可能导致过早收敛；此外，当前实现仍依赖于人工编写的少量 COAT 示例，完全自动化的格式学习仍是开放问题。

### 影响与延伸思考
Satori 把“搜索”从外部工具搬进了语言模型内部，开启了“自我搜索型 LLM”的新方向。后续有几篇工作尝试在更大模型上复现 COAT 思路，或把元动作扩展到代码生成、规划等更复杂的场景（如 “Action‑Chain” 系列）。如果想进一步探索，可以关注以下方向：① 自动发现并学习新的元动作标签；② 将奖励函数与人类偏好对齐的 RLHF（基于人类反馈的强化学习）结合；③ 在多模态模型中引入类似的自回归搜索机制。整体来看，Satori 为在算力受限的环境下提升 LLM 推理能力提供了可复制的范式。

### 一句话记住它
让大语言模型自己在答案前写“继续/反思/探索”，用强化学习把这些自我搜索动作内化，从而在单模型内部实现高效的多步推理。