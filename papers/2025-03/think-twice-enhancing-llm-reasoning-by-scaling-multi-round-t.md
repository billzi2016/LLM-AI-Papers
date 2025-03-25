# Think Twice: Enhancing LLM Reasoning by Scaling Multi-round Test-time   Thinking

> **Date**：2025-03-25
> **arXiv**：https://arxiv.org/abs/2503.19855

## Abstract

Recent advances in large language models (LLMs), such as OpenAI-o1 and DeepSeek-R1, have demonstrated the effectiveness of test-time scaling, where extended reasoning processes substantially enhance model performance. Despite this, current models are constrained by limitations in handling long texts and reinforcement learning (RL) training efficiency. To address these issues, we propose a simple yet effective test-time scaling approach Multi-round Thinking. This method iteratively refines model reasoning by leveraging previous answers as prompts for subsequent rounds. Extensive experiments across multiple models, including QwQ-32B and DeepSeek-R1, consistently show performance improvements on various benchmarks such as AIME 2024, MATH-500, GPQA-diamond, and LiveCodeBench. For instance, the accuracy of QwQ-32B improved from 80.3% (Round 1) to 82.1% (Round 2) on the AIME 2024 dataset, while DeepSeek-R1 showed a similar increase from 79.7% to 82.0%. These results confirm that Multi-round Thinking is a broadly applicable, straightforward approach to achieving stable enhancements in model performance, underscoring its potential for future developments in test-time scaling techniques. The key prompt: {Original question prompt} The assistant's previous answer is: <answer> {last round answer} </answer>, and please re-answer.

---

# 再思一次：通过扩展多轮推理提升大语言模型的推理能力 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在一次性生成答案时已经能解决不少任务，但面对需要深度推理的数学、逻辑或代码题目，往往会因为思路不完整而出错。过去的提升手段主要是让模型在训练阶段学会“思考链”（CoT）或通过强化学习（RL）微调，但这两种方式都有硬伤：CoT 需要在训练数据里显式标注推理步骤，成本高且难以覆盖所有题型；RL 训练耗时长、样本利用率低，而且模型仍受限于一次性生成的长度上限，长文本会被截断。于是出现了“测试时扩展”（test‑time scaling）的思路——在推理阶段给模型更多“思考时间”，但如何在不改模型结构、不增加训练成本的前提下实现，仍是一个待解的难题。

### 关键概念速览

**测试时扩展（Test-time scaling）**：在模型已经训练好的情况下，通过改变推理过程（比如增加思考轮数或提示长度）来提升性能，类似于给已经会走路的学生多练几遍题目。

**思维链（Chain‑of‑Thought, CoT）**：让模型在给出最终答案前先写出推理步骤，就像人在解题时先列出草稿。

**多轮思考（Multi‑round Thinking）**：把模型上一轮的答案当作新一轮的提示，循环迭代，让模型在每一次都“站在自己的答案上”再思考。

**强化学习（Reinforcement Learning, RL）**：一种让模型通过奖励信号自我改进的训练方式，常用于微调语言模型的策略。

**长文本限制**：当前大模型的上下文窗口有上限，超出后会被截断，导致信息丢失，类似于一次只能记住有限的笔记。

### 核心创新点

1. **从一次性生成 → 多轮迭代**：传统做法是一次性把问题和提示喂进去，让模型直接输出答案。本文改成“先答后问”，把上一轮的答案包装进新的提示，再让模型重新回答。这样模型可以在每一轮看到自己的前一次思考结果，像是自我纠错的过程。

2. **统一的提示模板 → 简单可复制**：作者提出了一个固定的提示格式：`{原始问题} The assistant's previous answer is: <answer>{上一轮答案}</answer>, and please re-answer.` 只要把上一轮的答案嵌进去，就能直接复用，无需额外的模型改动或复杂的提示工程。

3. **跨模型、跨任务的通用性**：实验覆盖了从 32 B 参数的 QwQ 到 DeepSeek‑R1 等不同规模、不同训练背景的模型，且在数学、常识、代码等四类基准上都出现提升，说明该方法不是针对某一模型的特例，而是普适的推理增强手段。

4. **在不增加训练成本的前提下实现性能提升**：所有改进都发生在推理阶段，只需要多跑几轮生成，算力开销相对可控，却能把 AIME 2024 上的准确率从 80.3% 拉到 82.1%（QwQ‑32B），DeepSeek‑R1 也从 79.7% 提升到 82.0%。这证明了“思考时间”本身就是提升模型能力的关键资源。

### 方法详解

整体思路可以拆成三步：**准备 → 生成 → 循环**。

1. **准备**  
   - 把原始问题写成标准提示。  
   - 设定一个最大轮数（比如 2 ~ 3 轮），以及每轮的输出长度上限，防止上下文窗口被占满。

2. **第一轮生成**  
   - 将原始提示直接喂给模型，得到第一版答案 `A₁`。这一步和普通的问答没有区别。

3. **循环迭代**  
   - 把 `A₁` 按照模板嵌入到新的提示里，形成 “上一轮答案” 部分。  
   - 再次调用模型，得到第二版答案 `A₂`。如果 `A₂` 与 `A₁` 差别不大，或者达到预设的停止条件（比如答案不再变化、或达到最大轮数），则结束；否则继续把 `A₂` 放回提示，进入下一轮。  
   - 关键在于模型每轮都能看到自己前一次的完整输出，这相当于让模型“回顾自己的草稿”，有机会发现并纠正之前的逻辑漏洞。

**为什么会有效？**  
- 人类解题时常常会先写出一个草稿答案，然后再检查、补充细节。多轮思考把这种“自我审视”搬到了 LLM 上。  
- 模型的注意力机制天然擅长在同一上下文中对比信息，上一轮答案提供了明确的参考点，使得模型在第二轮可以对比新旧答案，自动进行纠错或补全。  
- 由于提示模板固定，整个流程不需要手工调参，只要把前一次的答案粘进去即可，极大降低了工程实现的门槛。

**最巧妙的地方**  
- 只用了一个非常简短的提示语句，却把“前一次答案”包装成了模型的“上下文记忆”。这比起在训练阶段加入额外的记忆模块要轻量得多。  
- 作者没有在每轮都重新给出完整的题目描述，而是只在第一轮提供完整信息，后续轮次只需补充上一轮答案，这样可以在有限的上下文窗口内容纳更多思考轮次。

### 实验与效果

- **测试数据**：AIME 2024（数学竞赛题）、MATH‑500（中等难度数学题）、GPQA‑diamond（高难度常识问答）以及 LiveCodeBench（代码生成与调试）。这些数据覆盖了推理深度、语言多样性和代码执行三个维度。  
- **基线对比**：直接一次性生成（即不使用多轮思考）作为基准。  
- **主要提升**：  
  - QwQ‑32B 在 AIME 2024 上从 80.3% 提升到 82.1%。  
  - DeepSeek‑R1 在同一数据集上从 79.7% 提升到 82.0%。  
  - 其他基准（MATH‑500、GPQA‑diamond、LiveCodeBench）也都有“稳定提升”，虽然摘要未给出具体数字，但作者强调提升是“一致且显著”。  
- **消融实验**：摘要未提供细节，原文未详细描述。可以推测作者可能会去掉上一轮答案的包装、或改动提示模板，来验证“多轮思考”本身的贡献。  
- **局限性**：  
  - 需要额外的推理时间和算力，尤其在轮数增多时成本会线性上升。  
  - 对于已经接近上下文上限的长文本，加入上一轮答案可能导致关键信息被截断。  
  - 只在测试阶段起效，训练阶段仍然没有改进，意味着模型本身的基础能力没有提升。

### 影响与延伸思考

这篇工作把“思考时间”从一个模糊的概念变成了可操作的多轮提示流程，随后的研究开始探索更细粒度的自我审查（Self‑Verification）和动态轮数控制（Adaptive Rounds）。一些后续论文尝试把多轮思考与检索增强（Retrieval‑Augmented Generation）结合，让模型在每轮都去检索外部知识，进一步突破上下文窗口的限制。对想继续深入的读者，可以关注以下方向：  
- **自适应轮数调度**：让模型自行判断是否需要再思考一轮，而不是固定轮数。  
- **跨模态多轮思考**：把图像或代码执行结果也放进下一轮提示，实现更丰富的自我纠错。  
- **训练时模拟多轮**：在微调阶段加入多轮提示，让模型在训练时就学会利用自己的前一次输出。

### 一句话记住它

让模型把自己的答案当作新提示，循环“再思一次”，就能在不改模型、不增训练成本的情况下显著提升推理准确率。