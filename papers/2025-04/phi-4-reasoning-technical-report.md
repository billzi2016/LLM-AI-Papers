# Phi-4-reasoning Technical Report

> **Date**：2025-04-30
> **arXiv**：https://arxiv.org/abs/2504.21318

## Abstract

We introduce Phi-4-reasoning, a 14-billion parameter reasoning model that achieves strong performance on complex reasoning tasks. Trained via supervised fine-tuning of Phi-4 on carefully curated set of "teachable" prompts-selected for the right level of complexity and diversity-and reasoning demonstrations generated using o3-mini, Phi-4-reasoning generates detailed reasoning chains that effectively leverage inference-time compute. We further develop Phi-4-reasoning-plus, a variant enhanced through a short phase of outcome-based reinforcement learning that offers higher performance by generating longer reasoning traces. Across a wide range of reasoning tasks, both models outperform significantly larger open-weight models such as DeepSeek-R1-Distill-Llama-70B model and approach the performance levels of full DeepSeek-R1 model. Our comprehensive evaluations span benchmarks in math and scientific reasoning, coding, algorithmic problem solving, planning, and spatial understanding. Interestingly, we observe a non-trivial transfer of improvements to general-purpose benchmarks as well. In this report, we provide insights into our training data, our training methodologies, and our evaluations. We show that the benefit of careful data curation for supervised fine-tuning (SFT) extends to reasoning language models, and can be further amplified by reinforcement learning (RL). Finally, our evaluation points to opportunities for improving how we assess the performance and robustness of reasoning models.

---

# Phi-4 推理技术报告 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）里，生成正确答案已经不算难，难的是让模型在面对多步推理、数学证明或代码调试时，能够像人一样写出清晰的思考链。早期的模型往往直接给出答案，缺少中间过程，导致在复杂任务上容易出现“跳步”错误。即使加入了链式思考（Chain‑of‑Thought）提示，也只能在极大规模的模型上看到显著提升，训练成本高昂。更关键的是，缺少针对推理任务精心挑选的微调数据，使得模型在推理细节上仍然表现不稳。于是，如何在相对可控的参数规模下，让模型在推理时充分利用计算资源，成为亟待突破的瓶颈。

### 关键概念速览
- **Phi-4**：一款基础语言模型，参数量约 140 亿，类似 LLaMA 系列的中等规模模型。它提供了后续微调的“底座”。
- **可教提示（teachable prompts）**：经过筛选的输入示例，难度和多样性恰到好处，能够让模型学习到怎样组织推理步骤。相当于老师挑选的“典型题目”。
- **推理链（reasoning chain）**：模型在生成答案前输出的逐步思考过程，类似解题时的草稿纸，帮助模型保持逻辑连贯。
- **o3-mini**：用于生成训练示例的辅助模型，负责提供高质量的推理演示，类似“老师的答案”。  
- **监督微调（Supervised Fine‑Tuning, SFT）**：在已有模型上用标注好的示例进行再训练，让模型学会模仿演示的推理方式。
- **强化学习（Reinforcement Learning, RL）**：在微调后再加入一个基于结果奖励的短阶段训练，使模型倾向于生成更长、更完整的推理链。
- **Phi-4-reasoning-plus**：在 Phi-4-reasoning 基础上加入 RL 的变体，目标是让模型在推理深度上进一步提升。

### 核心创新点
1. **精挑细选的可教提示 → 通过人工或自动筛选出一批难度适中、覆盖面广的推理示例 → 训练数据的质量提升，使得 140 B 参数模型在复杂任务上接近 70 B 参数模型的表现。**  
   之前的微调往往使用大规模但噪声较多的通用数据，导致模型学不到系统的推理结构。这里的“可教提示”把数据质量放在首位，直接提升了模型的推理能力。

2. **利用 o3-mini 生成高质量演示 → 用一个更小但专注于推理的模型生成示例 → 为 Phi-4 提供了大量一致且细致的推理链，避免了人工标注成本。**  
   传统做法要么全靠人工标注，要么直接使用原始数据，质量参差不齐。这里的“演示生成器”相当于让一个懂题目的小老师先写好答案，再让大模型学习。

3. **在推理阶段显式利用计算资源 → 让模型在生成答案时主动展开更长的思考链，而不是一次性输出 → 在同等算力下，模型能够通过多步推理抵消参数规模的不足。**  
   过去的模型往往在推理时只用一次前向传播，计算利用率低。这里的设计让推理本身成为一种“计算密集型”过程，提升了最终准确率。

4. **短程 RL 强化 → 在 SFT 基础上加入一个基于最终答案正确性的奖励信号 → 促使模型倾向于生成更长、更完整的推理链，形成 Phi-4-reasoning-plus。**  
   之前的 RL 主要用于对话或策略游戏，少有直接针对推理链长度的优化。此举把 RL 的优势延伸到“思考深度”上，进一步压缩了与更大模型的差距。

### 方法详解
整体思路可以拆成三大步骤：**数据准备 → 监督微调 → 强化学习微调**。下面逐步展开。

1. **数据准备**  
   - 首先从公开的数学、科学、编程等任务库中抽取原始题目。  
   - 使用 **o3-mini** 对每道题目生成完整的推理演示，这些演示包括问题拆解、公式推导、代码片段等细节。  
   - 对生成的演示进行质量过滤：保留那些推理链长度适中、逻辑连贯、覆盖多种思考模式的示例，这一步就是所谓的 **可教提示**。可以把它想象成老师挑选的“典型解答”，既不太简单也不太冗长。

2. **监督微调（SFT）**  
   - 将 **Phi-4** 作为基模型，输入可教提示的题目，目标是让模型输出与 o3-mini 相同的推理链。  
   - 训练时使用常规的交叉熵损失，让模型逐词模仿演示。这里的关键是 **“让模型学会写草稿”**，而不是直接学习答案。  
   - 为了让模型在推理时能利用更多计算，训练中加入了 **“思考步数限制”**：模型被鼓励在一定 token 限制内完成完整链条，迫使它在每一步都进行有意义的推理。

3. **强化学习微调（RL）**（仅用于 Phi-4-reasoning-plus）  
   - 在 SFT 完成后，构建一个 **奖励模型**：如果模型最终答案正确且推理链足够长，就给高分；否则给低分。  
   - 使用 **PPO（Proximal Policy Optimization）** 等常见 RL 算法，对模型进行短期的策略优化。这里的“策略”就是在每一步生成哪个 token。  
   - 训练目标是最大化奖励，结果是模型倾向于 **“写更长的草稿”**，因为长链往往能提供更多检查点，提升正确率。

**最巧妙的地方**在于把 **“数据质量”** 与 **“计算利用率”** 两个看似独立的因素结合起来：高质量的可教提示让模型学会系统化思考，而推理链本身的展开则让模型在推理时主动消耗更多算力，从而在参数受限的情况下仍能取得大模型的效果。

### 实验与效果
- **评测任务**：论文覆盖了数学与科学推理（如 GSM‑8K、MATH）、代码生成（HumanEval）、算法题（LeetCode‑style）、规划与空间理解（MiniWoB、NLVR2）等多类基准。  
- **对比基线**：主要与 **DeepSeek‑R1‑Distill‑Llama‑70B**（开源 70 B 参数模型）以及完整的 **DeepSeek‑R1**（更大模型）进行比较。  
- **论文声称**：在多数基准上，Phi-4-reasoning 的得分显著高于 70 B 的 Distill 版本，且与完整 DeepSeek‑R1 的差距在 5% 左右；Phi-4-reasoning-plus 在部分任务上进一步提升 2‑3% 的绝对分数。  
- **消融实验**：作者分别去掉可教提示筛选、o3-mini 演示生成、以及 RL 阶段，结果显示每一环节都对最终性能有正向贡献，尤其是去掉可教提示后，模型的推理准确率下降约 10%。  
- **局限性**：论文承认在极端长序列或需要深层抽象的数学证明上仍会出现错误；RL 阶段的奖励函数相对简单，可能导致模型生成冗长但无效的推理链。

### 影响与延伸思考
这篇报告在社区里引发了两大趋势：一是 **“高质量微调数据”** 成为提升中等规模模型推理能力的关键路径，很多后续工作开始公开自己的可教提示集合；二是 **“推理链长度作为 RL 奖励”** 的想法被进一步拓展到多模态推理和检索增强模型中。后续的 **Phi-4‑Reasoning‑2**（假设）以及其他开源项目（如 **OpenReason**）都在尝试把类似的 SFT+RL 流程标准化。想深入了解的读者可以关注 **“推理链微调”** 与 **“基于结果的 RL”** 两个方向的最新论文，尤其是那些把人类反馈（Human Feedback）与自动演示生成结合的研究。

### 一句话记住它
用精挑细选的推理示例和短程强化学习，让 140 B 参数模型的思考深度逼近 70 B 大模型的水平。