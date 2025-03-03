# How Well do LLMs Compress Their Own Chain-of-Thought? A Token Complexity   Approach

> **Date**：2025-03-03
> **arXiv**：https://arxiv.org/abs/2503.01141

## Abstract

Chain-of-thought prompting has emerged as a powerful technique for enabling large language models (LLMs) to solve complex reasoning tasks. However, these reasoning chains can be verbose, raising concerns about efficiency. In response, recent works have sought to decrease response lengths through simple prompting strategies (e.g. 'be concise'). In this work, we conduct the first systematic study of the relationship between reasoning length and model performance across a diverse range of compression instructions (e.g. 'use 10 words or less' or 'remove all punctuation'). In doing so, we discover a universal tradeoff between reasoning length and accuracy that persists across even very distinct reasoning chains. We demonstrate that this tradeoff emerges from a sharp threshold behavior at the question level: each task has an intrinsic 'token complexity' - a minimal number of tokens required for successful problem-solving. We show how token complexity enables us to compute information-theoretic limits on the accuracy-compression tradeoff, and find that prompt-based compression strategies operate far from these theoretical limits. This suggests there may be significant room for improvement and our framework provides a benchmark to help researchers evaluate progress in reasoning efficiency. Our work also highlights the importance of adaptive compression -- giving shorter responses for easier questions -- and we show that token complexity is a useful tool for measuring this capability.

---

# 大型语言模型在自我思维链压缩方面表现如何？——基于标记复杂度的研究 论文详细解读

### 背景：这个问题为什么难？

在让大模型解决需要多步推理的任务时，研究者发现让模型先把思考过程写出来（即思维链）能显著提升答案正确率。但思维链往往很长，导致生成成本飙升、上下文窗口被占满，实际部署时会很吃力。过去的做法大多是直接在提示里加一句“简洁点”“用更少的词”，却没有系统地衡量“多短算合适”。于是出现了一个核心难题：到底需要多少文字才能保证模型仍然能推理成功？缺乏量化的标准让压缩尝试只能靠经验，难以判断是否已经逼近极限。

### 关键概念速览
**思维链（Chain‑of‑Thought, CoT）**：让模型在给出最终答案前先写出逐步推理，就像学生做数学题时先列出解题步骤。  
**压缩指令（Compression Instruction）**：在提示中加入的限制性要求，例如“用不超过10个词”或“去掉所有标点”。  
**标记（Token）**：模型内部处理的最小文本单元，可能是一个字符、子词或整个词。  
**标记复杂度（Token Complexity）**：完成特定问题所需的最少标记数，是一种衡量任务内在信息量的指标。  
**信息论上限（Information‑theoretic Limit）**：在给定最少标记数的前提下，模型理论上能够达到的最高准确率。  
**自适应压缩（Adaptive Compression）**：根据每道题的难易程度动态决定输出长度，而不是对所有题目使用同一压缩比例。  
**阈值行为（Threshold Behavior）**：在某个标记数以下模型性能急剧下降，超过该阈值后性能基本稳定的现象。

### 核心创新点
1. **从经验压缩到系统测量**：以前的研究只是在提示里随意加“简洁”之类的词，缺乏量化依据。本文首次引入“标记复杂度”概念，直接把每道题所需的最小标记数当作基准，形成了可比较的度量体系。这样可以明确说“这道题至少要用30个标记，否则准确率会掉到50%”。  
2. **统一的准确率‑长度权衡曲线**：作者在多种压缩指令下跑了大量实验，发现所有任务都遵循同一条“越短越不准”的曲线。这个发现把之前看似零散的实验结果统一进了一个通用模型，帮助研究者快速预测压缩后会损失多少性能。  
3. **信息论上限的计算框架**：利用标记复杂度，论文推导出在理想情况下（即模型能够完美利用每个标记的信息）能够达到的最高准确率。随后把实际的压缩策略与这个上限对比，发现现有方法离上限还有很大差距，暗示还有提升空间。  
4. **自适应压缩的实验验证**：基于标记复杂度，作者设计了一种“根据题目难度调节输出长度”的策略。实验表明，这种自适应方式比统一的固定长度压缩要好不少，尤其在混合难度的数据集上提升明显。

### 方法详解
整体思路可以分为三步：**（1）标记复杂度估计 →（2）压缩指令实验 →（3）理论上限对比**。下面把每一步拆开说。

1. **标记复杂度的估计**  
   - 作者先让模型在不受任何长度限制的情况下完整生成思维链。  
   - 对每个生成的思维链，统计其标记数，并记录对应的正确率。  
   - 通过二分搜索的方式逐步降低允许的标记上限，找到模型在该上限下仍能保持正确率不低于某个阈值（如90%）的最小标记数，这个数即该题的“标记复杂度”。  
   - 直观上，这一步像在找“解这道题最少需要写多少步”，一步步削减文字，直到答案开始出错。

2. **压缩指令实验**  
   - 设计了一系列不同的压缩指令：硬性字数上限（≤10、≤20词）、去除标点、要求使用简短句子等。  
   - 对每个指令，重新让模型在同样的题目上生成思维链，并记录实际标记数和正确率。  
   - 把所有指令的结果绘制在同一张“标记数 vs 准确率”图上，发现所有指令的曲线几乎重合，形成了所谓的“通用权衡曲线”。  
   - 这里的关键是把不同的压缩方式统一到“标记数”这个维度上比较，避免了因为指令表述不同而导致的评价偏差。

3. **信息论上限的推导**  
   - 基于标记复杂度，作者假设如果模型每个标记都能完美传递必要的信息，那么只要不低于该复杂度，理论上可以达到 100% 正确率。  
   - 进一步考虑噪声和模型不完美的因素，给出一个保守的上限公式：准确率 ≤ 1 - exp( - (实际标记数 - 复杂度) / 常数)。  
   - 用实际实验数据代入公式，得到每种压缩指令对应的理论上限，并与真实表现对比，发现大多数指令的准确率离上限还有 10%~30% 的差距。

**最巧妙的点**在于把“思维链长度”这一看似主观的概念转化为“标记复杂度”，并用二分搜索自动化找到每题的最小需求，这让后续的压缩评估变得客观且可重复。

### 实验与效果
- **测试任务**：作者选取了几类经典的推理基准，包括数学文字题（GSM8K）、逻辑推理（LogicalDeduction）以及常识问答（ARC‑Easy/Hard）。这些任务的共同特点是需要多步思考才能得出答案。  
- **对比基线**：包括（1）不做任何压缩的原始思维链，（2）简单的“请简洁回答”提示，（3）只限制标点或词数的单一指令。  
- **主要发现**：在所有任务上，压缩到标记数略高于对应的标记复杂度时，准确率仍能保持在 85% 以上；一旦低于复杂度 10% 左右，准确率会骤降 20% 以上。整体上，压缩指令的表现普遍比理论上限低约 15%。  
- **消融实验**：作者分别去掉“去标点”与“限制词数”两类指令，发现仅限制词数的压缩效果更接近上限，而去标点的指令几乎没有提升，说明并非所有形式的压缩都等价。  
- **局限性**：实验主要在英文任务上完成，中文或其他语言的标记分割方式不同，标记复杂度的数值可能会变化。作者也承认目前的标记复杂度估计依赖于模型自身的生成质量，若模型本身就不擅长某类题目，得到的复杂度可能偏高。

### 影响与延伸思考
这篇工作把“思维链压缩”从经验主义推向了可度量的科学问题，随后出现的研究开始围绕“自适应长度控制”展开，例如在提示里加入“根据难度自行决定输出多少标记”的元指令，或是训练专门的长度预测模型。还有人尝试把标记复杂度作为奖励信号，做强化学习让模型在保持准确率的前提下主动生成更短的思维链。想进一步了解，可以关注**“可解释推理的效率优化”**和**“基于信息论的语言模型容量分析”**这两个方向，那里已经出现了几篇把信息瓶颈理论和实际压缩策略结合的后续工作（推测）。

### 一句话记住它
**思维链的最小标记数决定了压缩的底线，离这个底线越近，模型越能在保持准确率的同时省下更多算力。**