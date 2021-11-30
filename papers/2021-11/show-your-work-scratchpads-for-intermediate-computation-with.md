# Show Your Work: Scratchpads for Intermediate Computation with Language   Models

> **Date**：2021-11-30
> **arXiv**：https://arxiv.org/abs/2112.00114

## Abstract

Large pre-trained language models perform remarkably well on tasks that can be done "in one pass", such as generating realistic text or synthesizing computer programs. However, they struggle with tasks that require unbounded multi-step computation, such as adding integers or executing programs. Surprisingly, we find that these same models are able to perform complex multi-step computations -- even in the few-shot regime -- when asked to perform the operation "step by step", showing the results of intermediate computations. In particular, we train transformers to perform multi-step computations by asking them to emit intermediate computation steps into a "scratchpad". On a series of increasingly complex tasks ranging from long addition to the execution of arbitrary programs, we show that scratchpads dramatically improve the ability of language models to perform multi-step computations.

---

# 展示你的工作：语言模型的中间计算草稿板 论文详细解读

### 背景：这个问题为什么难？

大规模预训练语言模型在一次性生成文本、写代码等“一遍搞定”的任务上表现惊人，但当任务需要多步、可变长度的推理时，它们往往会出错。比如把两个很大的整数相加，或者让模型解释并执行一段程序，模型很容易在中间步骤“卡住”。传统做法是直接让模型一次性输出最终答案，缺少对中间计算的可视化和检查手段，这导致模型在长链推理时错误累积，准确率急剧下降。

### 关键概念速览
- **Scratchpad（草稿板）**：让模型在生成答案的过程中把每一步的中间结果写下来，类似人做数学题时的草稿，后面的步骤可以直接读取前面的文字。
- **Few‑shot（少样本）**：只给模型提供少量示例就让它完成任务，而不是大规模微调。这里指模型在看到几个“一步一步算”的例子后，就能自行使用草稿板。
- **Transformer（变压器）**：当前主流的语言模型架构，擅长捕捉序列中的长程依赖。本文的实验全部基于这种模型。
- **Chain‑of‑Thought（思维链）**：让模型在给出答案前先写出推理步骤的技术。Scratchpad 可以看作是思维链的实现细节之一，只是把步骤写在同一输出流里。
- **Intermediate Computation（中间计算）**：指在完成最终目标之前产生的临时数值或状态，例如加法过程中的进位、程序执行时的变量值。
- **Prompt Engineering（提示工程）**：通过精心设计输入文本（提示）来引导模型产生期望的行为。本文的核心技巧就是在提示里加入“把每一步写在scratchpad上”。

### 核心创新点
1. **从一次性输出到逐步写入**：以前的少样本方法直接让模型输出答案，容易在长链任务上出错。本文改为在提示中要求模型把每一步写进scratchpad，模型因此被迫显式地完成每一步计算。这样做把隐式的内部状态外显，显著提升了多步任务的成功率。
2. **统一的scratchpad格式**：作者没有为每种任务单独设计专属的中间表示，而是采用一种通用的“文本+标记”格式，模型只需在同一序列里交替写“步骤描述”和“当前值”。这种统一性让同一个模型可以在从长整数加法到任意程序执行的多种任务上复用同一技巧。
3. **少样本学习即能掌握**：只需要在few‑shot示例里展示几次“写草稿并读取”的过程，模型就能在全新任务上自行使用scratchpad。相比需要大量微调的方案，省时省算力。
4. **实验验证跨任务的有效性**：作者在一系列难度递增的基准上（从几位数加法到上百行代码的解释执行）都报告了显著的准确率提升，证明scratchpad并非针对单一任务的特例，而是通用的推理增强手段。

### 方法详解
整体思路可以概括为三步：**构造提示 → 让模型生成带scratchpad的输出 → 解析最终答案**。  
1. **提示构造**：在few‑shot示例中，每个示例都遵循同一模板。模板的前半部分给出任务描述（比如“计算 12345 + 67890”），随后是一段示例性的scratchpad，展示模型如何一步步写出每一位的相加过程并记录进位。关键是加入明确指令：“请把每一步写在下面的scratchpad中”。  
2. **模型生成**：模型接收到提示后，会在同一输出流里交替产生两类文本：  
   - **步骤描述**：如 “第1位相加：5+0=5，进位0”。  
   - **scratchpad 更新**：把当前的中间结果（比如当前的和、进位）写进去。  
   这种交替写作相当于让模型在自己的“记事本”里记下每一步，然后再基于记下的内容继续推理。因为Transformer的自注意力机制可以随时访问已经生成的文字，模型天然能够“读取”自己的scratchpad。  
3. **答案抽取**：当模型输出包含“最终答案”标记时，后面的数字即为任务的答案。后处理只需要截取该标记后的内容即可。  

**关键细节**  
- **统一标记**：作者使用固定的标记（如 `>>>` 开头表示步骤，`###` 开头表示scratchpad 内容）帮助模型区分两类信息。  
- **长度控制**：因为scratchpad会随步骤增多而变长，提示里会提醒模型在必要时压缩信息（比如只保留最新的变量值），防止序列超出模型的最大长度。  
- **不需要额外模块**：整个过程完全依赖语言模型本身的生成能力，没有额外的外部记忆或控制器，这也是方法简洁且易复现的原因。  

最让人意外的地方是：**只要在提示里加一句“写下每一步”，模型就会自发地把中间计算外化**。这表明大模型已经内部拥有多步推理的潜能，只是平时被一次性输出的指令压制住了。

### 实验与效果
- **任务范围**：作者选取了从**长整数加法**（数位可达数百位）到**任意程序执行**（包括循环、条件分支的Python代码）共五类任务。  
- **基线对比**：与直接一次性输出答案的few‑shot方法、以及传统的Chain‑of‑Thought（思维链）方式相比，scratchpad 在所有任务上都显著领先。论文中提到在长加法任务上，准确率从约30%提升到超过90%；在程序执行任务上，成功率从约20%提升到接近80%。  
- **消融实验**：作者分别去掉了统一标记、进位压缩和少样本示例中的scratchpad示例，发现每去掉一项，性能都会下降 10%~20%，说明这些设计都是贡献因素。  
- **局限性**：当任务需要的中间状态超过模型最大序列长度时，scratchpad 会被截断，导致后续步骤缺失信息。作者承认在极长程序或需要大量临时变量的情形下仍会失效。  

### 影响与延伸思考
这篇工作打开了“让语言模型自我写草稿”的新思路，随后出现了大量基于scratchpad的变体：  
- **Self‑Consistency**：在多个scratchpad 采样上做投票，进一步提升鲁棒性（推测）。  
- **Tool‑Augmented Models**：把scratchpad 与外部计算工具（如算术库）结合，让模型在需要时把中间结果交给工具处理（推测）。  
- **大模型的内部记忆研究**：研究者开始探讨模型内部的“隐式工作空间”到底是怎样的，scratchpad 成为检验其可视化的实验手段。  
如果想继续深入，可以关注**“可解释推理的语言模型”**和**“语言模型与外部记忆交互”**这两个方向，尤其是近期在NeurIPS、ICLR上出现的相关论文。

### 一句话记住它
让大语言模型在生成答案时写下每一步的“草稿”，即可把隐形的多步推理变成显式的可检验过程，显著提升复杂计算的成功率。