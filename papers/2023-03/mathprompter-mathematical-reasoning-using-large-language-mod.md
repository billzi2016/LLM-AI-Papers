# MathPrompter: Mathematical Reasoning using Large Language Models

> **Date**：2023-03-04
> **arXiv**：https://arxiv.org/abs/2303.05398

## Abstract

Large Language Models (LLMs) have limited performance when solving arithmetic reasoning tasks and often provide incorrect answers. Unlike natural language understanding, math problems typically have a single correct answer, making the task of generating accurate solutions more challenging for LLMs. To the best of our knowledge, we are not aware of any LLMs that indicate their level of confidence in their responses which fuels a trust deficit in these models impeding their adoption. To address this deficiency, we propose `MathPrompter', a technique that improves performance of LLMs on arithmetic problems along with increased reliance in the predictions. MathPrompter uses the Zero-shot chain-of-thought prompting technique to generate multiple Algebraic expressions or Python functions to solve the same math problem in different ways and thereby raise the confidence level in the output results. This is in contrast to other prompt based CoT methods, where there is no check on the validity of the intermediate steps followed. Our technique improves over state-of-the-art on the MultiArith dataset ($78.7\%\rightarrow92.5\%$) evaluated using 175B parameter GPT-based LLM.

---

# MathPrompter：利用大语言模型进行数学推理 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在自然语言理解上已经相当强大，但在算术推理上仍常常出错。数学题目与日常语言不同，它们几乎只有唯一正确答案，模型的任何小偏差都会导致答案全错。过去的做法大多让模型一次性给出答案或一次性写出思考链（CoT），但缺乏对中间步骤的检查，也没有办法让模型表达“我有多自信”。于是用户在关键场景（如教育、金融）里难以信任模型的输出，这直接限制了 LLM 在数学推理领域的落地。

### 关键概念速览
**大语言模型（LLM）**：能够生成自然语言的深度模型，像 GPT‑4 那样拥有上百亿参数，能够“写作文”“写代码”。  
**算术推理**：指需要进行数值运算、代数变形或逻辑推导才能得到唯一答案的数学任务。  
**Chain‑of‑Thought（思维链）**：让模型在给出最终答案前先把推理步骤写出来，类似人做题时的草稿。  
**Zero‑shot CoT**：不需要额外示例，直接在提示中加入“请一步步思考”，模型自行生成思考链。  
**Confidence Estimation（置信度估计）**：模型对自己答案的可靠程度给出数值或文字描述，帮助使用者判断是否可信。  
**MultiArith 数据集**：常用的算术推理基准，包含多步文字题，答案唯一。  
**Prompt Engineering（提示工程）**：通过精心设计输入文本，引导模型产生期望的输出方式。

### 核心创新点
1. **单一思维链 → 多样化求解链**：传统 CoT 只让模型生成一条解题路径，若这条路径出错，答案必错。MathPrompter 在同一次提示下让模型生成 **多个** 代数表达式或 Python 函数，每个都独立求解同一道题。这样相当于让模型“多次做同一题”，提升了错误被相互抵消的概率。  
2. **零样本多解生成 → 自动置信度**：通过在提示中加入“请给出不同的求解方式”，模型会自行产生多套解法。随后系统对这些解法的结果进行一致性检查：若多数解法得到相同数值，则置信度高；若出现分歧，则标记为低置信度。这样解决了以往模型“不给出置信度”的痛点。  
3. **代码化中间步骤 → 可执行校验**：除了文字式的代数式，MathPrompter 还能让模型输出完整的 Python 函数并实际运行。运行结果若与代数式求值相符，则进一步确认答案的正确性。代码执行提供了一个客观的“真相检验”。  
4. **统一框架提升基准**：在 175B 参数的 GPT 系列上，使用上述多解+校验流程，MultiArith 的准确率从原来的 78.7% 提升到 92.5%，显著超过当时的最先进水平。

### 方法详解
**整体思路**  
MathPrompter 把一次算术题的求解拆成三步：① 通过 Zero‑shot CoT 让模型一次性生成若干不同的求解表达；② 对每个表达进行求值（代数求值或代码执行）；③ 汇总所有结果，统计一致性并给出置信度标签。整个流程只需要一次 API 调用，所有子任务都在同一个提示里完成。

**步骤拆解**  

1. **提示构造**  
   - 基础提示：`“请一步步思考并给出至少三种不同的求解方式，分别用代数式和 Python 函数表示。”`  
   - 关键在于明确要求“不同的方式”，并且把代数式和代码都列入选项。模型在零样本条件下会自行产生多条思维链，每条链的末尾附带一个可直接求值的表达式或函数。

2. **多解生成**  
   - **代数式**：模型把题目转化为符号表达式，如 `x = (12 + 7) * 3`。  
   - **Python 函数**：模型写出完整的函数 `def solve(): return (12 + 7) * 3`。  
   - 由于提示要求“至少三种”，模型往往会给出 3–5 条不同的等价或略有不同的写法（例如先合并括号后乘法，或先乘后加），这相当于在同一题上做了多次“独立实验”。

3. **求值与执行**  
   - 对每条代数式，系统使用安全的求值器（如 `sympy`）计算数值。  
   - 对每条 Python 函数，系统在沙箱环境中运行，捕获返回值。  
   - 这一步的核心是 **可执行校验**：代码运行的结果如果与代数式求值不一致，说明模型在生成某条链时出现了逻辑错误。

4. **一致性聚合**  
   - 收集所有数值结果，统计出现频率最高的数值。  
   - 设定阈值（如 ≥ 80% 的解法给出相同答案）则标记为 **高置信度**；否则标记为 **低置信度**。  
   - 同时返回所有解法的文本，供用户审阅。

**最巧妙的地方**  
- **零样本多解**：不需要额外的示例或微调，只靠提示就能让模型自发产生多条独立解法，这大幅降低了实现成本。  
- **代码执行校验**：把模型的文字推理转化为可运行代码，让机器本身再一次“检查”自己的答案，这在纯文字 CoT 中是前所未有的。

### 实验与效果
- **数据集**：主要在 MultiArith 上评估，这是一个包含 2,695 条多步算术题的公开基准。  
- **基线对比**：原始的 Zero‑shot CoT 在同样的 175B GPT 上的准确率约为 78.7%。MathPrompter 将其提升至 92.5%，提升幅度超过 13%。  
- **消融实验**：论文中分别去掉（1）多解生成，只保留单一解法；（2）代码执行校验，只保留代数求值；（3）置信度聚合。结果显示，去掉任意一环都会导致准确率回落到 80% 左右，说明每个模块都对最终提升至关重要。  
- **局限性**：实验仅在 175B 参数的 GPT 系列上完成，未验证在更小模型或开源模型上的效果；此外，代码执行依赖安全沙箱，实际部署时会增加系统复杂度。原文未详细讨论对更复杂的符号推理（如积分、矩阵）是否同样有效。

### 影响与延伸思考
MathPrompter 的思路打开了“让 LLM 自己生成多套答案并相互校验”的新方向。随后出现的工作（如 **Self‑Consistency**、**Ensemble Prompting**）都在不同程度上借鉴了多解一致性检查的理念。对想进一步探索的读者，可以关注以下方向：  
- 将多解生成扩展到更高阶数学（微积分、离散数学），检验代码执行的可行性。  
- 研究在资源受限的模型上如何高效实现多解生成，可能需要轻量化的提示或分步调用。  
- 探索置信度标签在人机协作中的交互设计，例如让用户只审阅低置信度的解法。  

### 一句话记住它
让大语言模型一次性给出多种独立求解路径，并用代码执行和结果一致性来估计置信度，从而把算术推理的准确率从 78% 提升到 92%。