# Solving Challenging Math Word Problems Using GPT-4 Code Interpreter with   Code-based Self-Verification

> **Date**：2023-08-15
> **arXiv**：https://arxiv.org/abs/2308.07921

## Abstract

Recent progress in large language models (LLMs) like GPT-4 and PaLM-2 has brought significant advancements in addressing math reasoning problems. In particular, OpenAI's latest version of GPT-4, known as GPT-4 Code Interpreter, shows remarkable performance on challenging math datasets. In this paper, we explore the effect of code on enhancing LLMs' reasoning capability by introducing different constraints on the \textit{Code Usage Frequency} of GPT-4 Code Interpreter. We found that its success can be largely attributed to its powerful skills in generating and executing code, evaluating the output of code execution, and rectifying its solution when receiving unreasonable outputs. Based on this insight, we propose a novel and effective prompting method, explicit \uline{c}ode-based \uline{s}elf-\uline{v}erification~(CSV), to further boost the mathematical reasoning potential of GPT-4 Code Interpreter. This method employs a zero-shot prompt on GPT-4 Code Interpreter to encourage it to use code to self-verify its answers. In instances where the verification state registers as ``False'', the model shall automatically amend its solution, analogous to our approach of rectifying errors during a mathematics examination. Furthermore, we recognize that the states of the verification result indicate the confidence of a solution, which can improve the effectiveness of majority voting. With GPT-4 Code Interpreter and CSV, we achieve an impressive zero-shot accuracy on MATH dataset \textbf{(53.9\% $\to$ 84.3\%)}.

---

# 使用 GPT-4 代码解释器和基于代码的自我验证解决高难度数学文字题 论文详细解读

### 背景：这个问题为什么难？

数学文字题往往需要把自然语言描述转化为严谨的符号推理，过程包括抽取变量、列式、计算以及检验结果。传统的大语言模型（LLM）只能靠“在脑子里”一步步推理，容易出现算术错误或逻辑跳步。即使加入了链式思考（CoT）等技巧，模型仍缺乏对中间计算结果的真实反馈，导致在复杂题目上准确率停滞不前。GPT‑4 的代码解释器（Code Interpreter）把“写代码、跑代码、看输出”这套实验室流程搬进了模型内部，为数学推理提供了可执行的检验手段，却还没有系统化地利用这一能力。

### 关键概念速览
- **代码解释器（Code Interpreter）**：GPT‑4 的一个插件，能够把模型生成的 Python 代码实际运行并返回结果，就像在笔记本里即时执行脚本一样。它把抽象的推理变成了可观测的数值输出。
- **代码使用频率（Code Usage Frequency）**：指模型在一次推理过程中调用代码解释器的次数。频率高意味着模型更倾向于把每一步都交给代码来实现，频率低则更像纯语言推理。
- **显式代码自我验证（Explicit Code‑based Self‑Verification，CSV）**：一种零样本提示技巧，要求模型在给出答案后，用代码再次计算或检查答案的正确性，并在验证失败时自动改写答案。相当于让模型在“考试后”再做一次检查。
- **验证状态（Verification State）**：CSV 产生的布尔值（True/False），表示模型自检的结果。True 相当于“我对答案有信心”，False 则触发纠错循环。
- **多数投票（Majority Voting）**：在多次独立推理后取出现频率最高的答案。验证状态可以用来加权投票，让更可靠的答案拥有更大影响力。
- **MATH 数据集**：一个公开的高难度数学文字题集合，题目覆盖代数、几何、组合等多个分支，常被用来衡量模型的数学推理水平。

### 核心创新点
1. **从“代码频率”视角剖析性能来源**  
   - 之前的工作只报告了代码解释器整体提升，却没有量化模型到底用了多少代码。  
   - 这篇论文通过人为限制代码调用次数，发现模型的高分主要来自频繁生成并执行代码、检查输出、以及在异常输出时自行修正。  
   - 这一发现让研究者明确了“代码是提升数学推理的关键杠杆”，为后续方法设计提供了方向。

2. **显式代码自我验证（CSV）提示**  
   - 过去的提示大多让模型直接给出答案或写出思考链。  
   - 这里提出在零样本下给模型一个“先写代码、再自检、若不通过则改写”的指令，让模型主动进入“写代码 → 运行 → 检查 → 修正”的闭环。  
   - 结果显示，单模型的零样本准确率从 53.9% 飙升到 84.3%，相当于一次巨大的性能跳跃。

3. **把验证状态当作置信度用于多数投票**  
   - 传统多数投票把每次推理的答案等权处理。  
   - 作者利用 CSV 输出的 True/False 作为置信度标记，对通过自检的答案加权投票，进一步提升了集体决策的鲁棒性。  
   - 这一步在实验中略微提升了整体准确率，证明了验证信息的实用价值。

### 方法详解
整体思路可以拆成四个阶段：**（1）题目解析 →（2）代码生成 →（3）自我验证 →（4）错误纠正**。下面按顺序展开：

1. **题目解析**  
   - 模型先阅读数学文字题，提取关键量（变量、已知条件、求解目标），并在内部形成一个简短的结构化描述。这个步骤仍然是纯语言的，没有代码介入。

2. **代码生成**  
   - 基于上述结构化描述，模型被提示生成对应的 Python 代码块。代码通常包括：变量赋值、公式实现、数值计算以及打印答案。因为使用了代码解释器，生成的代码会立即在沙箱环境中执行，返回具体数值。

3. **显式代码自我验证（CSV）**  
   - 在得到初步答案后，模型收到第二轮提示：**“请用代码再次计算答案，比较两次结果是否一致，若不一致请说明并重新求解。”**  
   - 模型会写出一个验证函数，重新运行原始计算或采用等价的检查方式（比如把答案代回原式检验是否成立）。运行后得到布尔值 `True`（验证通过）或 `False`（验证失败）。

4. **错误纠正**  
   - 若验证返回 `False`，模型进入纠错循环：它会分析错误原因（如算术溢出、符号误用），在提示的引导下修改代码或重新组织计算步骤，然后再次执行并验证。这个过程类似学生在考试后检查卷子、发现错误后改正的行为。  
   - 循环最多进行几次（论文未给出具体上限），直至验证通过或达到预设的尝试次数。

**关键细节**  
- **零样本提示**：所有指令都是一次性给出的，没有任何示例示范，完全依赖模型的通用能力。  
- **验证状态的置信度**：在多次独立推理（如 5 次）后，模型会把每次的答案和对应的验证状态一起输出，后续的多数投票只计入 `True` 状态的答案，提升了集体决策的可靠性。  
- **最巧妙的点**：把“代码执行结果”当作外部环境的反馈信号，让模型能够在内部形成“感知-行动-感知”的闭环，而不是单向的语言推理。

### 实验与效果
- **数据集**：主要在公开的 MATH 数据集上评估，覆盖 12 类数学子任务，难度普遍高于普通算术题。  
- **基线对比**：与普通 GPT‑4（不使用代码解释器）以及使用代码解释器但不做自检的版本相比，CSV 的零样本准确率从 53.9% 提升到 84.3%。这相当于在同样的模型规模下，性能提升了约 30 个百分点。  
- **消融实验**：论文通过限制代码使用频率、去掉自我验证步骤以及不使用验证状态加权投票三种设置，分别验证了每个模块的贡献。结果显示：去掉 CSV 直接下降到约 60%，仅保留验证状态加权投票的提升幅度约为 2%。  
- **局限性**：作者指出，CSV 依赖于代码解释器的执行环境，若题目涉及符号推导或需要高精度数学库，当前的 Python 环境可能不够；此外，验证循环的最大次数是经验设定，过多循环会导致推理时间显著增长。

### 影响与延伸思考
这篇工作把“写代码 → 运行 → 检查”引入语言模型的数学推理，开启了“代码驱动的自我纠错”新思路。随后的研究开始探索更通用的 **Tool‑augmented LLM** 框架，例如让模型调用符号计算引擎、图形绘制工具甚至搜索引擎来进行自检。还有工作尝试把 CSV 的思路迁移到编程题、物理推理等需要外部工具验证的任务上。想进一步了解，可以关注 **Toolformer**、**ReAct**、以及最近的 **Self‑Consistency** 系列论文，它们都在不同维度上强化模型的“自我监督”能力。

### 一句话记住它
让 GPT‑4 先写代码、再用代码检查答案，失败就自动改写——代码自我验证把数学推理的准确率从 50% 直接推到 80% 以上。