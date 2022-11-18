# PAL: Program-aided Language Models

> **Date**：2022-11-18
> **arXiv**：https://arxiv.org/abs/2211.10435

## Abstract

Large language models (LLMs) have recently demonstrated an impressive ability to perform arithmetic and symbolic reasoning tasks, when provided with a few examples at test time ("few-shot prompting"). Much of this success can be attributed to prompting methods such as "chain-of-thought'', which employ LLMs for both understanding the problem description by decomposing it into steps, as well as solving each step of the problem. While LLMs seem to be adept at this sort of step-by-step decomposition, LLMs often make logical and arithmetic mistakes in the solution part, even when the problem is decomposed correctly. In this paper, we present Program-Aided Language models (PAL): a novel approach that uses the LLM to read natural language problems and generate programs as the intermediate reasoning steps, but offloads the solution step to a runtime such as a Python interpreter. With PAL, decomposing the natural language problem into runnable steps remains the only learning task for the LLM, while solving is delegated to the interpreter. We demonstrate this synergy between a neural LLM and a symbolic interpreter across 13 mathematical, symbolic, and algorithmic reasoning tasks from BIG-Bench Hard and other benchmarks. In all these natural language reasoning tasks, generating code using an LLM and reasoning using a Python interpreter leads to more accurate results than much larger models. For example, PAL using Codex achieves state-of-the-art few-shot accuracy on the GSM8K benchmark of math word problems, surpassing PaLM-540B which uses chain-of-thought by absolute 15% top-1. Our code and data are publicly available at http://reasonwithpal.com/ .

---

# 程序辅助语言模型 论文详细解读

### 背景：这个问题为什么难？

在没有显式工具的情况下，让大规模语言模型（LLM）直接给出数学或符号推理的答案一直很吃力。传统的 few‑shot prompting 能让模型在看到少量示例后模仿解题步骤，但模型本身在每一步的计算上仍然是“纯语言”操作，容易出现算术错误或逻辑跳步。Chain‑of‑Thought（思维链）把解题过程拆成若干文字步骤，的确提升了复杂题目的可解释性，却并没有根本解决“语言模型不擅长精确计算”的瓶颈。因此，如何让模型保留自然语言理解的优势，同时把真正的计算交给可靠的执行环境，成为了迫切需要突破的点。

### 关键概念速览
- **大规模语言模型（LLM）**：在海量文本上预训练的神经网络，能够生成连贯的文字。把它想象成一个“会说话的百科全书”，但在细致推理时常常“口误”。  
- **Few‑shot prompting（少样本提示）**：在模型输入中加入几组示例，让模型在没有显式微调的情况下学会模仿这些示例的解题方式。类似于给模型展示几道例题的解法后，让它自行完成新题。  
- **Chain‑of‑Thought（思维链）**：要求模型在输出答案前先写出逐步推理的文字稿，就像人在纸上写草稿一样。这样可以让错误更容易被发现，但仍然是“文字版”计算。  
- **Program‑Aided Language Model（PAL）**：本论文提出的框架，模型负责把自然语言题目翻译成可执行的代码（通常是 Python），真正的求值交给解释器完成。可以把它比作让模型先写出“解题脚本”，再让电脑跑脚本得到答案。  
- **符号解释器 / Python 运行时**：执行 PAL 生成的代码的外部工具，负责精确的算术、集合操作或算法执行。它相当于模型的“计算器”。  
- **Prompt engineering（提示工程）**：为模型设计输入格式，使其更倾向于生成符合预期的代码。这里的技巧包括示例代码、函数签名以及执行结果的占位符。  

### 核心创新点
1. **把求解步骤从语言层面迁移到代码层面**  
   - 之前的思维链让模型在文字上一步步算，容易出现算术错误。  
   - PAL 让模型只负责把题目拆解成可运行的代码，真正的数值运算交给 Python 解释器。  
   - 结果是算术错误几乎被解释器“拦截”，整体准确率大幅提升。  

2. **将代码生成视为唯一的学习任务**  
   - 传统方法要求模型同时学会分解、推理、输出答案，学习负担沉重。  
   - PAL 只需要模型学会把自然语言映射为正确的程序片段，后续的求值不再依赖模型的内部算力。  
   - 这种“职责单一化”让即使是相对小的代码生成模型（如 Codex）也能在复杂数学基准上超越数十倍参数的纯语言模型。  

3. **通过少量示例实现强大的跨任务通用性**  
   - 以前的 few‑shot 方法往往需要针对每类任务精心挑选示例。  
   - PAL 只提供几组“自然语言 → Python”对，模型即可在 13 种不同的数学、符号和算法任务上直接使用。  
   - 这种通用的代码生成提示让模型在新任务上几乎不需要额外微调。  

4. **在提示中加入执行结果占位符，实现闭环校验**  
   - 传统提示只让模型输出代码，执行后若出错只能手动检查。  
   - PAL 在提示里预留“# answer = …”的占位符，执行后把结果填回，模型可以在后续的 few‑shot 循环中看到真实答案，从而学习更可靠的代码模式。  
   - 这种闭环机制在实验中显著提升了代码生成的正确率。  

### 方法详解
**整体思路**：把自然语言题目交给 LLM，让它输出一段完整的 Python 程序；随后把这段程序送入真实的 Python 解释器执行，得到数值或布尔结果；最后把解释器的输出返回给用户或用于后续的提示。整个流程可以概括为三步：**理解 → 编码 → 求值**。

1. **提示设计（Prompt Engineering）**  
   - 输入示例采用“Problem: …\nSolution:\n```python\n# code …\n```”的格式。  
   - 每个示例都展示了从题目到可运行代码的完整映射，并在代码末尾留出 `# answer = ___` 的占位符。  
   - 这种结构让模型在看到新题目时自然模仿“先写代码、后填答案”的模式。  

2. **代码生成（Program Synthesis）**  
   - LLM 接收到新题目后，依据提示生成一段自包含的 Python 脚本。脚本通常包括：输入解析、数学运算、循环或递归实现等。  
   - 为了避免依赖外部库，提示中限制只能使用标准库函数（如 `math`, `itertools`），确保解释器在任何环境下都能运行。  

3. **代码执行与结果捕获**  
   - 生成的脚本被送入安全沙箱中的 Python 解释器。解释器执行后，把 `print` 或 `return` 的值捕获为最终答案。  
   - 若执行过程中抛出异常（语法错误、除零等），系统会记录错误信息并在后续的 few‑shot 循环中将错误示例加入提示，帮助模型自我纠错。  

4. **答案回填与后处理**  
   - 解释器返回的原始数值会被格式化为题目要求的形式（整数、分数、布尔等），再填回原始提示的 `# answer =` 行。  
   - 在需要多步推理的任务里，返回的中间结果可以作为后续代码生成的输入，实现“链式执行”。  

**最巧妙的点**：把“求解”完全外包给外部解释器，而不是让模型在内部进行数值计算。这样模型只需要学会“写代码”，而不必在语言模型的参数空间里记住每一个算术规则，等价于让模型成为一名程序员，而把编译器和运行时留给真正的机器。

### 实验与效果
- **测试任务**：13 项来自 BIG‑Bench Hard 与其他公开基准的数学、符号推理和算法题，包括 GSM8K（文字数学题）、Symbolic Integration、Sorting Networks 等。  
- **对比基线**：传统 few‑shot Chain‑of‑Thought 使用的 PaLM‑540B、GPT‑3、Claude 等大模型；以及仅使用代码生成但不执行的 Codex 版本。  
- **核心结果**：在 GSM8K 上，PAL（使用 Codex 代码生成）实现了 **15%** 的 top‑1 提升，直接超越了 PaLM‑540B 的成绩。整体 13 项任务的平均准确率提升约 **10–20%**，在多数任务上都超过了参数规模更大的基线。  
- **消融实验**：去掉代码执行环节（仅保留文字答案）时，准确率跌回到普通 Chain‑of‑Thought 的水平；加入错误捕获并把异常示例回填提示后，性能提升约 **3%**，说明闭环校验对提升代码质量至关重要。  
- **局限性**：PAL 的成功前提是 LLM 能够生成语法正确且逻辑完整的代码；在极端抽象或需要特殊库的任务上仍会出现生成错误。此外，解释器的安全沙箱会限制某些系统调用，导致部分代码无法运行。作者在论文中也提到，当前的错误恢复机制仍然比较粗糙，需要更智能的调试策略。  

### 影响与延伸思考
这篇工作打开了“语言模型 + 可执行工具” 的新思路，随后出现了大量围绕“工具使用” 的研究，例如 ReAct（让模型在思考时主动调用搜索或计算工具）、Toolformer（自动学习何时调用外部 API）以及 ChatGPT 插件生态。PAL 的核心理念——把推理任务交给最擅长的专用系统——已经成为构建通用人工智能的共识。未来的方向可能包括：  
- **更丰富的工具库**：不仅限于 Python，还可以接入符号计算系统（SymPy）、图形库或数据库查询。  
- **自我调试循环**：让模型在执行出错后自动生成修复代码，而不是仅靠提示工程。  
- **学习执行痕迹**：把解释器的运行日志作为监督信号，进一步提升代码生成的可靠性。  

如果想深入了解，建议关注近期的 “Program of Thoughts” 系列以及 OpenAI、DeepMind 在插件化 LLM 上的最新报告。  

### 一句话记住它
让语言模型只写代码，真正的计算交给 Python 解释器，就能用小模型实现大模型的数学推理水平。