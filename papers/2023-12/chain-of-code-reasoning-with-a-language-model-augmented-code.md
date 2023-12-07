# Chain of Code: Reasoning with a Language Model-Augmented Code Emulator

> **Date**：2023-12-07
> **arXiv**：https://arxiv.org/abs/2312.04474

## Abstract

Code provides a general syntactic structure to build complex programs and perform precise computations when paired with a code interpreter - we hypothesize that language models (LMs) can leverage code-writing to improve Chain of Thought reasoning not only for logic and arithmetic tasks, but also for semantic ones (and in particular, those that are a mix of both). For example, consider prompting an LM to write code that counts the number of times it detects sarcasm in an essay: the LM may struggle to write an implementation for "detect_sarcasm(string)" that can be executed by the interpreter (handling the edge cases would be insurmountable). However, LMs may still produce a valid solution if they not only write code, but also selectively "emulate" the interpreter by generating the expected output of "detect_sarcasm(string)". In this work, we propose Chain of Code (CoC), a simple yet surprisingly effective extension that improves LM code-driven reasoning. The key idea is to encourage LMs to format semantic sub-tasks in a program as flexible pseudocode that the interpreter can explicitly catch undefined behaviors and hand off to simulate with an LM (as an "LMulator"). Experiments demonstrate that Chain of Code outperforms Chain of Thought and other baselines across a variety of benchmarks; on BIG-Bench Hard, Chain of Code achieves 84%, a gain of 12% over Chain of Thought. In a nutshell, CoC broadens the scope of reasoning questions that LMs can answer by "thinking in code".

---

# 代码链：语言模型增强代码模拟器的推理 论文详细解读

### 背景：这个问题为什么难？
在让大语言模型（LLM）解决逻辑、算数甚至语义混合任务时，传统的“思维链”（Chain‑of‑Thought）只能让模型把推理过程写成自然语言。虽然这种方式能提升复杂题目的正确率，但当任务需要精确的计算或对模糊概念进行细粒度判断时，模型往往会卡在细节实现上。直接让模型生成可执行代码可以利用解释器的精准计算能力，却又面临“代码写不对、边界情况处理不到位”的难题。于是，如何让模型既能利用代码的严谨，又能在无法完整实现的子任务上自行“填空”，成为亟待突破的瓶颈。

### 关键概念速览
- **Chain‑of‑Thought（思维链）**：让模型在给出答案前先把推理步骤写出来，类似于人在解题时先列草稿，帮助模型保持逻辑连贯。  
- **代码解释器**：能够把字符串形式的代码实际运行并返回结果的系统，像 Python REPL 那样提供精确计算。  
- **伪代码（flexible pseudocode）**：一种不要求完全符合语法的代码写法，模型可以在其中标记“这里需要实现 X”，让解释器捕获未定义行为。  
- **LMulator（语言模型模拟器）**：当解释器遇到未实现的函数时，转而让语言模型直接输出该函数的预期返回值，相当于让模型“假装”执行代码。  
- **未定义行为捕获**：解释器在运行时检测到缺失实现或异常时，会把控制权交给 LMulator，而不是直接报错。  
- **BIG‑Bench Hard**：一个集合了高难度推理、数学和常识任务的基准，用来衡量模型在极端挑战下的表现。  

### 核心创新点
1. **从纯自然语言推理到代码+语言混合**：以前的思维链只能让模型写文字步骤 → 这篇论文让模型把每一步包装成可执行的（或可模拟的）代码块 → 通过解释器的精准计算和 LMulator 的灵活填补，整体正确率显著提升。  
2. **引入“未定义行为捕获”机制**：传统代码生成若出现缺失实现会直接导致运行错误 → 论文在解释器层面加入检测，一旦发现未实现的函数就交给语言模型生成输出 → 这种“代码‑模型双保险”让模型即使不懂细节也能继续推理。  
3. **灵活伪代码规范**：过去要求模型输出严格可运行的代码，导致生成难度大 → 本文允许模型使用占位符式的伪代码，只要结构清晰、易于解释器捕获即可 → 降低了代码生成的门槛，同时保持了思考的结构化。  
4. **统一的“Chain of Code”框架**：把思维链的步骤映射到代码层面，并在每一步加入解释‑模拟闭环 → 与单纯的思维链或纯代码驱动的方案相比，CoC 在多个基准上实现了约 12% 的绝对提升（如 BIG‑Bench Hard 从 72% 提升到 84%）。  

### 方法详解
整体思路可以拆成三大阶段：**任务拆解 → 代码生成 → 解释/模拟**。  
1. **任务拆解**：给模型一个完整的问题提示，要求它把问题分解成若干子任务，并用伪代码的形式写出每个子任务的实现框架。比如“统计文章中的讽刺句数”，模型会输出 `count = 0; for sentence in essay: sarcasm = detect_sarcasm(sentence); count += sarcasm`，其中 `detect_sarcasm` 只是一行占位。  
2. **代码生成**：模型在每个子任务的占位处继续生成实现细节。如果模型对某个函数不确定，它可以直接写成 `# TODO: implement detect_sarcasm`，或者直接写出期望的返回值（如 `detect_sarcasm(sentence) -> 0 or 1`）。  
3. **解释/模拟闭环**：解释器逐行执行生成的代码。遇到真正实现的函数时交给底层解释器运行；遇到占位或未实现的函数时触发 **未定义行为捕获**，把调用信息（函数名、参数）回传给语言模型。模型在此时充当 **LMulator**，基于上下文直接输出该函数的预测结果，解释器再把这个结果当作真实返回值继续执行后续代码。  

**关键细节**  
- **捕获点的标记**：在伪代码中使用特定注释或特殊函数名（如 `__LMULATE__`）让解释器易于识别。  
- **上下文传递**：LMulator 在生成返回值时会看到完整的执行栈和已经得到的中间变量，确保它的预测与当前推理状态一致。  
- **错误回滚**：如果 LMulator 给出的返回值导致后续计算出现异常，解释器会再次请求模型重新生成该函数的输出，形成一种“纠错”循环。  

最巧妙的地方在于：**模型不必一次性写出完整、无误的代码**，只要提供结构化的思考框架并在关键点交给自己“模拟”，就能把代码的严谨性和语言模型的灵活性结合起来。

### 实验与效果
- **测试任务**：论文在 BIG‑Bench Hard、数学推理、常识问答以及语义分析等多类基准上评估。  
- **对比基线**：包括纯思维链（CoT）、直接代码生成（Code‑only）以及最新的 few‑shot 大模型。  
- **核心结果**：在 BIG‑Bench Hard 上，CoC 达到 **84%** 的整体正确率，比思维链提升 **12%**（原 CoT 约 72%）。在其他任务上也普遍保持 5‑10% 的提升幅度。  
- **消融实验**：去掉未定义行为捕获或改用纯代码执行，性能跌回到接近普通 CoT 的水平，说明 LMulator 与捕获机制是提升的关键因素。  
- **局限性**：论文承认对极其复杂的函数（如需要深度学习模型内部推理）仍然依赖模型自行模拟，误差会累积；此外，解释器的实现需要与模型 API 紧密配合，实际部署成本不低。  

### 影响与延伸思考
这篇工作打开了“代码+语言双向协同” 的新思路，随后出现了多篇把 LLM 当作 **可微分解释器** 或 **代码调试器** 的研究，例如把模型嵌入到 Jupyter 环境中实时补全未实现函数。推测未来会有更多围绕 **自适应代码生成 + 模型模拟** 的框架，尤其在需要高精度数值计算但又缺少完整实现的科学计算场景。想进一步了解，可关注 **NeurIPS 2024** 上的 “LM‑augmented execution” 系列论文以及开源项目 **LMulator‑Py**。  

### 一句话记住它
让大模型先写结构化的代码框架，再在无法实现的细节上“假装”执行——代码与语言模型共同思考，推理能力大幅升级。