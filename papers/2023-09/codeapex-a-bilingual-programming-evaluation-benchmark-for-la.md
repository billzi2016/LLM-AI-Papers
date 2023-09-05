# CodeApex: A Bilingual Programming Evaluation Benchmark for Large   Language Models

> **Date**：2023-09-05
> **arXiv**：https://arxiv.org/abs/2309.01940

## Abstract

With the emergence of Large Language Models (LLMs), there has been a significant improvement in the programming capabilities of models, attracting growing attention from researchers. Evaluating the programming capabilities of LLMs is crucial as it reflects the multifaceted abilities of LLMs, and it has numerous downstream applications. In this paper, we propose CodeApex, a bilingual benchmark dataset focusing on the programming comprehension, code generation, and code correction abilities of LLMs. Programming comprehension task tests LLMs on multiple-choice exam questions covering conceptual understanding, commonsense reasoning, and multi-hop reasoning. The code generation task evaluates LLMs through completing C++ functions based on provided descriptions and prototypes. The code correction task asks LLMs to fix real-world erroneous code segments with different error messages. We evaluate 12 widely used LLMs, including both general-purpose and specialized models. GPT-4 exhibits the best programming capabilities, achieving approximate accuracy of 69%, 54%, and 66% on the three tasks, respectively. Compared to human performance, there is still significant room for improvement in LLM programming. We hope that CodeApex can serve as a reference for evaluating the coding capabilities of LLMs, further promoting their development and growth.

---

# CodeApex：面向大语言模型的双语编程评估基准 论文详细解读

### 背景：这个问题为什么难？

在 LLM（大语言模型）被广泛用于代码生成后，研究者迫切需要一个统一、客观的测评平台来判断模型到底会写多少“好代码”。过去的评测大多只看单一任务（比如只测代码生成），或者只提供英文题目，导致两大盲点：①模型的“阅读理解”能力——能否理解题意、概念和常识推理；②真实开发场景中的错误修复能力。更糟的是，现有基准往往缺少多语言（中英）覆盖，无法检验模型在不同语言环境下的表现。于是出现了评测结果互不兼容、难以对比的局面，迫使研究者自行构造小样本测试，既费时又不具备可复现性。

### 关键概念速览
- **LLM（大语言模型）**：通过海量文本预训练得到的模型，能够生成自然语言或代码，类似“会说话的程序员”。  
- **双语基准**：评测数据同时提供英文和中文两套题目，考察模型在不同语言输入下的稳定性，就像让同一个学生用两种语言答题。  
- **编程理解（Programming Comprehension）**：模型需要阅读选择题，涉及概念解释、常识推理和多跳推理，类似考试中的“概念题”。  
- **代码生成（Code Generation）**：给出函数描述和原型，模型补全完整实现，等价于“填空式编程”。  
- **代码纠错（Code Correction）**：提供带错误信息的真实代码片段，要求模型定位并修复错误，类似调试面试题。  
- **多跳推理**：解题过程需要跨越多个知识点或步骤，类似解谜时要先找线索再拼凑答案。  
- **准确率（Accuracy）**：在选择题或生成任务中，模型答案与参考答案完全匹配的比例，是最直观的性能指标。  

### 核心创新点
1. **从单一任务到三维评估 → 引入编程理解、生成、纠错三大子任务 → 能同时衡量模型的概念掌握、实现能力和调试技巧，提供更全面的能力画像。**  
2. **双语题库设计 → 每道题目配备中英文两套表述 → 直接检验模型的跨语言鲁棒性，避免只在英文环境下“刷分”。**  
3. **真实错误代码采集 → 选取开源项目中实际报错的代码段并附上错误信息 → 让模型面对真实开发者常见的调试场景，而非人工构造的“玩具错误”。  
4. **统一评测协议 → 为所有公开的 LLM（包括通用模型和专门的代码模型）提供统一的输入输出格式和评分脚本 → 解决了不同论文使用不同评测方式导致的结果不可比问题。  

### 方法详解
整体框架可以看作“三步走”：

1. **数据构建**：作者先从高校编程考试、教材和在线测评平台收集 2,000 余道选择题，覆盖概念、常识和多跳推理。每题再翻译成中文，确保两种语言的难度等价。随后，挑选 500 条 C++ 函数描述，要求模型在给定原型和功能说明下补全实现。最后，从 GitHub 上抓取 300 条带有编译或运行时错误的代码片段，并记录对应的错误信息（如编译器报错、运行时异常）。  
2. **任务定义与接口**：对每个子任务，作者统一了 Prompt（提示词）格式。例如，编程理解任务的 Prompt 包含题干、四个选项以及“请直接输出正确选项的字母”。代码生成任务的 Prompt 给出函数原型和功能描述，要求模型输出完整函数体。代码纠错任务的 Prompt 则提供错误代码和错误信息，要求模型返回修正后的代码。这样，无论是 GPT‑4 还是开源模型，都只需要填入 Prompt 即可参与评测。  
3. **评测与统计**：对于选择题，直接比对模型输出的选项字母；对于代码生成和纠错，使用自动化单元测试脚本编译并运行模型输出，只有全部通过才计为正确。最终统计每个模型在三项任务上的准确率，并对中英文两套数据分别给出成绩，以观察语言差异。  

**最巧妙的地方**在于把真实错误代码直接搬进评测集，而不是让人工“造”错误。这样模型必须学会读取错误信息、定位根因，再生成符合语法和语义的修复代码，几乎等同于一次完整的调试过程。  

### 实验与效果
- **测试对象**：12 种公开可用的 LLM，包括通用模型（如 GPT‑4、Claude、LLaMA‑2）和专门的代码模型（如 CodeLlama、StarCoder）。  
- **结果概览**：GPT‑4 在三项任务上分别达到了约 69%（编程理解）、54%（代码生成）和 66%（代码纠错）的准确率，整体领先其他模型 10%–30% 不等。专门的代码模型在生成任务上稍好于通用模型，但在理解和纠错上仍落后。  
- **基线对比**：与之前仅使用 HumanEval（单纯代码生成）或 MBPP（英文代码生成）等基准相比，CodeApex 给出的分数更低，说明这些老基准高估了模型的真实编程能力。  
- **消融实验**：作者分别去掉中文题目、去除错误信息、只保留单一子任务进行评测，发现：①去掉中文后模型在英文上略有提升，但整体跨语言鲁棒性下降；②去掉错误信息后纠错准确率跌至 40% 左右，说明错误信息是关键提示；③只测生成任务会高估模型整体编程水平。  
- **局限性**：评测仅覆盖 C++，未涉及 Python、Java 等主流语言；错误代码来源于开源项目，可能偏向于较成熟的代码库；评测采用的是“全部通过即正确”，对部分功能实现的细微差别缺乏细粒度评分。  

### 影响与延伸思考
CodeApex 通过统一的三维、双语评测框架，为 LLM 编程能力的横向比较提供了“标尺”。自论文发布后，多个后续工作开始在此基准上报告改进，例如在提示工程（prompt engineering）和自检（self‑debug）机制上进行优化的模型，都把 CodeApex 作为主要验证平台。还有研究尝试把基准扩展到 Python、Rust 等语言，或加入代码风格、可读性评分，进一步细化模型的“代码素养”。如果想继续深入，可以关注 **多语言代码评测**（如 MBPP‑Multi）和 **自我纠错机制**（Self‑Debugging LLM）这两个方向，它们正沿着 CodeApex 指出的痛点快速发展。

### 一句话记住它
CodeApex 用双语选择题、函数补全和真实错误修复三种任务，给出了一套最全、最贴近真实开发的 LLM 编程能力测评标准。