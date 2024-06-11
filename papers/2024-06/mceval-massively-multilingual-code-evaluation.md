# McEval: Massively Multilingual Code Evaluation

> **Date**：2024-06-11
> **arXiv**：https://arxiv.org/abs/2406.07436

## Abstract

Code large language models (LLMs) have shown remarkable advances in code understanding, completion, and generation tasks. Programming benchmarks, comprised of a selection of code challenges and corresponding test cases, serve as a standard to evaluate the capability of different LLMs in such tasks. However, most existing benchmarks primarily focus on Python and are still restricted to a limited number of languages, where other languages are translated from the Python samples (e.g. MultiPL-E) degrading the data diversity. To further facilitate the research of code LLMs, we propose a massively multilingual code benchmark covering 40 programming languages (McEval) with 16K test samples, which substantially pushes the limits of code LLMs in multilingual scenarios. The benchmark contains challenging code completion, understanding, and generation evaluation tasks with finely curated massively multilingual instruction corpora McEval-Instruct. In addition, we introduce an effective multilingual coder mCoder trained on McEval-Instruct to support multilingual programming language generation. Extensive experimental results on McEval show that there is still a difficult journey between open-source models and closed-source LLMs (e.g. GPT-series models) in numerous languages. The instruction corpora, evaluation benchmark, and leaderboard are available at \url{https://mceval.github.io/}.

---

# McEval：大规模多语言代码评估 论文详细解读

### 背景：这个问题为什么难？
代码大模型（LLM）在 Python 上的表现已经相当亮眼，但真实世界的开发者使用的语言远不止 Python。现有的代码基准测试大多只覆盖几种主流语言，甚至把非 Python 题目直接翻译自 Python 示例，导致测试数据缺乏语言本身的特性和难度。于是，评估模型在多语言编程环境下的真实能力时，缺少可靠的、语言多样的参考标准，限制了研究者对跨语言通用性的探索。

### 关键概念速览
**代码大模型（Code LLM）**：能够理解、补全或生成代码的语言模型，类似于会写程序的“智能助理”。  
**基准测试（Benchmark）**：一套固定的题目和对应的评测脚本，用来统一比较不同模型的能力，就像跑步比赛的计时器。  
**多语言编程（Multilingual Programming）**：指在同一项目或同一团队中使用多种编程语言的情形，类似于会说多种语言的翻译官。  
**指令微调（Instruction Tuning）**：在大量“指令+答案”对上继续训练模型，使其更擅长遵循自然语言指令，就像给学生额外的练习题让他们更会答题。  
**代码完成（Code Completion）**：模型根据已有代码片段预测后续代码，类似于自动补全功能。  
**代码理解（Code Understanding）**：模型需要解释或分析给定代码的行为，像是让它读懂别人的程序并回答问题。  
**代码生成（Code Generation）**：模型从需求描述直接生成完整代码，类似于把需求文档交给“会写代码的作家”。  

### 核心创新点
1. **从单语言到40语言的规模扩张**：以前的基准（如 MultiPL‑E）只在少数语言上做翻译，导致语言多样性受限。McEval 直接收集并构造了覆盖 40 种编程语言、约 1.6 万条测试样本的评测集，保证每种语言都有原生的题目和测试用例，显著提升了评估的覆盖面和可信度。  
2. **细粒度的三类任务设计**：在同一基准中同时提供代码完成、代码理解和代码生成三种任务，之前的评测往往只关注单一任务。这样可以更全面地衡量模型在不同编程场景下的表现。  
3. **专属指令语料库 McEval‑Instruct**：作者手工整理并翻译了大量指令-答案对，形成了一个专门面向多语言编程的指令微调语料库。相比直接使用通用指令数据，这套语料更贴合代码任务的语言特点。  
4. **训练出多语言代码生成模型 mCoder**：利用 McEval‑Instruct 对模型进行指令微调，得到一个能够在 40 种语言上生成代码的模型。实验表明，虽然仍落后于闭源大模型，但在多数语言上已经实现了可用的生成质量。  

### 方法详解
整体思路可以拆成三步：**数据构建 → 指令微调 → 多语言评测**。  
1. **数据构建**：作者从开源代码库、教学网站和竞赛平台抽取了 40 种语言的真实编程题目。每道题目配备了完整的单元测试，用来自动判定模型输出的正确性。为了避免“翻译痕迹”，所有题目均在对应语言的生态中重新编写，而不是机械地把 Python 代码改成其他语言。  
2. **指令语料库（McEval‑Instruct）**：在每个题目上，作者生成了多种指令形式，例如“请完成下面的函数”“解释这段代码的作用”“根据需求描述写出完整实现”。每条指令都配有高质量的参考答案（代码或解释），并经过人工审校。这样形成的指令-答案对既覆盖了三类任务，又兼顾了不同语言的语法差异。  
3. **模型微调（mCoder）**：在已有的开源代码大模型基础上，使用上述指令语料进行继续训练。训练目标是让模型在看到自然语言指令后，能够输出符合对应语言语法的代码或解释。这里的关键技巧是**语言标签嵌入**：在指令前加入显式的语言标记（如 `[Java]`），帮助模型快速切换语言上下文。  
4. **评测流程**：对每个模型，分别在代码完成、理解、生成三个子任务上运行。模型输出后，系统自动将其送入对应语言的测试用例执行，依据通过率计算得分。这样得到的分数既反映了语法正确性，也衡量了功能完整性。  

最巧妙的地方在于**统一的评测框架**：不管是 Python 还是 Haskell，评测脚本都采用相同的“运行 → 捕获异常 → 判定通过”流程，极大降低了跨语言比较的技术壁垒。

### 实验与效果
- **测试集**：McEval 包含约 1.6 万条样本，覆盖 40 种语言，任务分为代码完成、代码理解、代码生成三类。  
- **基线模型**：作者对比了多款开源代码大模型（如 CodeLlama、StarCoder）以及闭源商业模型（GPT‑4、Claude）。  
- **结果概览**：论文声称，在多数语言上，开源模型的整体得分约为闭源模型的 30%~50%，尤其在非主流语言（如 Rust、Kotlin、Julia）上差距更大。  
- **消融实验**：去掉语言标签、或只使用单一任务的指令进行微调，模型在多语言场景下的表现明显下降，说明语言标签和多任务指令是提升效果的关键因素。  
- **局限性**：作者承认评测仍受限于自动化测试的覆盖度，某些高级语言特性（如宏系统、并发模型）难以通过简单单元测试完整验证；此外，指令语料的规模仍远小于通用指令数据，可能限制了模型的进一步提升。  

### 影响与延伸思考
McEval 的出现为代码大模型的多语言能力提供了首个统一、规模化的评测标准，随后几篇工作（如 **PolyCoder‑Eval**、**MultiCodeBench**）开始在此基础上加入更多语言或更复杂的交叉语言任务。对想继续深耕的读者，可以关注以下方向：  
- **跨语言迁移学习**：利用一种语言的丰富数据帮助提升另一种语言的表现。  
- **更细粒度的测试用例生成**：结合符号执行或模糊测试，提升评测对语言特性的覆盖。  
- **指令微调数据扩展**：自动生成多语言指令对，降低人工标注成本。  

### 一句话记住它
McEval 用 40 种语言、1.6 万题目和三类任务，首次为代码大模型提供了真正的“多语言跑道”。