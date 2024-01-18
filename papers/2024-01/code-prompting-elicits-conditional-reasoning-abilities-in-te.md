# Code Prompting Elicits Conditional Reasoning Abilities in Text+Code LLMs

> **Date**：2024-01-18
> **arXiv**：https://arxiv.org/abs/2401.10065

## Abstract

Reasoning is a fundamental component of language understanding. Recent prompting techniques, such as chain of thought, have consistently improved LLMs' performance on various reasoning tasks. Nevertheless, there is still little understanding of what triggers reasoning abilities in LLMs in the inference stage. In this paper, we introduce code prompting, a chain of prompts that transforms a natural language problem into code and directly prompts the LLM using the generated code without resorting to external code execution. We hypothesize that code prompts can elicit certain reasoning capabilities of LLMs trained on text and code and utilize the proposed method to improve conditional reasoning, the ability to infer different conclusions depending on the fulfillment of certain conditions. We find that code prompting exhibits a high-performance boost for multiple LLMs (up to 22.52 percentage points on GPT 3.5, 7.75 on Mixtral, and 16.78 on Mistral) across multiple conditional reasoning datasets. We then conduct comprehensive experiments to understand how code prompts trigger reasoning abilities and which capabilities are elicited in the underlying models. Our analysis of GPT 3.5 reveals that the code formatting of the input problem is essential for performance improvement. Furthermore, code prompts improve sample efficiency of in-context learning and facilitate state tracking of variables or entities.

---

# 代码提示激发文本+代码大语言模型的条件推理能力 论文详细解读

### 背景：这个问题为什么难？

条件推理要求模型在不同前提下给出不同结论，类似“如果A成立则B，否则C”。传统的大语言模型（LLM）在零样本或少样本设置下往往只能给出单一答案，原因是它们的内部表征更擅长捕捉整体语义而不是细粒度的状态切换。已有的思维链（Chain‑of‑Thought，CoT）通过让模型先写推理步骤提升了算数、逻辑等任务的表现，但对“如果‑则‑否则”这类需要显式追踪条件满足情况的任务帮助有限。根本的瓶颈在于：模型在推理时缺少一种自然的、能够把条件结构化表达的中间语言。于是研究者开始探索能否借助模型已经学到的代码知识来填补这块空白。

### 关键概念速览
**条件推理**：根据是否满足特定前提，推导出不同的结论。比如“若今天下雨，则带伞，否则不带”。  
**思维链（CoT）**：让模型在输出答案前先写出逐步推理过程，类似人在解题时的草稿。  
**代码提示（Code Prompting）**：把自然语言问题转写成一段代码（通常是伪代码或可执行语言的片段），然后直接喂给模型，让它在代码语境中完成推理。  
**文本+代码大语言模型**：在预训练阶段既见过大量自然语言，又见过开源代码的模型，如 GPT‑3.5、Mistral、Mixtral 等。  
**原位推理（In‑situ Reasoning）**：模型在不调用外部解释器或执行环境的情况下，仅凭内部知识完成推理。  
**样本效率**：在少量示例（few‑shot）下模型能够学到任务规律的能力。  
**状态追踪**：在推理过程中保持并更新变量或实体的当前取值，类似程序里的变量赋值与检查。

### 核心创新点
1. **把自然语言问题包装成代码 → 直接在模型面前呈现代码结构**  
   传统做法是给模型一段文字或 CoT 提示，这种形式对条件分支的表达不够明确。作者设计了一套“代码提示”链：先用模板把题目转成带有 `if`、`else`、变量声明等关键字的代码块，再让模型在这段代码里写出答案。这样模型可以利用它在代码预训练中学到的控制流概念，自然地进行条件分支推理。实验显示，GPT‑3.5 在多个条件推理基准上提升了 22.52 个百分点。

2. **无需实际执行代码的原位推理 → 让模型内部完成“模拟执行”**  
   与需要外部解释器的“程序调用”不同，作者仅把代码作为提示输入，模型自行在内部“想象”代码的执行路径。这避免了额外的系统开销，也验证了模型本身已经具备一定的代码语义理解能力。

3. **系统性分析代码格式对推理的驱动作用 → 揭示关键因素**  
   通过对 GPT‑3.5 的消融实验，作者发现去掉代码的缩进、关键字或注释会显著削弱性能，说明模型对代码的结构化信息非常敏感。此发现帮助解释为什么代码提示能够激活潜在的条件推理能力。

4. **提升少样本学习的样本效率与变量追踪能力**  
   在同样的 few‑shot 设置下，使用代码提示的模型比普通 CoT 能以更少的示例学到任务规则；同时，模型在代码块中对变量的赋值与检查表现出更稳健的状态追踪，类似程序员在调试时的思路。

### 方法详解
**整体思路**  
整个流程可以概括为三步：① 将自然语言题目转化为代码模板；② 用该代码作为提示喂给文本+代码 LLM；③ 从模型输出的代码中抽取最终答案。整个过程不涉及任何外部代码运行，所有推理都在模型内部完成。

**步骤拆解**  

1. **题目到代码的映射**  
   - **模板准备**：针对条件推理任务，作者预先设计了一个通用的代码框架，例如 Python 风格的 `if … else …` 结构。  
   - **填充变量**：自然语言中的实体、数值或布尔条件被映射为代码中的变量赋值语句，例如 `temperature = 30`、`is_raining = True`。  
   - **注释说明**：在代码前加入简短的注释，保留原始问题的文字描述，帮助模型对齐自然语言与代码语义。  

2. **模型推理**  
   - **提示构造**：完整的代码块（包括注释、变量声明、条件结构）被直接作为模型的输入。提示的最后通常会加上一行类似 `# 请在下面写出答案` 的指令，引导模型在代码后续位置输出答案。  
   - **内部“模拟执行”**：模型利用它在代码预训练中学到的控制流语义，隐式地评估 `if` 条件是否成立，并在相应的分支中生成答案。此过程不需要真实的解释器，只是模型内部的概率推断。  

3. **答案抽取**  
   - **输出解析**：模型的回复通常是继续代码块的形式，例如 `result = "带伞"` 或直接在注释里写 `# 答案：带伞`。后处理脚本根据预设的模式提取出答案字符串。  

**关键细节与巧思**  
- **代码缩进与关键字的保留**：实验表明，保持标准的缩进和 `if/else` 关键字是触发模型条件分支推理的核心。  
- **注释桥接**：在代码前加入自然语言注释，使模型能够在两种表征之间建立对应，提升了答案的准确率。  
- **不执行代码的安全性**：因为所有推理都在模型内部完成，避免了潜在的安全风险和执行环境配置问题。  

### 实验与效果
- **测试任务**：作者在多个公开的条件推理数据集上评估，包括“Conditional Reasoning Benchmark”、以及自建的若干自然语言条件问答集合。  
- **基线对比**：与零样本直接回答、few‑shot 直接回答、以及传统 CoT 三种基线相比，代码提示在所有模型上均实现显著提升。  
  - GPT‑3.5：最高提升 22.52%  
  - Mixtral：提升 7.75%  
  - Mistral：提升 16.78%  
- **消融实验**：去掉代码的 `if`/`else`、去掉缩进或删除注释后，性能回落到接近普通 CoT 的水平，说明代码结构是关键因素。  
- **样本效率**：在仅提供 1‑2 个示例的 few‑shot 设置下，代码提示的模型已经能够达到 5‑10% 的准确率提升，远超传统 CoT 需要 4‑5 个示例才能达到的效果。  
- **局限性**：实验仅覆盖条件推理场景，未验证对更复杂的数学或常识推理的通用性；此外，方法依赖模型对代码语言的熟悉度，对仅接受文本预训练的模型可能效果有限。  

### 影响与延伸思考
这篇工作打开了“用代码激活模型推理”的新思路，随后出现了多篇围绕“程序化提示”（Programmatic Prompting）或“结构化提示”（Structured Prompting）的研究，尝试把更多任务转化为伪代码或 DSL（领域专用语言）形式。还有工作把真实代码执行与模型推理结合，形成“工具调用”范式，进一步提升了模型在需要精确计算或外部知识时的表现。对想继续深入的读者，可以关注以下方向：① 将代码提示与实际解释器结合，实现可验证的推理；② 探索其他结构化语言（如 SQL、正则表达式）对不同任务的驱动效果；③ 研究如何在多模态模型（文本+代码+图像）中统一使用结构化提示。  

### 一句话记住它
把自然语言题目包装成代码，让模型像写程序一样思考，从而大幅提升条件推理能力。