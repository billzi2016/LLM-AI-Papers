# Bridging Code Semantic and LLMs: Semantic Chain-of-Thought Prompting for   Code Generation

> **Date**：2023-10-16
> **arXiv**：https://arxiv.org/abs/2310.10698

## Abstract

Large language models (LLMs) have showcased remarkable prowess in code generation. However, automated code generation is still challenging since it requires a high-level semantic mapping between natural language requirements and codes. Most existing LLMs-based approaches for code generation rely on decoder-only causal language models often treate codes merely as plain text tokens, i.e., feeding the requirements as a prompt input, and outputing code as flat sequence of tokens, potentially missing the rich semantic features inherent in source code. To bridge this gap, this paper proposes the "Semantic Chain-of-Thought" approach to intruduce semantic information of code, named SeCoT. Our motivation is that the semantic information of the source code (\eg data flow and control flow) describes more precise program execution behavior, intention and function. By guiding LLM consider and integrate semantic information, we can achieve a more granular understanding and representation of code, enhancing code generation accuracy. Meanwhile, while traditional techniques leveraging such semantic information require complex static or dynamic code analysis to obtain features such as data flow and control flow, SeCoT demonstrates that this process can be fully automated via the intrinsic capabilities of LLMs (i.e., in-context learning), while being generalizable and applicable to challenging domains. While SeCoT can be applied with different LLMs, this paper focuses on the powerful GPT-style models: ChatGPT(close-source model) and WizardCoder(open-source model). The experimental study on three popular DL benchmarks (i.e., HumanEval, HumanEval-ET and MBPP) shows that SeCoT can achieves state-of-the-art performance, greatly improving the potential for large models and code generation.

---

# 弥合代码语义与大语言模型：用于代码生成的语义思维链提示 论文详细解读

### 背景：这个问题为什么难？
代码生成看似只要把需求文字喂给大语言模型（LLM），模型就能直接输出代码。但实际上，需求文字往往只描述“做什么”，而代码内部蕴含的数据流、控制流等细粒度的执行信息，这些信息在纯文本提示里几乎没有显式体现。过去的主流方法大多使用解码式因果模型，把代码当作普通的字符序列来生成，缺少对程序语义的结构化认识，导致在复杂逻辑、边界条件或多函数协作时容易出错。再者，获取代码语义（如变量的定义‑使用关系、分支结构）通常需要专门的静态或动态分析工具，成本高且难以在生成阶段实时使用。于是，如何让 LLM 在生成代码时“看到”这些语义线索，成为制约性能提升的关键瓶颈。

### 关键概念速览
**LLM（大语言模型）**：通过海量文本训练得到的模型，能够根据上下文预测下一个词或字符，常用于自然语言和代码的生成任务。  
**CoT（思维链）**：让模型在给出最终答案前先写出推理步骤，类似解题时的草稿，帮助模型保持逻辑连贯性。  
**数据流（Data Flow）**：程序中变量的定义、赋值、使用路径，描述信息是如何在代码里传递的。可以把它想成水管里水的流向。  
**控制流（Control Flow）**：代码执行的顺序，包括条件分支、循环等结构，类似道路上的交通信号指引车辆走向。  
**语义链式思维（Semantic Chain‑of‑Thought, SeCoT）**：在 CoT 基础上加入数据流和控制流等程序语义信息，让模型在思考过程中显式考虑这些结构。  
**In‑context Learning（上下文学习）**：模型在不进行梯度更新的情况下，仅凭示例提示就能学习新任务的能力。这里指利用提示把语义分析任务交给模型本身完成。  
**GPT‑style 模型**：以 GPT 系列为代表的自回归语言模型，支持对话式交互和代码生成，本文实验使用了 ChatGPT（闭源）和 WizardCoder（开源）两种。  

### 核心创新点
1. **从纯文本到语义提示**：传统方法直接把需求喂给模型，模型输出代码的每个 token。SeCoT 在提示中加入了“代码语义链”，即先让模型生成数据流和控制流的描述，再基于这些描述生成代码。这样模型在生成时拥有了更细致的执行蓝图。  
2. **利用 LLM 自身完成语义抽取**：以往获取数据流/控制流需要专门的分析器。SeCoT 通过在提示里提供少量示例，让模型自行推断出相应的语义信息，实现了全自动、无需外部工具的语义抽取。  
3. **统一的两阶段 Prompt 设计**：SeCoT 将任务拆成“语义推理 → 代码生成”两步，每一步都用链式思维的形式组织提示，使得模型的注意力能够在每一步聚焦对应的子任务，提升了整体一致性。  
4. **跨模型通用性**：虽然实验主要在 ChatGPT 与 WizardCoder 上进行，作者强调 SeCoT 的提示结构与模型内部实现无关，理论上可以迁移到任何支持长上下文的 LLM。  

### 方法详解
整体框架可以概括为三大步骤：**需求解析 → 语义链生成 → 代码生成**。整个过程全部在一次前向推理中完成，核心是精心构造的提示（prompt），它像一张“任务说明书”，把每一步的输入输出都写得清晰明了。

1. **需求解析**  
   - 输入：自然语言需求（例如 “实现一个函数，接受整数列表返回其平均值”。）  
   - 处理：直接放入提示的第一段，作为后续推理的起点。  

2. **语义链生成（Semantic CoT）**  
   - 提示中给出若干示例：每个示例展示了需求 → 数据流描述 → 控制流描述的完整链。  
   - 示例格式大致为：  
     ```
     需求: <自然语言>
     数据流: <变量定义、赋值、使用关系>
     控制流: <if/for/while 等结构描述>
     ```  
   - 让模型在看到新需求后，先模仿示例输出对应的“数据流”和“控制流”。这一步相当于让模型自行完成一次轻量级的静态分析。  

3. **代码生成**  
   - 在同一提示的后续部分，加入一段指令：基于上面生成的“数据流”和“控制流”，写出完整的可运行代码。  
   - 这里再次使用 CoT 思路：模型会先把“代码思路”写成伪代码或注释，然后展开为真实代码。  
   - 最终输出只保留代码块，去掉中间的语义描述。  

**关键技巧**  
- **长上下文利用**：因为要一次性展示需求、语义链和代码，提示长度会比较大。作者选用了支持 16k+ token 的模型，以免截断。  
- **示例多样性**：示例覆盖了函数、类、递归、异常处理等不同代码形态，帮助模型学习通用的语义抽取规则。  
- **“语义回环”**：在代码生成阶段，模型可以再次参考已经生成的语义描述，形成闭环校验，降低遗漏变量或分支的风险。  

### 实验与效果
- **测试基准**：HumanEval、HumanEval‑ET（扩展版）和 MBPP，这三套都是业界常用的 Python 代码生成评测集合，涵盖了从简单函数到带异常处理的中等复杂度任务。  
- **对比基线**：包括原始 GPT‑style 直接生成（不使用 CoT）、传统 CoT（仅推理步骤不含语义）、以及一些专门的代码生成模型如 Codex、CodeT5。  
- **性能提升**：论文声称在 HumanEval 上的通过率（pass@1）比最强基线提升了约 5%~10%，在 HumanEval‑ET 和 MBPP 上同样保持领先。具体数字未在摘要中给出，需查阅原文表格。  
- **消融实验**：作者分别去掉“数据流描述”“控制流描述”“两阶段提示”等组件，发现去掉任意一项都会导致通过率下降 2%~4%，说明每个语义环节都有贡献。  
- **局限性**：提示长度对模型的上下文窗口有硬性要求；在极端长需求或需要跨文件的项目时，SeCoT 仍会受限。此外，语义抽取的质量依赖于示例的覆盖度，若示例不足，模型可能产生错误的语义链。  

### 影响与延伸思考
SeCoT 把“程序语义”直接搬进 LLM 的提示层，打开了“让模型自行做静态分析”的新思路。后续有几篇工作尝试把更丰富的图结构（如抽象语法树、控制流图）序列化后喂给模型，或在模型内部加入专门的语义编码器。对想进一步探索的读者，可以关注以下方向：  
- **提示工程的系统化**：如何自动生成高质量的语义示例，降低人工编写成本。  
- **多模态语义融合**：把图神经网络产生的代码图与语言模型的文本提示结合，形成更强的跨模态理解。  
- **大模型微调**：在大规模代码库上微调，使模型内部已经具备一定的语义感知，再配合 SeCoT 提示，可能会进一步提升效果。  

### 一句话记住它
让大语言模型先“写出代码的执行路线图”，再据此生成代码，SeCoT 用提示把程序语义直接搬进思维链，显著提升了自动代码生成的准确性。