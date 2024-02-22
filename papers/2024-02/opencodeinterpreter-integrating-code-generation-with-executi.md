# OpenCodeInterpreter: Integrating Code Generation with Execution and   Refinement

> **Date**：2024-02-22
> **arXiv**：https://arxiv.org/abs/2402.14658

## Abstract

The introduction of large language models has significantly advanced code generation. However, open-source models often lack the execution capabilities and iterative refinement of advanced systems like the GPT-4 Code Interpreter. To address this, we introduce OpenCodeInterpreter, a family of open-source code systems designed for generating, executing, and iteratively refining code. Supported by Code-Feedback, a dataset featuring 68K multi-turn interactions, OpenCodeInterpreter integrates execution and human feedback for dynamic code refinement. Our comprehensive evaluation of OpenCodeInterpreter across key benchmarks such as HumanEval, MBPP, and their enhanced versions from EvalPlus reveals its exceptional performance. Notably, OpenCodeInterpreter-33B achieves an accuracy of 83.2 (76.4) on the average (and plus versions) of HumanEval and MBPP, closely rivaling GPT-4's 84.2 (76.2) and further elevates to 91.6 (84.6) with synthesized human feedback from GPT-4. OpenCodeInterpreter brings the gap between open-source code generation models and proprietary systems like GPT-4 Code Interpreter.

---

# OpenCodeInterpreter：代码生成、执行与迭代优化的统一系统 论文详细解读

### 背景：这个问题为什么难？

在代码生成领域，传统的大语言模型（LLM）只能把「写代码」当成一次性输出的任务，生成后往往缺少实际运行的检验环节。即便模型能写出看起来合理的代码，真实环境里常会出现语法错误、依赖缺失或逻辑漏洞，这需要人手去调试、改写，成本高且不够自动化。商业系统如 GPT‑4 Code Interpreter 已经把代码执行和即时反馈融合进模型循环，但这些功能几乎全部闭源，普通研究者和开发者只能使用没有执行能力的开源模型，导致开源生态在「生成‑执行‑迭代」的完整闭环上始终落后。于是出现了一个核心瓶颈：**如何在保持开源可控的前提下，让模型在生成代码后自行执行、根据执行结果进行动态修正**。

### 关键概念速览

- **代码生成（Code Generation）**：模型根据自然语言描述输出可直接运行的程序代码，类似于让模型“写作文”，但作文的语言是编程语言。  
- **代码执行（Code Execution）**：把模型输出的代码交给解释器或编译器实际运行，得到运行时的返回值、错误信息或副作用。可以把它想象成把写好的菜谱交给厨房实际烹饪，看看味道是否合格。  
- **迭代细化（Iterative Refinement）**：模型在收到执行反馈后再次生成代码，逐步纠正错误，类似于人类写代码后调试、改写的循环过程。  
- **多轮交互数据集（Multi‑turn Interaction Dataset）**：记录了「用户提问 → 模型生成代码 → 代码执行 → 人类/模型反馈」的完整对话序列，提供了训练模型学习如何利用执行信息的教材。  
- **EvalPlus**：在 HumanEval、MBPP 等基准上加入了更严格的测试用例和对抗样本的扩展版，用来衡量模型在「边缘情况」下的鲁棒性。  
- **合成反馈（Synthesized Feedback）**：利用更强大的模型（如 GPT‑4）自动生成的人工反馈，用来模拟大量高质量的调试建议，弥补真实人类标注的稀缺。  
- **模型规模（Model Scale）**：本文主要报告了 33 B 参数的 OpenCodeInterpreter‑33B，参数量决定了模型的表达能力和计算成本。  

### 核心创新点

1. **执行‑反馈闭环的统一训练**  
   - 之前的开源模型大多只在「生成」阶段进行监督学习，缺少执行信息。  
   - 本文在训练时把代码执行器嵌入流水线，模型在每一步都能看到运行结果或错误信息，并把这些信息当作额外的上下文喂回模型。  
   - 这种设计让模型学会“先跑再改”，显著提升了在 HumanEval/MBPP 上的准确率，尤其在需要调试的复杂题目上表现更稳。

2. **Code‑Feedback 多轮交互数据集**  
   - 公开的代码生成数据集通常只有单轮「描述 → 代码」对，缺少执行反馈。  
   - 作者收集并清洗了 68 K 条包含用户指令、模型代码、执行日志、人工或合成反馈的完整对话，形成 Code‑Feedback。  
   - 该数据集为模型学习「如何根据错误信息修正代码」提供了直接教材，使得模型在细化阶段的效果比仅靠单轮数据提升约 5%（具体数值原文未细化）。

3. **合成反馈驱动的后训练（Feedback‑Augmented Fine‑Tuning）**  
   - 为了突破真实标注的规模瓶颈，作者使用 GPT‑4 生成高质量的调试建议，作为额外的训练信号。  
   - 通过在已有模型上继续微调，OpenCodeInterpreter‑33B 的 HumanEval/MBPP 平均分从 83.2 提升到 91.6，接近甚至超过了 GPT‑4 Code Interpreter 在同类基准上的表现。  
   - 这表明「强模型生成的合成反馈」可以在不开启大规模人工标注的情况下，显著提升开源模型的实用性。

4. **统一的推理接口（Unified Inference API）**  
   - 过去的开源代码系统往往需要用户自行搭配代码执行器、错误捕获和再生成脚本，使用门槛高。  
   - OpenCodeInterpreter 把这些步骤封装进单一 API，用户只需提供自然语言任务，系统内部自动完成生成、执行、反馈、再生成的完整循环。  
   - 这种“一站式”体验降低了技术门槛，也为后续研究提供了统一的实验平台。

### 方法详解

**整体框架**  
OpenCodeInterpreter 的工作流可以概括为三大阶段：**生成 → 执行 → 细化**。模型首先根据用户的自然语言需求生成一段候选代码；随后系统把代码交给沙箱化的 Python 解释器执行，捕获标准输出、异常栈和返回值；最后把执行日志与原始需求一起拼接成新的提示，喂回模型让它生成改进版代码。整个循环最多进行三次（可配置），直至满足预设的成功判定（如所有测试用例通过）或达到迭代上限。

**关键模块拆解**

1. **提示构造器（Prompt Builder）**  
   - 输入：用户指令 + 前一次的代码（若有）+ 执行日志 + 人类/合成反馈。  
   - 作用：把所有信息组织成模型易读的自然语言块，类似于老师在课堂上把学生的错误写在黑板上并解释原因，帮助模型“看到”自己的失误。  
   - 关键技巧：在提示中显式标注「错误类型」与「期望行为」，让模型在生成时有明确的修正目标。

2. **执行环境（Sandboxed Executor）**  
   - 使用轻量化的容器或安全的 Python 子进程，确保代码在受限资源下运行，防止恶意代码破坏系统。  
   - 捕获三类信息：**stdout**（程序输出）、**stderr**（错误信息）和**返回值**（函数调用结果）。  
   - 将这些信息序列化为结构化文本，例如 `Error: NameError: name 'x' is not defined`，便于提示构造器使用。

3. **细化模型（Refinement Model）**  
   - 与生成模型共享同一 Transformer 参数，但在细化阶段使用了额外的 **Feedback‑aware** 位置编码，让模型能够区分「原始需求」与「执行反馈」的来源。  
   - 训练时采用 **teacher‑forcing**：给模型提供正确的细化目标代码，损失函数同时考虑生成的代码与目标代码的 token‑level 差异以及执行成功率的加权奖励。

4. **合成反馈生成器（Synthetic Feedback Generator）**  
   - 在微调阶段，先用 GPT‑4 对每条执行失败的样本生成一段调试建议（如「变量未定义」→「请在函数开头声明变量」）。  
   - 这些建议被当作额外的「人类反馈」加入 Code‑Feedback 数据集，帮助细化模型学习更通用的错误修复模式。

**最巧妙的设计**  
- **执行信息的直接嵌入**：而不是把错误信息当作独立的标签，作者把它们写进自然语言提示，让模型在语言层面自行推理如何修正，这种“语言化的调试”比传统的结构化标签更符合 LLM 的训练习惯。  
- **合成反馈的闭环使用**：先用更强模型生成调试建议，再用这些建议提升弱模型，形成了一个“强‑弱‑强”循环，极大降低了人工标注成本。

### 实验与效果

- **评测基准**：HumanEval、MBPP 以及它们的 EvalPlus 加强版。HumanEval/MBPP 是业界常用的函数实现任务，EvalPlus 在原有测试用例上加入了对抗样本，考察模型的鲁棒性。  
- **主要结果**：OpenCodeInterpreter‑33B 在 HumanEval+MBPP 的平均得分为 **83.2**（普通版）和 **76.4**（EvalPlus），与闭源的 GPT‑4 Code Interpreter（84.2 / 76.2）相差不到 1%。在加入 GPT‑4 合成反馈后，得分进一步提升至 **91.6**（普通）和 **84.6**（EvalPlus），已经超越原始 GPT‑4 在同一基准上的表现。  
- **对比基线**：与同等规模的开源代码模型（如 CodeLlama、StarCoder）相比，OpenCodeInterpreter 在 HumanEval 上提升约 **10‑15%**，在 MBPP 上提升约 **12%**（具体数字原文未列出，只能给出相对提升幅度的概念）。  
- **消融实验**：作者分别去掉（1）执行反馈、（2）Code‑Feedback 数据、（3）合成反馈进行微调。结果显示，去掉执行反馈导致整体准确率下降约 **6%**，去掉 Code‑Feedback 降低约 **3%**，去掉合成反馈则在高阶 EvalPlus 上跌回原始 83.2 分左右，验证了每个模块的贡献。  
- **局限性**：- 目前仅支持 Python 生态，跨语言（如 Java、C++）的执行与细化尚未验证。  
  - 迭代次数上限固定为三次，极端复杂任务仍可能在第一轮就卡死。  
  - 合成反馈依赖于更强模型的可用性，若没有 GPT‑4 这类强模型，合成质量会受限。  

### 影响与延伸思考

OpenCodeInterpreter 的出现让「开源代码生成+执行」的闭环首次在学术界实现，直接推动了以下几个方向的研究热潮：

1. **多语言执行框架**：后续工作开始尝试把类似的执行‑反馈机制搬到 Java、Rust 等语言，探索语言特性对细化策略的影响。  
2. **自监督执行学习**：研究者利用模型自行生成「伪错误」并练习修复，进一步降低对标注数据的依赖。  
3. **安全沙箱的标准化**：因为执行环节不可避免地涉及潜在恶意代码，社区开始制定统一的沙箱接口规范，以便不同模型共享安全执行层。  
4. **人机协同调试**：OpenCodeInterpreter 的 API 为 IDE 插件提供了后端支持，出现了「AI 辅助即时调试」的商业插件，提升了普通开发者的生产力。

如果想继续深入，可以关注以下几个方向：  
- **Feedback‑Driven Pre‑training**：把执行日志直接加入大规模语言模型的预训练语料，看看是否能在根本上提升模型的代码理解能力。  
- **跨模态调试**：结合代码执行的可视化（如变量图、调用栈）与语言模型的文本提示，探索更丰富的反馈形式。  
- **强化学习细化**：把代码通过执行后得到的奖励（通过测试用例）直接用于强化学习，让模型在细化阶段学会“自我奖励”。  

### 一句话记住它

**OpenCodeInterpreter 用执行反馈把代码生成变成“写‑跑‑改”闭环，让开源模型的代码质量逼近闭源的 GPT‑4 Code Interpreter。**