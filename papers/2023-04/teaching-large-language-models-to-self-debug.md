# Teaching Large Language Models to Self-Debug

> **Date**：2023-04-11
> **arXiv**：https://arxiv.org/abs/2304.05128

## Abstract

Large language models (LLMs) have achieved impressive performance on code generation. However, for complex programming tasks, generating the correct solution in one go becomes challenging, thus some prior works have designed program repair approaches to improve code generation performance. In this work, we propose Self-Debugging, which teaches a large language model to debug its predicted program via few-shot demonstrations. In particular, we demonstrate that Self-Debugging can teach the large language model to perform rubber duck debugging; i.e., without any human feedback on the code correctness or error messages, the model is able to identify its mistakes by investigating the execution results and explaining the generated code in natural language. Self-Debugging achieves the state-of-the-art performance on several code generation benchmarks, including the Spider dataset for text-to-SQL generation, TransCoder for C++-to-Python translation, and MBPP for text-to-Python generation. On the Spider benchmark where there are no unit tests to verify the correctness of predictions, Self-Debugging with code explanation consistently improves the baseline by 2-3%, and improves the prediction accuracy on problems of the hardest level by 9%. On TransCoder and MBPP where unit tests are available, Self-Debugging improves the baseline accuracy by up to 12%. Meanwhile, by leveraging feedback messages and reusing failed predictions, Self-Debugging notably improves sample efficiency, and can match or outperform baseline models that generate more than 10x candidate programs.

---

# 教会大语言模型自我调试 论文详细解读

### 背景：这个问题为什么难？

代码生成已经是大语言模型（LLM）展示强大能力的典型场景，但当任务变得复杂——比如需要多行逻辑、跨语言转换或严格的 SQL 语义时，模型一次性输出正确代码的概率急剧下降。过去的做法大多是让模型一次生成多个候选，然后用单元测试或外部评估挑出正确的一个，这种“盲投”方式浪费算力，也难以突破瓶颈。更进一步的程序修复工作往往依赖人工标注的错误信息或专门的错误定位模型，成本高且难以推广。于是出现了一个核心需求：让模型自己发现并改正错误，而不必每一步都有人类介入。

### 关键概念速览
- **大语言模型（LLM）**：参数量在数十亿以上、通过海量文本预训练的生成式模型，能够理解自然语言指令并输出代码、解释等多模态内容。可以把它想象成“会说话的程序员”。
- **代码生成**：模型根据自然语言描述直接输出可执行代码，相当于把需求文档翻译成程序。
- **程序修复（Program Repair）**：在已有代码出错后，自动生成修改版以通过测试或满足约束，类似于“给出错误代码后让模型写补丁”。
- **Rubber Duck Debugging（橡皮鸭调试）**：程序员把代码解释给一只橡皮鸭听，以此理清思路并发现错误。这里指模型在没有外部错误提示的情况下，通过自我解释来定位 bug。
- **Few‑shot Demonstrations（少样本示例）**：在提示中加入几组“输入‑输出”示例，引导模型学习特定任务的模式。相当于给模型上几堂微型课堂。
- **Self‑Debugging（自我调试）**：本文提出的整体流程——模型先生成代码，执行后观察结果，再用自然语言解释代码和运行输出，最后在同一轮对话中给出改进版代码。整个循环不需要人工标注的错误信息。
- **Sample Efficiency（样本效率）**：在相同算力或生成次数下，模型能够得到更高质量的答案。这里指自我调试能用更少的候选程序达到或超过传统多候选策略的效果。
- **单元测试（Unit Test）**：针对代码的细粒度检查，用一组输入‑输出对验证程序是否符合预期。很多代码生成基准都会提供这些测试。

### 核心创新点
1. **把橡皮鸭调试搬进模型内部**  
   过去的调试方法要么依赖外部错误信息（编译器报错、单元测试），要么需要专门的错误定位模型。本文直接让 LLM 通过自我解释来发现错误——模型在执行代码后，把运行结果和代码逐行解释给自己听，然后在同一对话里指出不一致之处。这样做把“人类的思考过程”完整搬进了模型内部，省去了外部反馈环节。

2. **用少样本示例教会模型自我解释**  
   传统的 few‑shot 提示只展示“需求 → 代码”或“代码 → 修复”。这里的示例同时展示了“代码 → 解释 → 错误定位 → 修复”，让模型学会把代码转化为自然语言描述，再从描述中找出逻辑冲突。相比仅给出修复对，加入解释步骤显著提升了模型的自我审视能力。

3. **复用失败的候选程序**  
   在一次生成后，如果代码执行出错，模型不会直接抛弃，而是把这段失败的代码作为后续提示的一部分，让新一轮的生成“记住”之前的错误。相当于把每一次失败当作经验教训，提升了样本利用率。实验显示，这种复用让模型的表现可以匹配甚至超越需要生成 10 倍候选的基线。

4. **统一跨任务的自我调试框架**  
   之前的程序修复往往针对特定语言或特定评测（如仅针对 Python 单元测试）。本文的自我调试流程只依赖代码执行结果和自然语言解释，能够直接迁移到 SQL 生成、跨语言翻译等任务。实验在 Spider、TransCoder、MBPP 三大基准上均刷新了 SOTA，证明了方法的通用性。

### 方法详解
**整体思路**  
整个系统可以划分为四个阶段：① 需求 → 代码生成，② 代码执行并捕获输出，③ 模型自我解释并定位错误，④ 基于解释生成修复代码。这个循环可以重复多次，直到满足预设的停止条件（如通过所有单元测试或达到最大迭代次数）。

**步骤拆解**  

1. **初始生成**  
   - 输入：自然语言需求或 SQL 问题描述。  
   - 提示中加入 2‑3 条少样本示例，示例展示了“需求 → 代码”。  
   - LLM 输出第一版代码。

2. **执行与收集**  
   - 将生成的代码送入安全沙箱执行。  
   - 捕获返回值、异常信息或 SQL 查询结果。  
   - 如果任务没有单元测试（如 Spider），直接把查询结果作为“执行输出”。

3. **自我解释（Rubber Duck）**  
   - 构造新的提示，内容包括：  
     a) 之前的代码块；  
     b) 执行输出；  
     c) 2‑3 条示例，示例展示了“代码 + 执行结果 → 代码解释”。  
   - LLM 被要求用自然语言逐行解释代码的意图，并对照执行结果指出不一致或逻辑错误。  
   - 这一步相当于让模型“对自己说话”，把隐式的执行信息显性化。

4. **错误定位与修复**  
   - 在同一轮对话中，模型根据自己的解释生成一段“错误描述”。  
   - 再给出提示，让模型在保留原代码的基础上，输出修正后的代码。提示中会把“错误描述”作为上下文，帮助模型聚焦改动点。  
   - 修正后的代码再次送入执行环节，进入下一轮迭代。

**关键技巧**  
- **少样本示例的设计**：示例不仅展示正确答案，还展示错误案例的解释过程，使模型学会“先解释再改”。  
- **失败代码的复用**：每轮迭代把上一次的代码和解释一起放进提示，让模型“记住”之前的坑。  
- **统一的提示模板**：所有任务共用同一套模板，只在输入/输出格式上做微调，保证了跨任务的可迁移性。  
- **停止策略**：如果在一次迭代中解释没有发现错误（即解释与执行结果完全吻合），或者已经通过所有单元测试，就停止循环。

**最巧妙的地方**  
把“解释”这一步放在模型内部，而不是交给外部工具或人工审查。这样模型的“思考链”完整保留在一次前向传播中，既省时又让模型能够利用自己的语言理解能力进行自我纠错。

### 实验与效果
- **测试数据集**  
  - **Spider**（文本到 SQL）：没有单元测试，只能通过查询结果判断对错。  
  - **TransCoder**（C++ → Python 翻译）：提供编译后运行的单元测试。  
  - **MBPP**（文本到 Python 小程序）：同样配有单元测试。

- **对比基线**  
  - 传统的多候选生成 + 直接测试（如 CodeT5、Codex）。  
  - 现有的程序修复方法（如 PLBART‑Repair）。

- **主要提升**  
  - 在 Spider 上，自我调试把基线提升了 2‑3%，在最难的难度层级（Hard）提升了约 9%。  
  - 在 TransCoder 与 MBPP 上，最高提升达 12%。  
  - 与需要生成 10 倍候选的基线相比，使用自我调试只需 1‑2 倍候选即可匹配或超越其表现，说明样本效率显著提升。

- **消融实验**  
  - 去掉解释步骤，直接让模型基于执行结果生成修复，性能下降约 4‑5%。  
  - 不复用失败代码，改为每轮重新生成，提升幅度减半。  
  - 少样本示例数量从 2 增加到 5，提升有限，说明核心收益来自解释与复用，而非示例数量。

- **局限性**  
  - 依赖可靠的执行环境，若沙箱不支持某些库或语言特性，调试过程会中断。  
  - 对于非确定性错误（如随机种子导致的间歇性失败），模型的解释可能不稳定。  
  - 论文未详细说明在极大代码规模（上千行）时的迭代成本，实际部署时可能需要额外的截断策略。

### 影响与延伸思考
这篇工作把“自我反思”从语言模型的对话层面提升到代码层面，开启了“模型自我调试”的新方向。随后的研究陆续出现：  
- **Self‑Refine** 系列把类似的自我解释用于数学推理和常识问答。  
- **Tool‑Augmented LLM** 把调试器、单元测试框架直接嵌入模型的工具库，进一步降低对手工提示的依赖。  
- **Iterative Prompting** 方向把多轮交互视为强化学习的轨迹，尝试用奖励信号自动决定何时停止迭代。

想继续深入，可以关注以下几个方向：  
1. 把更专业的调试信息（如栈追踪、内存泄漏报告）加入模型的解释输入，提升对低层错误的感知。  
2. 将自我调试与检索增强（Retrieval‑Augmented Generation）结合，让模型在解释阶段查阅外部文档或代码库。  
3. 探索跨语言的统一调试框架，尤其是对 DSL（领域特定语言）或混合语言项目的适配。

### 一句话记住它
让大语言模型先把代码“讲给自己听”，再在同一轮对话里自行改错，省去外部反馈即可大幅提升代码生成质量。