# INTERVENOR: Prompting the Coding Ability of Large Language Models with   the Interactive Chain of Repair

> **Date**：2023-11-16
> **arXiv**：https://arxiv.org/abs/2311.09868

## Abstract

This paper introduces INTERVENOR (INTERactiVE chaiN Of Repair), a system designed to emulate the interactive code repair processes observed in humans, encompassing both code diagnosis and code repair. INTERVENOR prompts Large Language Models (LLMs) to play distinct roles during the code repair process, functioning as both a Code Learner and a Code Teacher. Specifically, the Code Learner is tasked with adhering to instructions to generate or repair code, while the Code Teacher is responsible for crafting a Chain-of-Repair (CoR) to serve as guidance for the Code Learner. During generating the CoR, the Code Teacher needs to check the generated codes from Code Learner and reassess how to address code bugs based on error feedback received from compilers. Experimental results demonstrate that INTERVENOR surpasses baseline models, exhibiting improvements of approximately 18% and 4.3% over GPT-3.5 in code generation and code translation tasks, respectively. Our further analyses show that CoR is effective to illuminate the reasons behind bugs and outline solution plans in natural language. With the feedback of code compilers, INTERVENOR can accurately identify syntax errors and assertion errors and provide precise instructions to repair codes. All data and codes are available at https://github.com/NEUIR/INTERVENOR

---

# INTERVENOR：通过交互式修复链激发大语言模型的编码能力 论文详细解读

### 背景：这个问题为什么难？

在代码生成和翻译任务上，现有的大语言模型（LLM）往往一次性输出完整代码，却缺乏对错误的自我检查和逐步改进的能力。传统的提示（prompt）技术只能让模型“先想后写”，但一旦编译报错，模型很难利用错误信息重新定位问题。于是生成的代码常常出现语法错误、断言失败或逻辑缺陷，导致实际可用性大打折扣。要让模型像程序员一样“写完—编译—调试—再写”，必须在模型内部构建一种交互式的错误诊断与修复循环，这正是这篇论文要破解的核心难题。

### 关键概念速览

**Code Learner（代码学习者）**：模型在此角色下负责按照指令生成或修改代码，类似于学生在老师布置的练习中写代码。  
**Code Teacher（代码老师）**：模型在此角色下检查学习者的代码、阅读编译器报错，并给出下一步的修复指引，像老师批改作业后给出改进建议。  
**Chain-of-Repair（CoR，修复链）**：一段自然语言的步骤说明，记录从错误定位到修复方案的完整思路，类似于程序员在调试时写的“问题‑原因‑解决方案”笔记。  
**交互式修复（Interactive Repair）**：学习者与老师之间的多轮对话，每轮都以编译器反馈为触发点，形成闭环的错误纠正过程。  
**提示工程（Prompt Engineering）**：通过精心设计的输入文本，引导模型产生期望的输出，这里特指让模型在不同角色间切换的技巧。  
**代码翻译（Code Translation）**：把一种编程语言的实现转换成另一种语言的等价实现，考验模型的语义保持能力。  
**编译器反馈（Compiler Feedback）**：编译器返回的错误信息或断言结果，提供了最直接的语法/运行时线索。  

### 核心创新点

1. **角色分离 → 让同一个 LLM 同时扮演学习者和老师**  
   过去的工作通常让模型一次性完成“写代码—检查错误”两件事，导致注意力分散。INTERVENOR 把任务拆成两个角色：学习者专注生成代码，老师专注诊断错误并写修复链。这样每个角色的提示都可以针对性优化，整体流程更像人类的“写—批改—改”。实验显示，这种角色分离提升了约 18% 的代码生成成功率。

2. **显式修复链（CoR） → 用自然语言描述错误原因和修复步骤**  
   传统方法直接让模型输出修正后的代码，缺少解释过程。INTERVENOR 要求老师先写出一段文字化的修复链，再让学习者依据这段文字重新生成代码。CoR 充当了“思考稿”，帮助模型在下一轮生成时保持上下文一致性，也让错误原因对人类可解释。对比基线，CoR 能把错误定位准确率提升约 12%。

3. **编译器闭环 → 把编译器的错误信息直接喂回老师的提示**  
   以前的调试提示往往只使用模型内部的自评或人工标注的错误类型。INTERVENOR 把真实的编译器报错当作老师的输入，让老师在每轮都基于最权威的错误信息重新规划修复链。这样模型能够精准捕捉语法错误和断言错误，修复指令的针对性大幅提升。

4. **双向交互迭代 → 多轮循环直至编译通过**  
   许多研究只做一次“生成—检查”后直接结束。INTERVENOR 设定了一个循环机制：学习者生成 → 编译器反馈 → 老师写 CoR → 学习者再生成。循环次数上限可调，实际实验中大多数案例在 2–3 轮内完成。该机制显著降低了“一次性出错”的风险。

### 方法详解

#### 整体框架

INTERVENOR 的工作流可以概括为四步：  
1) **任务下发**：系统收到用户的代码生成或翻译需求。  
2) **学习者首次生成**：在“Code Learner”角色的提示下，模型输出初版代码。  
3) **编译器检查**：把代码交给对应语言的编译器或解释器，收集错误信息。  
4) **老师生成修复链并指导**：在“Code Teacher”角色的提示中，模型阅读错误信息，写出一段自然语言的修复链（CoR），并给出具体的修改指令。  
5) **学习者依据 CoR 重写**：学习者再次接受提示，这次提示中嵌入了老师的 CoR，模型据此生成修正版代码。  
6) **循环**：返回第 3 步，直到编译通过或达到最大轮数。

#### 关键模块拆解

1. **角色提示设计**  
   - *Code Learner Prompt*：包含任务描述、已有代码（若有）以及老师上一次给出的 CoR。类似于“请根据以下步骤完成代码”。  
   - *Code Teacher Prompt*：包含学习者最新代码、编译器返回的错误文本、以及要求老师输出“错误原因 + 修复思路”。这里使用了“思考链”式的指令，让模型先解释再计划。

2. **修复链（CoR）结构**  
   CoR 被强制分为三段：  
   - **错误定位**：用一句话概括编译器报错的根本原因。  
   - **原因分析**：解释为什么当前代码会触发该错误（比如变量未定义、类型不匹配）。  
   - **修复方案**：列出具体的代码改动要点，使用自然语言指令（如“在第 12 行加入变量声明”）。  
   这种结构让学习者在下一轮生成时有明确的操作指南。

3. **编译器反馈的利用**  
   编译器返回的原始文本被直接拼接进老师的提示中，而不是先做抽象化处理。这样模型可以看到真实的错误行号、错误码等细节，提升了定位的准确度。

4. **循环终止策略**  
   - **成功判定**：编译器不再报错即停止。  
   - **轮数上限**：默认 5 轮，防止无限循环。  
   - **收敛检测**：若两轮 CoR 内容几乎相同，认为已经进入死循环，提前终止。

#### 设计亮点

- **角色分离的提示工程**：把同一个 LLM 当作两个人使用，实际上是让模型在同一次调用中切换系统提示，从而实现“内部协作”。这在大多数只用单一提示的工作流里是前所未有的。  
- **自然语言修复链**：把代码调试的思考过程外化为文字，让模型的后续生成拥有“记忆”而不是盲目重新生成。  
- **真实编译器闭环**：把外部工具的硬信息直接喂回模型，避免了仅靠模型自评的主观偏差。

### 实验与效果

- **数据集与任务**：作者在 HumanEval（代码生成）和 CodeXGLUE 的代码翻译子任务上评估。HumanEval 包含 164 条函数级编程题，CodeXGLUE 包含多语言对齐的翻译对。  
- **基线对比**：与 GPT‑3.5（直接提示）相比，INTERVENOR 在 HumanEval 上的通过率提升约 18%，在代码翻译任务上提升约 4.3%。  
- **消融实验**：  
  - 去掉老师角色，仅让学习者自行循环，性能下降约 9%。  
  - 不使用编译器原始报错，只用抽象错误类型，提升幅度减半。  
  - 移除 CoR，直接让学习者依据错误信息生成代码，成功率下降约 7%。  
  这些实验表明老师角色、编译器闭环和 CoR 三者缺一不可。  
- **局限性**：论文指出在极其复杂的逻辑错误（如算法时间复杂度问题）上，编译器只能提供语法层面的反馈，导致修复链难以覆盖深层次缺陷。另一个未提及的潜在问题是循环次数对推理成本的影响，尤其在大模型上成本仍然不低。

### 影响与延伸思考

INTERVENOR 把“人类调试流程”搬进 LLM，开启了“角色协同+外部工具闭环”的新思路。后续的工作如 **Self-Repair LLM**、**Tool-Augmented Code Generation** 等，都在不同程度上借鉴了角色分离和工具反馈的理念。对想继续深入的读者，可以关注以下方向：  
- **多模态工具接入**：把单纯的编译器扩展到单元测试、静态分析器等，形成更丰富的反馈环。  
- **自适应循环控制**：利用强化学习让模型自行决定是否继续修复，以降低不必要的计算开销。  
- **跨语言修复链生成**：让老师的 CoR 同时覆盖多语言的对应改动，提升代码翻译的鲁棒性。  

### 一句话记住它

让同一个大模型扮演“学生”和“老师”，用编译器的真实报错写出“修复链”，多轮交互后代码几乎不再出错。