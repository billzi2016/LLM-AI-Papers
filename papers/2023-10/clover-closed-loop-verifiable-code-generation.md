# Clover: Closed-Loop Verifiable Code Generation

> **Date**：2023-10-26
> **arXiv**：https://arxiv.org/abs/2310.17807

## Abstract

The use of large language models for code generation is a rapidly growing trend in software development. However, without effective methods for ensuring the correctness of generated code, this trend could lead to undesirable outcomes. In this paper, we introduce a new approach for addressing this challenge: the Clover paradigm, short for Closed-Loop Verifiable Code Generation, which uses consistency checking to provide a strong filter for incorrect code. Clover performs consistency checks among code, docstrings, and formal annotations. The checker is implemented using a novel integration of formal verification tools and large language models. We provide a theoretical analysis to support our thesis that Clover should be effective at consistency checking. We also empirically investigate its performance on a hand-designed dataset (CloverBench) featuring annotated Dafny programs at a textbook level of difficulty. Experimental results show that for this dataset: (i) LLMs are reasonably successful at automatically generating formal specifications; and (ii) our consistency checker achieves a promising acceptance rate (up to 87%) for correct instances while maintaining zero tolerance for adversarial incorrect ones (no false positives). Clover also discovered 6 incorrect programs in the existing human-written dataset MBPP-DFY-50.

---

# Clover：闭环可验证代码生成 论文详细解读

### 背景：这个问题为什么难？

在软件开发中，使用大语言模型（LLM）自动生成代码已经变得很流行，但模型往往只能“猜”出代码，缺乏对正确性的保证。传统的代码生成流水线只靠模型输出，若生成的实现与需求不匹配，就会产生难以发现的 bug。已有的后置测试或人工审查成本高，而且测试覆盖率有限，无法保证形式化属性（如不变式、前置后置条件）被满足。于是，如何在生成阶段就对代码进行可靠的正确性过滤，成为阻碍 LLM 实际落地的关键瓶颈。

### 关键概念速览
- **闭环（Closed‑Loop）**：指生成、验证、反馈三者形成一个闭合回路，类似写作后先自检再改稿，确保每一次输出都经过检查再决定是否接受。  
- **可验证代码生成（Verifiable Code Generation）**：在生成代码的同时，产生可以被形式化工具验证的规格（如 pre/post 条件），让机器能够自动证明代码满足这些规格。  
- **一致性检查（Consistency Checking）**：把代码、docstring（自然语言说明）和形式化注解三者进行交叉比对，确保它们描述的是同一件事。想象三位证人分别说同一件事，只有三者说法一致时才可信。  
- **形式化验证工具**：如 Dafny、Boogie 等能够把带有规格的代码转化为逻辑公式并自动求解，以判断规格是否被满足。  
- **CloverBench**：作者手工构造的基准，包含带有完整 Dafny 规格的教材级程序，用来评估生成与一致性检查的整体效果。  
- **Adversarial Incorrect Instances**：刻意构造的错误示例，用来检验过滤器是否会误放过错误代码（即假阳性）。  

### 核心创新点
1. **从单向生成 → 闭环生成**：传统方法只让 LLM 输出代码，Clover 在此基础上加入了“生成‑验证‑反馈”三步循环。生成后立即交给形式化工具和 LLM 共同做一致性检查，只有通过的才被接受。这样把错误代码在生成阶段就剔除，显著提升了交付质量。  
2. **代码‑Docstring‑规格三元一致性检查 → 多模态过滤**：以前的验证只看代码与规格的对应关系，Clover 进一步把自然语言的 docstring 拉进来，利用 LLM 的语言理解能力把 docstring 翻译成形式化断言，再与代码的实际行为比对。多模态交叉验证让错误更难逃脱。  
3. **形式化工具 + LLM 的新型集成 → 互补优势**：形式化工具擅长严谨的逻辑求解，LLM 擅长把自然语言转成规格。Clover 把两者串成流水线：LLM 先生成代码和规格，形式化工具尝试证明，若失败则把失败信息喂回 LLM 让其“自我纠正”。这种人机协同的闭环在之前的工作中少有尝试。  
4. **手工基准 CloverBench → 可复现评估**：作者专门设计了一个包含教材级难度的 Dafny 程序集合，提供统一的生成‑验证‑评估平台，弥补了公开数据集缺少形式化注解的空白。

### 方法详解
**整体框架**  
Clover 的工作流可以概括为四步：  
1) **需求输入**：用户提供自然语言描述或已有的 docstring。  
2) **LLM 生成**：同一个大语言模型一次性输出（a）实现代码、（b）对应的 docstring（若原始输入缺失）以及（c）形式化注解（pre/post 条件、循环不变式等）。  
3) **一致性检查**：把三者送入两路验证器——（i）LLM‑驱动的语义对齐模块把 docstring 翻译成形式化断言并与 LLM 生成的规格比对；（ii）形式化验证工具（如 Dafny）对代码+规格进行自动证明。  
4) **闭环反馈**：如果任一环节发现不一致或证明失败，系统会把错误信息（如“post 条件未被满足”）回写给 LLM，让它重新生成或修正；只有全部通过时，代码才被标记为“接受”。  

**关键模块拆解**  
- **生成模块**：使用指令式提示（prompt）让模型一次性输出三段文本。提示中明确要求代码、docstring、规格相互对应，类似让学生在答题时同时写出解题步骤、答案和证明。  
- **语言‑规格对齐**：把 docstring 通过第二次 LLM 调用转化为形式化断言。这里的技巧是让模型先生成“自然语言 → 逻辑表达式”的中间表述，再用解析器把它变成 Dafny 可识别的语法。  
- **形式化验证**：把代码和所有断言喂入 Dafny 编译器。Dafny 会尝试自动证明每个 pre/post 条件，如果成功则返回“verified”，否则返回未通过的具体断言。  
- **反馈生成**：当验证失败时，系统提取失败的断言和对应的代码位置，构造一条错误提示（例如“循环不变式在第 12 行被破坏”），再把这条提示拼进下一轮的 LLM 输入，让模型在同一上下文中进行局部修复。  

**最巧妙的设计**  
- **双向验证**：既用 LLM 检查自然语言与形式化规格的一致性，又用形式化工具检查代码与规格的匹配。两者相互补强，单靠任意一方都容易漏掉细微错误。  
- **零容忍的错误过滤**：作者在实验中设置了“adversarial incorrect instances”，并保证检查器对这些实例永不误判（即没有假阳性），这在实际部署中相当关键，因为误放错代码的代价往往比多次迭代更高。  

### 实验与效果
- **数据集**：主要在作者自建的 CloverBench 上评估，数据集包含数十个教材级 Dafny 程序，每个程序都有完整的 docstring 与形式化规格。作者还在公开的 MBPP‑DFY‑50（50 条人写的 Dafny 题目）上做了额外验证。  
- **基线对比**：与“直接使用 LLM 生成代码并跳过验证”的裸跑方式相比，Clover 在正确实例的接受率上提升至 **87%**（原始裸跑的接受率未给出，但显著低于此）。在 adversarial 错误集上实现 **零假阳性**，即所有错误都被成功拦截。  
- **发现的实际错误**：在 MBPP‑DFY‑50 中，Clover 自动检测出 **6 条** 人工编写的错误程序，说明即使是专家写的代码也会受益于闭环检查。  
- **消融实验**：论文中对每个模块进行消融（如仅使用形式化验证、仅使用 docstring‑规格对齐），结果显示去掉任意一环都会导致接受率显著下降，验证了“双向一致性检查”是性能提升的关键因素。  
- **局限性**：实验仅限于 Dafny 这类内置形式化验证的语言，未在更主流的 Python/Java 环境中验证；此外，生成规格的质量仍受 LLM 能力限制，复杂业务逻辑的规格仍可能不完整。作者也承认在大规模真实项目上仍缺乏评估。  

### 影响与延伸思考
Clover 提出了“闭环”思路后，后续不少工作开始探索 **生成‑验证‑迭代** 的流水线，例如在 Python 中结合 Pytype、mypy 的自动类型检查，或在 JavaScript 中引入 Flow/TypeScript 的类型推断作为反馈。还有研究把 **对抗生成**（adversarial code generation）与 Clover 的零容忍过滤结合，进一步提升安全性。对想深入的读者，可以关注以下方向：① 将闭环框架迁移到主流语言的生态（如 VSCode 插件化）；② 探索更高效的规格自动抽取技术，尤其是从大型代码库中学习不变式；③ 研究人机协同的交互界面，让开发者在闭环中提供少量关键提示而不是全自动。  

### 一句话记住它
**Clover 用代码、文档和形式化规格的三重一致性检查，把大模型生成的代码在“写完”之前就过滤掉所有错误，实现了零假阳性的闭环验证。**