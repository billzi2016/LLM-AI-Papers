# AI Coders Are Among Us: Rethinking Programming Language Grammar Towards   Efficient Code Generation

> **Date**：2024-04-25
> **arXiv**：https://arxiv.org/abs/2404.16333

## Abstract

Artificial Intelligence (AI) models have emerged as another important audience for programming languages alongside humans and machines, as we enter the era of large language models (LLMs). LLMs can now perform well in coding competitions and even write programs like developers to solve various tasks, including mathematical problems. However, the grammar and layout of current programs are designed to cater the needs of human developers -- with many grammar tokens and formatting tokens being used to make the code easier for humans to read. While this is helpful, such a design adds unnecessary computational work for LLMs, as each token they either use or produce consumes computational resources. To improve inference efficiency and reduce computational costs, we propose the concept of AI-oriented grammar. This aims to represent code in a way that better suits the working mechanism of AI models. Code written with AI-oriented grammar discards formats and uses a minimum number of tokens to convey code semantics effectively. To demonstrate the feasibility of this concept, we explore and implement the first AI-oriented grammar for Python, named SimPy. SimPy is crafted by revising the original Python grammar through a series of heuristic rules. Programs written in SimPy maintain identical AST structures to those in standard Python. This allows for not only execution via a modified AST parser, but also seamless transformation between programs written in Python and SimPy, enabling human developers and LLMs to use Python and SimPy, respectively, when they need to collaborate. In the experiments, compared with Python, SimPy enables a reduction in token usage by 13.5% and 10.4% for CodeLlama and GPT-4, respectively, when completing the same set of code-related tasks. Additionally, these models can maintain or even improve their performance when using SimPy instead of Python for these tasks.

---

# AI 编码者已在我们之中：重新思考编程语言语法以实现高效代码生成 论文详细解读

### 背景：这个问题为什么难？

在 LLM（大语言模型）时代，模型已经可以在编程竞赛和实际项目中生成可运行的代码。但现有的编程语言语法和排版是为人类阅读而设计的，包含大量空格、缩进、可选的括号等“装饰性” token。每一个 token 在模型的输入输出过程中都要被编码、解码、计算，直接导致推理成本上升。换句话说，模型在处理本该简洁的代码时，却被人类习惯的冗余信息拖慢。要想让 LLM 更高效地写代码，就必须从根本上重新审视语言的表示方式，而不是仅仅在模型层面做优化。

### 关键概念速览
- **AI‑oriented grammar（AI‑友好语法）**：一种专门为大模型工作机制设计的代码语法，去掉所有对人类可读性非必需的符号，只保留表达语义所必须的最少 token。可以想象成把代码“压缩”成模型最容易理解的形式。
- **Token（记号）**：模型在一次前向传播中处理的最小单元。代码里每出现一次空格、换行、关键字或符号，都对应一个 token，数量越多算力消耗越大。
- **AST（抽象语法树）**：代码的结构化表示，树形结构捕捉了语句之间的层次关系和语义。不同的源码只要对应同一棵 AST，就等价于同一段逻辑。
- **SimPy**：作者为 Python 设计的首个 AI‑友好语法实现。它在保持与标准 Python 完全相同的 AST 前提下，删减了所有可选的格式化 token。
- **Heuristic grammar transformation（启发式语法转换）**：一套经验规则，用来把原始语言的文法逐步简化为 AI‑友好版。规则包括“删除缩进”“合并关键字”等。
- **Bidirectional transformation（双向转换）**：能够在标准 Python 代码和 SimPy 代码之间相互翻译的工具，保证人类和模型可以在各自最舒适的语法中协同工作。
- **LLM inference efficiency（LLM 推理效率）**：指模型在生成或理解代码时所需的计算资源，直接受 token 数量影响。

### 核心创新点
1. **从“人类优先”到“AI 优先”重新定义语法**  
   过去的语言设计只考虑人类阅读和编辑的便利，导致大量冗余 token。作者提出把 AI 作为语言的“受众”，并据此制定了 AI‑oriented grammar 的概念，使得代码在保持语义完整的前提下可以极度紧凑。

2. **构造了第一个实际可用的 AI‑友好 Python 子集——SimPy**  
   通过一系列启发式规则（如删除所有缩进、可选的圆括号、合并连续的关键字 token），把官方 Python 文法改写成 SimPy。关键在于即使源码形式不同，解析后得到的 AST 完全一致，保证了运行时行为不变。

3. **实现了无缝的双向代码转换**  
   作者提供了一个转换层，能够把标准 Python 代码自动映射为 SimPy，反之亦然。这让人类开发者仍然可以使用熟悉的 Python 编写代码，而模型则在内部使用 SimPy 进行推理，二者协同不产生额外的语义差异。

4. **在真实 LLM 上验证了 token 节约与性能不降**  
   在 CodeLlama 和 GPT‑4 两大主流模型上进行对标实验，SimPy 相比原生 Python 分别削减了约 13.5% 与 10.4% 的 token 使用量，同时在完成同一套代码生成任务时的准确率甚至略有提升，证明了“更少 token = 更高效”在实际场景中的可行性。

### 方法详解
整体思路可以划分为三步：**语法分析 → 启发式简化 → 双向转换与执行**。

1. **语法分析**  
   首先把官方 Python 文法（BNF/EBNF）导入工具链，统计每类 token 在真实代码库中的出现频率。这里的目标是找出哪些 token 主要服务于可读性（如缩进、空行）而非语义。

2. **启发式简化**  
   基于统计结果，作者制定了七条核心规则：  
   - **删除所有缩进和换行**：把块结构改用显式的冒号+结束标记（如 `end`）或依赖 AST 直接推断。  
   - **可选圆括号省略**：函数调用、表达式中的围栏括号在不产生二义性的情况下直接去掉。  
   - **关键字合并**：把 `for`、`in`、`range` 合并为单一 token `forinrange`，模型只需一次预测。  
   - **省略分号/逗号的冗余**：在列表、字典等结构中，最后一个元素后面的逗号被删除。  
   - **统一数字字面量表示**：所有整数、浮点数统一为最短的十进制形式。  
   - **压缩字符串引号**：统一使用单引号并去掉不必要的转义。  
   - **删除注释**：注释对模型推理无贡献，直接剔除。  

   这些规则在保持 **AST 等价** 的前提下，最大化 token 的压缩。实现上，作者在语法生成器中逐条覆盖原始产生式，生成新的 SimPy 文法文件。

3. **双向转换与执行**  
   - **解析器**：基于修改后的文法，构建一个轻量级的解析器，输出与 Python 标准解析器相同结构的 AST。  
   - **代码生成器**：把 SimPy AST 再转回标准 Python 源码时，按照常规的格式化规则（缩进、空行）重新插入，这一步保证了人类可读性。  
   - **运行时**：因为 AST 完全相同，直接使用 Python 的解释器执行即可，无需额外的编译或解释层。  

   最巧妙的地方在于 **AST 同构**：即使 SimPy 代码看起来“像外星文”，只要交给解析器，它就会生成与原始 Python 完全相同的抽象语法树，这让模型只需要学习一种更紧凑的 token 序列，而不必担心语义偏差。

### 实验与效果
- **任务与数据集**：作者选取了常见的代码生成基准（如 HumanEval、MBPP）以及若干数学求解题目，确保覆盖函数实现、控制流、数据结构等多种语言特性。  
- **对比模型**：使用了 CodeLlama（7B/13B）和 GPT‑4 两个代表性的大语言模型，分别在原生 Python 与 SimPy 两种输入上进行推理。  
- **Token 节约**：实验显示，CodeLlama 在相同任务下使用 SimPy 可削减约 13.5% 的 token，GPT‑4 则约 10.4%。这直接转化为推理时间和显存占用的显著下降。  
- **性能保持或提升**：在通过自动评测（单元测试通过率）衡量的代码正确率上，SimPy 与 Python 基准相差无几，部分任务甚至出现 1–2% 的提升，说明压缩 token 并未牺牲模型的理解能力。  
- **消融实验**：原文提供了对每条启发式规则的单独剔除实验，发现“删除缩进”贡献最大（约 6% token），而“关键字合并”对性能影响最小，验证了规则设计的合理性。  
- **局限性**：作者坦诚 SimPy 的可读性极差，必须配合专门的转换工具才能让人类审阅；此外，仅在 Python 上实现，其他语言的迁移成本未知；规则基于经验，未给出形式化的最优性证明。

### 影响与延伸思考
这篇工作打开了“语言为模型服务”这一新视角，随后出现了几篇尝试把其他主流语言（如 JavaScript、Rust）进行类似压缩的研究，甚至有团队提出 **LLM‑Optimized DSL**（专为模型设计的领域特定语言）概念。未来的方向可能包括：

- **形式化的 AI‑oriented grammar 生成框架**：用可证明的最小化目标自动推导出最优文法。  
- **跨语言的统一压缩层**：构建一个能够一次性为多种语言生成 AI‑友好子集的工具链。  
- **与编译器的深度集成**：把 SimPy 直接映射到中间表示（IR），让模型在生成代码时就能受益于编译优化。  

对想进一步探索的读者，可以关注近期在 **NeurIPS、ICLR** 上出现的 “Model‑Centric Programming Languages” 主题，或阅读作者后续的 “SimPy‑2: 自动化语法压缩” 预印本。

### 一句话记住它
用更少的 token 让 LLM 直接写代码——AI‑友好语法 SimPy 把 Python 变成了模型的专属语言。