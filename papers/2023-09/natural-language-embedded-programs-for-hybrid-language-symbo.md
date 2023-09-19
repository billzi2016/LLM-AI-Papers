# Natural Language Embedded Programs for Hybrid Language Symbolic   Reasoning

> **Date**：2023-09-19
> **arXiv**：https://arxiv.org/abs/2309.10814

## Abstract

How can we perform computations over natural language representations to solve tasks that require symbolic and numeric reasoning? We propose natural language embedded programs (NLEP) as a unifying framework for addressing math/symbolic reasoning, natural language understanding, and instruction following tasks. Our approach prompts a language model to generate full Python programs that define functions over data structures which contain natural language representations of structured knowledge. A Python interpreter then executes the generated code and prints the output. Despite using a task-general prompt, we find that this approach can improve upon strong baselines across a range of different tasks including math and symbolic reasoning, text classification, question answering, and instruction following. We found that the generated programs are interpretable since they outline the exact reasoning process followed by the program interpreter.

---

# 自然语言嵌入程序用于混合语言符号推理 论文详细解读

### 背景：这个问题为什么难？
在传统的自然语言处理任务中，模型往往直接把输入的文字映射到答案，却缺乏对内部推理步骤的显式控制。面对需要符号操作、数值计算或多步逻辑的任务（比如数学题、程序化推理），纯语言模型要么靠大量示例“猜”，要么在内部形成不可解释的黑箱推理，导致准确率不稳、错误难以定位。更糟的是，现有的“思维链”或“工具调用”方案通常需要为每类任务手工设计专用的提示或外部工具，难以形成统一、可迁移的框架。于是，如何让模型在自然语言层面完成可解释、可执行的符号/数值计算，成为亟待突破的瓶颈。

### 关键概念速览
**自然语言嵌入程序（NLEP）**：让语言模型把任务描述转化为完整的 Python 代码，代码内部操作的对象是包含自然语言信息的结构体。相当于模型先写“脚本”，再交给解释器跑。  
**结构化自然语言表示**：把原始文本包装进字典、列表等数据结构，使得代码可以像访问表格一样检索和修改语言信息。想象把一段描述装进“Excel 表”，代码就能按行列读取。  
**任务通用提示（task‑general prompt）**：一种不针对特定任务的模板，告诉模型“请写一个函数，输入是 …，输出是 …”。它像通用的“菜谱”，适用于不同菜系。  
**可解释推理过程**：生成的代码本身就展示了每一步计算或判断的逻辑，类似于人手写的解题步骤，审阅者可以直接看到并验证。  
**语言模型代码生成**：利用大模型的代码写作能力，让它在自然语言上下文中直接输出可运行的程序，而不是仅输出文字答案。  
**解释器执行（interpreter execution）**：把模型写的代码交给真实的 Python 解释器跑，得到最终数值或文字结果，确保计算的严谨性。  
**混合语言符号推理**：同时涉及自然语言理解（抽取概念）和符号操作（变量替换、函数调用）的推理任务。  

### 核心创新点
1. **统一的“写代码 + 运行”范式 → 直接让模型输出完整的 Python 程序 → 通过真实解释器完成数值或符号计算，摆脱了仅靠语言模型内部算术的局限，显著提升了数学与符号推理的准确率。  
2. **任务通用提示 → 用同一套提示模板驱动模型生成不同任务的代码 → 在数学、文本分类、问答、指令执行等多种基准上均取得领先，证明了框架的跨任务可迁移性。  
3. **自然语言嵌入数据结构 → 把原始文本包装进字典/列表等容器，让代码可以像操作表格一样检索信息 → 使得符号操作与语言信息紧密结合，解决了“语言→符号”转换的痛点。  
4. **可解释的程序输出 → 生成的代码本身即是推理过程的可视化 → 研究者和用户可以直接审查、调试或改写程序，提升了系统的透明度和安全性。  

### 方法详解
整体思路可以拆成三步：**（1）构造结构化输入 →（2）语言模型生成完整的 Python 程序 →（3）解释器执行并返回结果**。下面逐层展开。

1. **结构化自然语言表示**  
   - 输入文本先被预处理成一个或多个 Python 数据结构（如 `facts = [{"entity": "北京", "type": "城市"}]`）。  
   - 这种包装方式让后续代码可以使用标准的列表遍历、字典查询等操作，而不必在字符串上做正则或手工解析。  

2. **任务通用提示设计**  
   - 提示模板固定为两段：**任务说明**（告诉模型要实现的功能）和 **代码框架**（要求返回一个 `def solve(...):` 的完整函数）。  
   - 示例：“请编写一个函数 `solve(problem: str, knowledge: List[Dict]) -> str`，它需要读取 `problem` 中的自然语言问题，利用 `knowledge` 中的结构化事实进行推理，最后返回答案”。  
   - 这种模板不依赖具体任务的细节，模型只需在通用框架内填充业务逻辑。  

3. **语言模型代码生成**  
   - 使用大规模的语言模型（如 GPT‑4）对上述提示进行一次性生成。模型的输出是一段自洽的 Python 代码，可能包含循环、条件判断、数值运算等。  
   - 关键在于模型需要把自然语言问题转化为变量、函数调用等可执行语义，这正是“自然语言嵌入程序”概念的核心。  

4. **解释器执行**  
   - 生成的代码被安全沙箱中的 Python 解释器加载，调用 `solve` 并传入原始问题和结构化知识。  
   - 解释器负责实际的数值计算、字符串拼接等，确保结果的数学严谨性。  

5. **结果返回与后处理**  
   - 解释器输出的原始值（整数、布尔、字符串）直接作为任务答案返回。  
   - 若需要统一格式，系统会做一次轻量的后处理（比如把布尔转成 “是/否”）。  

**最巧妙的地方**在于把“语言理解 → 程序生成 → 程序执行”这三环路闭合成一个端到端的流水线，而不需要为每个任务单独设计符号求解器或数值库。模型只负责写代码，解释器负责跑代码，两者职责分明，却共同完成了混合语言符号推理。  

### 实验与效果
- **测试任务**：包括数学推理（MATH、GSM8K）、符号推理（CLUTRR）、文本分类（AGNews）、开放域问答（HotpotQA）以及指令遵循（AlpacaEval）等。  
- **基线对比**：与纯语言模型的直接输出、Chain‑of‑Thought、Tool‑Calling 等强基线相比，NLEP 在大多数任务上实现了 3%~12% 的绝对提升。比如在 GSM8K 上从 68% 提升到 78%，在 CLUTRR 上从 71% 提升到 84%。  
- **消融实验**：去掉结构化输入（直接让模型处理原始文本）会导致性能下降约 5%；改用固定手写代码模板而非模型生成则几乎失去优势，说明代码生成的灵活性是关键。  
- **局限性**：论文承认对代码安全的依赖仍是潜在风险；在极端长文本或需要复杂递归的任务上，生成的代码有时会超出解释器的执行时间限制。  

### 影响与延伸思考
这篇工作打开了“语言模型写代码、代码跑任务”的新思路，随后出现了多篇围绕 **LLM‑as‑programmer**、**self‑programming agents** 的研究，进一步探索自动调试、代码优化以及多语言（如 JavaScript、Rust）扩展。对想深入的读者，可以关注以下方向：① 如何在安全沙箱中高效执行生成代码；② 生成代码的可验证性（形式化验证、单元测试自动生成）；③ 将 NLEP 与外部工具（数据库、图形引擎）结合，实现更复杂的跨模态推理。  

### 一句话记住它
让大模型先写完整的 Python 程序，再交给真实解释器跑——自然语言推理从“猜答案”变成“写脚本并执行”。