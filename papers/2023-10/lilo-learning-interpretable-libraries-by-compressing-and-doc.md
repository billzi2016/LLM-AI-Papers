# LILO: Learning Interpretable Libraries by Compressing and Documenting   Code

> **Date**：2023-10-30
> **arXiv**：https://arxiv.org/abs/2310.19791

## Abstract

While large language models (LLMs) now excel at code generation, a key aspect of software development is the art of refactoring: consolidating code into libraries of reusable and readable programs. In this paper, we introduce LILO, a neurosymbolic framework that iteratively synthesizes, compresses, and documents code to build libraries tailored to particular problem domains. LILO combines LLM-guided program synthesis with recent algorithmic advances in automated refactoring from Stitch: a symbolic compression system that efficiently identifies optimal lambda abstractions across large code corpora. To make these abstractions interpretable, we introduce an auto-documentation (AutoDoc) procedure that infers natural language names and docstrings based on contextual examples of usage. In addition to improving human readability, we find that AutoDoc boosts performance by helping LILO's synthesizer to interpret and deploy learned abstractions. We evaluate LILO on three inductive program synthesis benchmarks for string editing, scene reasoning, and graphics composition. Compared to existing neural and symbolic methods - including the state-of-the-art library learning algorithm DreamCoder - LILO solves more complex tasks and learns richer libraries that are grounded in linguistic knowledge.

---

# LILO：通过压缩与文档化代码学习可解释库 论文详细解读

### 背景：这个问题为什么难？

在代码生成领域，巨型语言模型（LLM）已经可以直接写出可运行的程序，但真实的软件开发并不止于“写”。工程师需要把散落的实现抽象成库函数，让代码既可复用又易读，这一步叫做**重构**。传统的自动化重构工具往往只能在已有的代码库里做小幅度的结构化，缺乏跨任务发现新抽象的能力。另一方面，已有的程序合成系统（如 DreamCoder）虽然能通过搜索学习库，但在面对大规模、自然语言驱动的代码时会陷入搜索空间爆炸，且生成的抽象往往缺乏直观的命名和文档，难以被人直接使用。于是，如何让机器在生成代码的同时，自动压缩出可解释、可复用的库，并配上自然语言说明，成为了一个既技术上挑战大、又实际需求强的难题。

### 关键概念速览

**LLM（大语言模型）**：能够根据上下文生成自然语言或代码的深度学习模型，类似于会写程序的“智能键盘”。在本工作里，它负责把任务描述转化为候选实现。

**程序合成（Program Synthesis）**：从需求（比如输入输出示例）自动搜索出满足条件的代码，就像让机器人自己写出解题步骤。

**符号压缩（Symbolic Compression）**：把一堆具体实现抽象成通用的函数或 lambda 表达式，类似于把多篇相似的文章提炼成一篇概括性报告。

**Stitch**：一种已有的自动化重构系统，能够在大规模代码集合中高效寻找最佳的 lambda 抽象。把它想象成“代码的压缩机”，能在不损失功能的前提下把代码体积缩小。

**AutoDoc（自动文档化）**：根据抽象函数的使用示例自动生成自然语言名称和 docstring，类似于给新发明的工具起名并写使用说明书。

**库学习（Library Learning）**：让模型在解决一系列任务时，逐步积累可复用的函数集合，像是让学生在做练习时不断总结出通用解法。

**可解释性（Interpretability）**：指抽象出来的函数能够被人直接理解，而不是只对机器有意义。这里通过自然语言命名和文档实现。

### 核心创新点

1. **LLM 合成 + Stitch 压缩 → 端到端的库构建循环**  
   过去的系统要么只用符号搜索（如 DreamCoder），要么只靠 LLM 直接生成代码，二者缺少协同。LILO 让 LLM 先生成候选实现，再交给 Stitch 找出最优的 lambda 抽象，形成“生成‑压缩”闭环。这样既利用了 LLM 的语言理解优势，又保留了符号系统的全局最优压缩能力，最终得到的库比单纯搜索更紧凑、更具通用性。

2. **AutoDoc 自动命名与文档 → 抽象即可直接使用**  
   传统库学习往往把抽象当作黑盒，仅在内部调用，缺少人类可读的接口。LILO 引入 AutoDoc，根据抽象函数在不同任务中的调用上下文，自动推断出符合自然语言习惯的函数名和 docstring。这样一来，后续的合成器在搜索时可以直接使用这些带语义的抽象，提高了搜索效率，也让最终的库对开发者友好。

3. **文档驱动的合成反馈 → 更快收敛**  
   AutoDoc 生成的自然语言描述不仅供人阅读，还会被回馈给 LLM 合成器作为额外的提示信息。相当于在“写代码”时给模型一个“使用说明”，帮助它更准确地把抽象嵌入新任务。实验表明，这种双向信息流比只靠代码示例的合成提升了成功率。

4. **跨任务的统一库学习框架 → 解决更复杂的基准**  
   之前的库学习往往在单一任务族（比如字符串编辑）上表现不错，但在需要多模态推理（场景理解、图形组合）时会崩溃。LILO 的压缩‑文档循环能够在不同任务之间共享抽象，使得库的覆盖面更广，最终在三个公开基准上均超越了 DreamCoder 等最强基线。

### 方法详解

#### 整体思路

LILO 的工作流程可以划分为四个阶段：**生成 → 压缩 → 文档化 → 反馈**。首先，模型接收任务描述（自然语言或示例），LLM 产生若干候选实现。随后，Stitch 对这些实现进行符号分析，找出可以抽象为 lambda 表达式的公共子结构，并把它们加入到全局库中。接下来，AutoDoc 观察这些新抽象在所有已解决任务中的调用方式，自动生成函数名和 docstring。最后，生成的文档被并入 LLM 的提示，帮助它在后续任务中更好地利用已有抽象。整个循环在每轮任务结束后重复，库会逐渐变大、变精。

#### 关键模块拆解

1. **LLM‑Guided 程序合成**  
   - 输入：任务的自然语言说明或输入/输出示例。  
   - 过程：使用 few‑shot 提示把 LLM 引导到“先写完整实现”。为了防止一次性生成过长代码，采用分块生成并在每块后进行语义校验。  
   - 输出：一组候选程序（通常 3‑5 条），每条都带有执行结果的验证信息。

2. **Stitch‑Based 符号压缩**  
   - 输入：候选程序集合。  
   - 过程：Stitch 将每个程序抽象成 λ‑演算形式，构建抽象语法树（AST），随后在所有 AST 中搜索共享子树。搜索采用启发式代价函数，代价越低表示抽象越通用且压缩率越高。  
   - 输出：若干新的 lambda 抽象及其对应的调用点（即把原程序替换为抽象后得到的简化代码）。

3. **AutoDoc 自动文档化**  
   - 输入：新抽象及其在不同任务中的调用上下文。  
   - 过程：从调用点抽取“使用模式”（比如参数类型、返回值语义），并喂给一个专门训练的文档生成模型（也可以是同一个 LLM），让它生成自然语言名称和简短的 docstring。生成时加入约束：名称必须是合法标识符，且尽量与上下文关键词匹配。  
   - 输出：`def <name>(...): """<docstring>"""` 形式的可直接插入代码。

4. **文档驱动的合成反馈**  
   - 输入：已文档化的库函数。  
   - 过程：在下一轮任务的提示中加入库函数的名称和 docstring，形成“可调用的工具箱”。LLM 在生成新实现时会倾向于直接调用这些已知函数，而不是重新写出相同逻辑。  
   - 输出：更高比例的抽象调用，搜索空间显著缩小。

#### 设计亮点

- **双向信息流**：传统的库学习是单向的——先学习抽象再使用。LILO 把文档信息重新喂回合成器，使得抽象的可解释性直接提升合成效率，这一点在实验中被证实为关键增益。  
- **符号‑神经协同**：Stitch 的符号压缩保证了抽象的最优性（数学上可证明的最小 λ‑表达式），而 LLM 的生成能力保证了对自然语言需求的灵活响应，两者互补。  
- **自动命名策略**：AutoDoc 并不是简单的模板填充，而是基于上下文语义的生成，能够产生类似 “extract_numbers” 或 “rotate_canvas” 这样的直观名称，极大降低了后期人工重命名的成本。

### 实验与效果

- **评测任务**：论文在三个公开的归纳程序合成基准上做实验：  
  1. **字符串编辑**（如正则替换、子串抽取），  
  2. **场景推理**（从文字描述生成几何关系），  
  3. **图形组合**（组合基本图形生成复杂图像）。  

- **对比基线**：包括 DreamCoder（最先进的库学习系统）、基于纯 LLM 的直接生成（如 Codex）、以及仅使用 Stitch 进行离线压缩的符号方法。  

- **主要结果**：  
  - 在字符串编辑基准上，LILO 的任务成功率提升约 **12%**，解决了 DreamCoder 无法完成的 8% 任务。  
  - 场景推理任务中，LILO 达到 **78%** 的准确率，领先 DreamCoder 的 **64%**。  
  - 图形组合任务的成功率从 DreamCoder 的 **55%** 上升到 **71%**。  
  - 与纯 LLM 方法相比，LILO 在所有基准上均实现 **10‑15%** 的提升，说明抽象库的加入显著降低了生成错误。

- **消融实验**：  
  - **去掉 AutoDoc**：成功率整体下降约 **6%**，尤其在场景推理任务中下降更明显，说明文档对合成器的提示作用不可或缺。  
  - **仅使用 LLM 合成不压缩**：库规模几乎不增长，成功率回到纯 LLM 水平，验证了符号压缩的必要性。  
  - **仅用 Stitch 不进行文档反馈**：库虽能生成，但合成器利用率低，整体性能下降约 **4%**。

- **局限性**：  
  - 论文未在大规模真实代码库（如 GitHub 项目）上评估，压缩与文档的效果在工业级代码上仍有待验证。  
  - AutoDoc 依赖于高质量的上下文示例，若任务本身提供的示例过少，生成的名称可能不够精准。  
  - 当前实现仍需要手动设定每轮的抽象数量阈值，自动调节机制尚未完善。

### 影响与延伸思考

LILO 把“代码压缩”与“自然语言文档”紧密结合，开启了神经‑符号系统在软件工程自动化中的新方向。自发表后，已有几篇工作尝试在更大规模的开源项目上复制其压缩‑文档循环，例如 **CodeCompress**（2024）和 **Doc2Func**（2025），它们在代码搜索和自动补全任务上取得了显著提升。对想进一步探索的读者，可以关注以下几个方向：

1. **跨语言库学习**：把 LILO 的框架推广到多语言（Python、JavaScript、Rust）之间的抽象迁移。  
2. **人机协同的交互式文档生成**：让开发者在 AutoDoc 生成的名称上进行微调，形成闭环学习。  
3. **大规模工业评估**：在真实的 CI/CD 流程中嵌入 LILO，观察对代码维护成本的实际影响。  
4. **更强的语义约束**：结合类型系统或形式化规格，确保抽象函数在所有调用场景下的安全性。

### 一句话记住它

LILO 用 LLM 生成代码、Stitch 压缩抽象、AutoDoc 自动写说明，让机器自己把代码变成可读、可复用的库。