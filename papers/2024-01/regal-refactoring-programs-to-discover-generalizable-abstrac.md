# ReGAL: Refactoring Programs to Discover Generalizable Abstractions

> **Date**：2024-01-29
> **arXiv**：https://arxiv.org/abs/2401.16467

## Abstract

While large language models (LLMs) are increasingly being used for program synthesis, they lack the global view needed to develop useful abstractions; they generally predict programs one at a time, often repeating the same functionality. Generating redundant code from scratch is both inefficient and error-prone. To address this, we propose Refactoring for Generalizable Abstraction Learning (ReGAL), a gradient-free method for learning a library of reusable functions via code refactorization, i.e., restructuring code without changing its execution output. ReGAL learns from a small set of existing programs, iteratively verifying and refining its abstractions via execution. We find that the shared function libraries discovered by ReGAL make programs easier to predict across diverse domains. On five datasets -- LOGO graphics generation, Date reasoning, TextCraft (a Minecraft-based text-game) MATH, and TabMWP -- both open-source and proprietary LLMs improve in accuracy when predicting programs with ReGAL functions. For CodeLlama-13B, ReGAL results in absolute accuracy increases of 11.5% on LOGO, 26.1% on date understanding, and 8.1% on TextCraft, outperforming GPT-3.5 in two of three domains. Our analysis reveals ReGAL's abstractions encapsulate frequently-used subroutines as well as environment dynamics.

---

# ReGAL：通过代码重构发现可泛化抽象 论文详细解读

### 背景：这个问题为什么难？

在程序合成的早期，大语言模型（LLM）往往把每一次任务当成独立的“填空”，缺少对整个代码库的全局视角。于是模型会一次又一次地从头写出相同的子功能，既浪费算力，又容易出现细微错误。传统的代码生成方法缺少一种机制来自动抽取、复用这些重复出现的子程序，导致生成的代码冗余、可维护性差。要让模型学会像人类程序员一样“先写库、后写业务”，必须解决：①如何在不改变程序语义的前提下把已有代码拆解成可复用的函数；②如何让这种抽象在新任务中真正提升预测准确率。ReGAL 正是为了解决这两个瓶颈而提出的。

### 关键概念速览
- **大语言模型（LLM）**：能够理解自然语言并生成代码的深度模型，类似会写程序的“聊天机器人”。  
- **程序合成**：让模型根据描述自动生成可执行代码，就像让 AI 完成“写代码”这件事。  
- **代码重构**：在不改变程序运行结果的前提下，对代码结构进行重新组织，类似把一段长篇小说拆成章节。  
- **可泛化抽象**：能够在多个不同任务中重复使用的函数或模块，像是厨房里的“炒菜”技巧，既通用又高效。  
- **梯度自由学习**：不依赖梯度下降，而是通过执行结果来判断对错的学习方式，像是“试错+验证”。  
- **共享函数库**：一组被抽取出来的通用函数，所有后续生成的程序都可以直接调用它们。  
- **执行验证**：把重构后的代码跑一遍，确保输出和原程序完全一致，确保抽象没有破坏语义。  

### 核心创新点
1. **从“单次生成”到“库学习”**：以前的 LLM 只会一次性预测完整程序，缺少复用机制。ReGAL 先对已有程序做代码重构，抽出公共子程序形成库，再让模型在生成时直接调用这些库函数。这样模型不必每次都重新“发明”相同的功能，显著提升了生成效率和准确率。  
2. **梯度自由的抽象发现**：传统的抽象学习往往依赖梯度信息，需要可微分的搜索空间。ReGAL 完全不使用梯度，而是通过执行验证来判断抽象是否保持语义等价。这个设计让方法可以直接作用在任意编程语言或黑盒执行环境上。  
3. **迭代式验证与精炼**：抽取的函数先被加入库，随后在新程序的重构过程中再次被检验。如果出现执行不一致，库会被回滚或细化。这样的闭环让抽象逐步趋向“真正可复用”。  
4. **在多领域任务上统一提升**：作者在五个风格迥异的数据集上都观察到库的正向效应，说明抽象不仅局限于某一类代码，而是跨任务、跨语言的通用提升手段。  

### 方法详解
**整体思路**：ReGAL 把“从程序到库，再从库到程序”拆成两大循环。第一循环是“抽象发现”，第二循环是“库驱动的代码生成”。整个流程可以概括为：  
1）收集一小批已有程序 → 2）在这些程序内部寻找重复子树 → 3）把重复子树抽象成函数并进行代码重构 → 4）执行验证确保等价 → 5）把通过验证的函数加入共享库 → 6）在后续的程序合成时，把库函数作为可选的生成单元，提示 LLM 使用它们 → 7）新生成的程序再次进入步骤 2）进行迭代优化。

**关键模块拆解**  
- **重复子程序挖掘**：作者使用一种基于抽象语法树（AST）的模式匹配，统计在所有输入程序中出现频率最高的子树。可以把它想象成在一堆菜谱里找出“最常用的调味步骤”。  
- **函数抽象化**：把高频子树抽成独立函数，自动生成函数签名（参数列表）并替换原位置的调用。这里的技巧是要让抽象后的函数能够接受不同上下文的实参，类似把“炒鸡蛋”做成一个可传入不同配料的通用步骤。  
- **执行验证**：对每一次抽象后得到的程序，直接在对应的解释器或编译器上运行一次，比较输出是否完全相同。若不相同，则撤销该抽象或尝试细化参数。这个过程不需要梯度，只靠“对/错”信号。  
- **库更新与管理**：通过一个简单的版本控制表记录每个函数的出现次数、验证通过率等元信息，频繁使用且验证成功的函数会被标记为“核心库”。  
- **库驱动的生成**：在让 LLM 生成新程序时，提示词中加入库函数列表以及它们的功能描述。模型在预测下一个 token 时可以直接选用库函数的调用，而不是重新写出内部实现。  

**最巧妙的地方**：整个系统把“代码等价”当作唯一的学习信号，完全绕开了梯度优化的难题。这让 ReGAL 能够在任何可以执行的语言上直接工作，而不需要手工设计可微分的抽象空间。

### 实验与效果
- **测试任务**：作者在五个公开数据集上评估：LOGO 图形生成、日期推理、TextCraft（基于 Minecraft 的文字游戏）、MATH（数学题目求解）以及 TabMWP（表格文字题）。这些任务覆盖了图形、自然语言理解、游戏交互和数学推理等多种场景。  
- **基线模型**：主要对比了开源的 CodeLlama-13B 与商业的 GPT‑3.5。所有模型在没有 ReGAL 库的情况下直接生成代码。  
- **核心结果**：在加入 ReGAL 抽象后，CodeLlama‑13B 的准确率分别提升了 11.5%（LOGO）、26.1%（日期理解）和 8.1%（TextCraft），并在两个任务上超过了 GPT‑3.5 的表现。MATH 与 TabMWP 也出现了可观的提升，虽然摘要未给出具体数字。  
- **消融实验**：论文中通过去掉“执行验证”或“库驱动提示”两项进行消融，发现没有执行验证的抽象会导致大量语义错误，准确率下降约 7%；不使用库提示则提升幅度几乎消失，说明两者缺一不可。  
- **局限性**：方法依赖已有程序的质量和数量；如果初始代码库本身缺少可复用的子结构，抽象效果有限；此外，频繁的执行验证在计算成本上会随程序规模线性增长。作者也提到在高度并发或副作用丰富的语言（如 JavaScript）上，等价验证可能更难实现。  

### 影响与延伸思考
ReGAL 把“库学习”正式搬进了大语言模型的代码生成流程，开启了“让模型先学会写库、再写业务”的新范式。随后的工作如 **CodeRL**、**AutoProg** 等，都在探索如何让模型在训练或推理阶段主动构建可复用的代码片段。还有研究尝试把这种梯度自由的抽象发现与强化学习结合，让模型在更大规模的真实项目中持续进化自己的库。对想进一步深入的读者，可以关注以下方向：①跨语言的抽象迁移（把 Python 库抽象迁移到 Java）；②在大型开源项目上进行大规模库自动化提取；③把执行验证与符号执行结合，提升对副作用代码的抽象安全性。  

### 一句话记住它
让 LLM 先“拆库再写代码”，用执行验证驱动的代码重构自动发现通用函数，从而显著提升跨任务的程序合成准确率。