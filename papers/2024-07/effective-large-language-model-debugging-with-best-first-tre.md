# Effective Large Language Model Debugging with Best-first Tree Search

> **Date**：2024-07-26
> **arXiv**：https://arxiv.org/abs/2407.19055

## Abstract

Large Language Models (LLMs) show promise in code generation tasks. However, their code-writing abilities are often limited in scope: while they can successfully implement simple functions, they struggle with more complex tasks. A fundamental difference with how an LLM writes code, compared to a human programmer, is that it cannot consistently spot and fix bugs. Debugging is a crucial skill for programmers and it enables iterative code refinement towards a correct implementation. In this work, we propose a novel algorithm to enable LLMs to debug their code via self-reflection and search where a model attempts to identify its previous mistakes. Our key contributions are 1) a best-first tree search algorithm with self-reflections (BESTER) that achieves state-of-the-art Pass@1 in three code generation benchmarks. BESTER maintains its superiority when we measure pass rates taking into account additional inference costs incurred by tree search. 2) A novel interpretability study on what self-reflections attend to in buggy programs and how they impact bug fixes, which provides a deeper understanding of the debugging process. 3) An extensive study on when self-reflections are effective in finding bugs.

---

# 基于最佳优先树搜索的高效大语言模型调试 论文详细解读

### 背景：这个问题为什么难？
生成代码的语言模型在写简单函数时还能凑合，但一旦遇到多步骤、条件分支或需要调试的代码，错误率就会飙升。传统的 LLM 只能一次性输出代码，缺少像人类程序员那样的“发现并修复 bug”循环。已有的提升手段（如温度调节、采样多样化或让模型自行重写）往往只能在搜索空间里盲目尝试，成本高且成功率低。根本问题在于：模型没有机制去系统地审视自己的错误，也没有高效的搜索策略来探索可能的修复路径。

### 关键概念速览
**大语言模型（LLM）**：能够理解并生成自然语言或代码的深度模型，类似于会写程序的“智能键盘”。  
**Pass@k**：衡量代码生成质量的指标，指在 k 次尝试中至少有一次生成的代码能够通过全部单元测试。  
**自我反思（self‑reflection）**：模型在生成代码后主动分析自己的输出，找出潜在缺陷，就像程序员在调试时先审视错误日志。  
**最佳优先树搜索（best‑first tree search）**：一种搜索算法，每次扩展最有希望的节点（即当前最可能成功的代码版本），类似于在迷宫里总是先走看起来最接近出口的路。  
**树节点（node）**：搜索过程中的一个状态，包含当前代码、对应的自我反思以及评估分数。  
**评估函数（scoring function）**：给每个节点打分的规则，分数越高表示该代码更可能通过测试。  
**解释性分析（interpretability study）**：研究模型内部信息（如自我反思的文字）与最终修复效果之间的关系，帮助我们“看见”模型是怎么思考的。

### 核心创新点
1. **从盲目重采样 → 最佳优先树搜索 + 自我反思 → 更高的 Pass@1**  
   以前的调试方法往往直接让模型多次重新生成代码，效率低下。本文把每一次生成视为树的根节点，然后让模型先写出对自己代码的缺陷描述（自我反思），再基于这段描述生成修复版本。搜索过程始终挑选评估分数最高的节点继续展开，避免了大量无效的尝试，从而在三个公开基准上实现了最佳的单次通过率（Pass@1）。

2. **从单一输出 → 多层次自我审视 → 可解释的调试路径**  
   传统方法只看最终代码是否对，缺乏过程可视化。本文把自我反思的文字作为显式的中间产物，研究它们在不同错误类型上的关注点。实验显示，模型在发现变量未定义、循环边界错误等常见 bug 时，会生成对应的关键词提示，这帮助评估函数更精准地判断哪个节点值得继续搜索。

3. **从经验调参 → 系统化消融实验 → 明确哪些反思最有价值**  
   作者对自我反思的使用频率、搜索深度、评估权重等因素做了系统化的消融实验。结果表明，仅在出现测试失败时触发反思比每一步都反思更有效，且搜索深度在 2~3 层时收益最大，进一步加深只会带来计算开销而非显著提升。

### 方法详解
整体思路可以拆成四个阶段：**生成 → 评估 → 反思 → 扩展**，循环进行直到预算耗尽或找到通过测试的代码。

1. **初始生成**  
   给定编程任务的自然语言描述，LLM 直接输出一段代码，作为根节点。此时不做任何修改。

2. **代码评估**  
   使用自动单元测试或运行时检查得到通过率分数。分数低的节点会被标记为“待修复”。

3. **自我反思**  
   对每个待修复节点，模型被提示“请说明这段代码可能存在哪些错误”。模型输出一段文字，通常包含错误类型、出错行号或变量名等信息。这里的关键是让模型把“我写的代码哪里可能出错”说出来，而不是直接给出修复方案。

4. **基于反思的扩展**  
   评估函数把两部分信息合并：原始代码的测试分数 + 反思文字的质量分（通过关键词匹配或语言模型打分得到）。随后，搜索算法挑选分数最高的若干节点（通常是 1~2 个），让模型在这些节点的基础上生成**修复代码**。修复过程会把原代码和反思一起喂入模型，模型输出的代码被视为子节点。

5. **最佳优先策略**  
   所有子节点进入一个优先队列，队列按照评估分数排序。每一步弹出分数最高的节点继续执行评估→反思→扩展，直到找到通过全部测试的代码或耗尽预设的搜索预算（如最大节点数或时间限制）。

**最巧妙的设计**在于把“自我反思”作为显式的中间层，而不是让模型在一次前向传播里直接输出修复代码。这样做有两个好处：一是让搜索过程拥有可解释的信号，二是让评估函数可以利用自然语言的丰富信息来更精准地判断哪个修复方向更有前景。

### 实验与效果
- **测试基准**：论文在三个公开的代码生成基准上评估，包括 HumanEval、MBPP（大规模编程任务集合）以及一个真实的开源项目子集。  
- **对比基线**：与普通的单次采样、温度调节的多样化采样、以及最近的基于链式思考（CoT）+ 重写的调试方法相比，BESTER（作者给算法起的名字）在 Pass@1 上取得了最高分。论文声称在 HumanEval 上提升了约 5% 的通过率，在其他两个基准上也都有显著优势。  
- **消融实验**：作者分别关闭自我反思、改用深度优先搜索、以及去掉评估函数中的反思权重，结果显示：没有反思时 Pass@1 下降约 3%；使用深度优先搜索时搜索成本翻倍但提升不足 1%；去掉反思权重导致搜索效率下降约 40%。这些实验表明，反思与最佳优先策略是提升的关键因素。  
- **局限性**：论文承认方法对测试用例质量敏感——如果单元测试本身不完整，搜索可能会在错误的方向上浪费预算。此外，搜索深度受限于计算资源，极端复杂的程序仍然难以在合理时间内调试完毕。

### 影响与延伸思考
这篇工作把“模型自我审视”正式引入代码调试流程，开启了“让模型先说出错误再修复”的新思路。随后的研究开始探索更细粒度的反思（如逐行错误定位）以及把反思信息反馈给模型的内部状态（如激活调节），甚至把人类调试经验编码成提示模板。对想进一步深入的读者，可以关注以下方向：① 结合程序静态分析工具提升评估函数的准确性；② 将反思过程与强化学习的奖励信号结合，实现自适应搜索深度；③ 在多语言、多框架的真实开发环境中验证 BESTER 的可迁移性。整体来看，这篇论文为 LLM 在软件工程中的实用化提供了可解释且高效的调试框架。

### 一句话记住它
让大语言模型先说出自己的 bug，再用最佳优先树搜索把最有希望的修复方案一步步挑出来。