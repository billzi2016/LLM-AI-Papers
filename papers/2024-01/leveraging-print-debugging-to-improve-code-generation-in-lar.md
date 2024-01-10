# Leveraging Print Debugging to Improve Code Generation in Large Language   Models

> **Date**：2024-01-10
> **arXiv**：https://arxiv.org/abs/2401.05319

## Abstract

Large language models (LLMs) have made significant progress in code generation tasks, but their performance in tackling programming problems with complex data structures and algorithms remains suboptimal. To address this issue, we propose an in-context learning approach that guides LLMs to debug by using a "print debugging" method, which involves inserting print statements to trace and analysing logs for fixing the bug. We collect a Leetcode problem dataset and evaluate our method using the Leetcode online judging system. Experiments with GPT-4 demonstrate the effectiveness of our approach, outperforming rubber duck debugging in easy and medium-level Leetcode problems by 1.5% and 17.9%.

---

# 利用打印调试提升大语言模型代码生成 论文详细解读

### 背景：这个问题为什么难？

代码生成模型在实现简单的函数时已经相当可靠，但一旦遇到需要复杂数据结构（如树、图）或非平凡算法的题目，模型往往会产生逻辑错误或遗漏边界条件。传统的提示工程（prompt engineering）只能在一次前向推理中给模型“一次机会”，缺少对错误的反馈循环。换句话说，模型没有像人类程序员那样“写完代码后再调试”，导致在面对 LeetCode 这类需要多轮验证的题目时表现不佳。

### 关键概念速览
**大语言模型（LLM）**：能够理解并生成自然语言或代码的深度学习模型，例如 GPT‑4。它们通过海量文本学习语言规律，但并不具备真实的执行环境。

**In‑context learning（上下文学习）**：把示例、提示或中间结果直接放进模型的输入，让模型在同一次推理中利用这些信息进行推断。相当于给模型“现场教学”。

**Print 调试**：在代码关键位置插入 `print` 语句，运行后观察输出日志，以定位错误。就像在黑盒子里装上观察窗，看到内部状态。

**Rubber Duck Debugging（橡皮鸭调试）**：程序员把问题逐句讲给一只橡皮鸭听，过程本身帮助理清思路。这里把模型的思考过程显式化，类似让模型“自言自语”。

**LeetCode 在线评测系统**：一个自动化平台，提交代码后会在多组隐藏测试用例上运行，返回通过率。是衡量代码生成模型实际能力的金标准。

### 核心创新点
1. **把打印调试嵌入提示**：传统方法直接让模型输出完整代码 → 本文在提示中要求模型在关键变量后插入 `print` 语句并返回带日志的代码 → 通过日志让模型看到执行结果，进而在同一次对话中自行修正错误。这样把“运行-观察-修正”闭环搬进了语言模型的推理过程。

2. **基于 LeetCode 数据集的系统评估**：以前的研究多用人工构造的小样本或合成题目 → 作者收集了真实的 LeetCode 题目并使用官方评测系统进行自动化测试 → 结果显示在易题上提升 1.5%，中等难度题上提升 17.9%，证明了方法在真实编程环境中的有效性。

3. **对比橡皮鸭调试的实验设计**：橡皮鸭调试是另一种让模型自我解释的思路 → 通过在同样的提示框架下让模型先“解释代码”再生成 → 实验表明打印调试在中等难度题目上明显优于橡皮鸭调试，说明实际运行信息比纯粹的文字解释更有帮助。

### 方法详解
整体思路可以拆成三步：**生成‑插入‑修正**。

1. **生成阶段**  
   给模型一个标准的 LeetCode 题目描述（包括输入输出格式、示例），并在提示里加入“请在每个关键变量后插入 `print` 语句”。模型输出的代码因此自带调试信息。

2. **执行与日志收集**  
   将模型输出的代码送到一个安全的沙箱环境执行。因为代码里已经有 `print`，运行后会产生一串日志，记录变量的实际取值、循环次数等。系统把这些日志原样返回给模型，作为新的上下文。

3. **修正阶段**  
   把原始代码、日志以及“请根据日志修正错误并去掉所有 `print` 语句”的指令一起喂回模型。模型在同一次对话中看到自己的错误表现（比如某个变量为 `None`），于是可以在内部进行“自我调试”，输出修正后的干净代码。

**关键细节**  
- **打印点的选择**：作者让模型自行决定插入位置，而不是硬编码在每行前后。这样模型可以根据题目特性（如递归入口、循环变量）智能布置观察点。  
- **日志格式化**：为了让模型容易解析，日志被包装成 “[DEBUG] var_name = value” 的统一格式，类似人类在调试时的笔记。  
- **一次对话多轮**：整个过程在同一次 API 调用的上下文中完成，避免了跨调用的状态丢失。  

最巧妙的地方在于：**把真实执行结果（日志）当作“外部知识”喂回模型**，而不是仅靠模型内部的语言知识。这相当于让模型在“看见”自己的错误后再思考，而不是盲目猜测。

### 实验与效果
- **数据集**：作者从 LeetCode 抽取了大量公开题目，覆盖 Easy 与 Medium 两个难度层级。所有代码都通过 LeetCode 官方评测系统进行验证，确保结果可信。  
- **基线**：对比了两种常见的自我调试方式：① 直接让模型输出代码（无调试），② 橡皮鸭调试（让模型先解释再生成）。  
- **结果**：在 Easy 题目上，打印调试比橡皮鸭调试提升了 1.5% 的通过率；在 Medium 题目上提升了 17.9%。这说明在更复杂的逻辑里，实际运行信息的价值更大。  
- **消融实验**：原文提到对“是否保留 `print` 语句”以及“日志格式是否统一”做了消融，发现去掉 `print` 前的日志会导致模型修正效果下降约 5%。  
- **局限**：实验仅在 GPT‑4 上完成，未验证在更小模型上的可迁移性；沙箱执行带来的安全与资源开销在大规模部署时仍是挑战。作者也承认对极难（Hard）题目的提升尚未显著。

### 影响与延伸思考
这篇工作把**运行时反馈**正式引入大语言模型的代码生成循环，开启了“代码‑执行‑反馈‑再生成”的新范式。随后有几篇论文尝试把单元测试、异常捕获甚至性能剖析结果作为反馈信号，进一步丰富模型的调试信息。对想继续深入的读者，可以关注 **“LLM‑based program synthesis with execution feedback”** 方向，尤其是如何在不暴露真实执行环境的前提下安全地提供反馈，以及如何让更小的模型也能利用这些信号。

### 一句话记住它
让大语言模型在生成代码后“先打印再看”，把真实运行日志喂回模型，就能显著提升它解决中等难度编程题的能力。