# Structured Chain-of-Thought Prompting for Code Generation

> **Date**：2023-05-11
> **arXiv**：https://arxiv.org/abs/2305.06599

## Abstract

Large Language Models (LLMs) (e.g., ChatGPT) have shown impressive performance in code generation. LLMs take prompts as inputs, and Chain-of-Thought (CoT) prompting is the state-of-the-art prompting technique. CoT prompting asks LLMs first to generate CoTs (i.e., intermediate natural language reasoning steps) and then output the code. However, CoT prompting is designed for natural language generation and has low accuracy in code generation.   In this paper, we propose Structured CoTs (SCoTs) and present a novel prompting technique for code generation, named SCoT prompting. Our motivation is source code contains rich structural information and any code can be composed of three program structures (i.e., sequence, branch, and loop structures). Intuitively, structured intermediate reasoning steps make for structured source code. Thus, we ask LLMs to use program structures to build CoTs, obtaining SCoTs. Then, LLMs generate the final code based on SCoTs. Compared to CoT prompting, SCoT prompting explicitly constrains LLMs to think about how to solve requirements from the view of source code and further the performance of LLMs in code generation. We apply SCoT prompting to two LLMs (i.e., ChatGPT and Codex) and evaluate it on three benchmarks (i.e., HumanEval, MBPP, and MBCPP). (1) SCoT prompting outperforms the state-of-the-art baseline - CoT prompting by up to 13.79% in Pass@1. (2) Human evaluation shows human developers prefer programs from SCoT prompting. (3) SCoT prompting is robust to examples and achieves substantial improvements.

---

# 结构化思维链提示用于代码生成 论文详细解读

### 背景：这个问题为什么难？
代码生成看似只要让大模型直接输出程序，但实际需求往往涉及多层次的控制流——顺序执行、条件分支、循环迭代。传统的 **Chain-of-Thought（CoT）** 提示让模型先写自然语言的推理步骤，适合解答数学题或阅读理解，却没有强制模型围绕代码的结构化特性思考。于是模型常会在细节上走偏，导致生成的代码在逻辑上不完整或语法错误。换句话说，缺少对程序结构的显式约束是导致代码生成准确率受限的根本原因。

### 关键概念速览
**LLM（大语言模型）**：能够理解并生成自然语言或代码的深度学习模型，如 ChatGPT、Codex。  
**Prompt（提示）**：给模型的输入文本，决定模型的思考方向和输出格式。  
**CoT（思维链）**：让模型在给出答案前先写出一步步的自然语言推理，就像解题时的草稿。  
**结构化 CoT（SCoT）**：在思维链中加入程序结构标签（顺序、分支、循环），让模型的中间推理本身具备代码的骨架。  
**Pass@k**：代码生成评估指标，表示在 k 次尝试中至少有一次生成的代码能够通过所有单元测试。  
**HumanEval / MBPP / MBCPP**：三个公开的代码生成基准，分别覆盖 Python、JavaScript 等语言的函数实现任务。  
**示例鲁棒性**：指提示中加入的示例（few‑shot）数量或质量变化时，模型性能的稳定程度。

### 核心创新点
1. **从自然语言思维链到结构化思维链**  
   *之前的 CoT 让模型写“先判断输入是否合法，再计算结果”，完全基于自然语言推理。*  
   *本文改为让模型先输出包含“Sequence、Branch、Loop”标签的结构化步骤（SCoT），相当于先画出程序的流程图。*  
   *这种显式的结构约束把模型的注意力从抽象推理转向代码的控制流，显著提升了生成代码的逻辑完整性。*

2. **统一的三大程序结构抽象**  
   *过去的提示没有统一的结构化语言，导致不同任务的中间步骤风格千差万别。*  
   *作者提出所有代码都可以拆解为顺序、分支、循环三类基本结构，并在提示模板中强制模型使用这三类描述。*  
   *这种抽象让模型在面对任意需求时都有统一的思考框架，提升了跨任务的迁移效果。*

3. **两阶段生成流程**  
   *传统方法直接从需求到代码，一步到位。*  
   *SCoT 提示把过程拆成“先生成结构化思维链 → 再依据思维链生成代码”。*  
   *两阶段设计让错误更容易定位：如果代码不对，先检查 SCoT 是否合理；如果 SCoT 本身有问题，再回到提示调优。*

4. **对示例数量的鲁棒性提升**  
   *在 few‑shot 场景下，增加或减少示例往往会导致性能波动。*  
   *实验显示，SCoT 提示在示例数量变化时仍能保持较高的 Pass@1，说明结构化约束本身已经提供了足够的指导信息。*

### 方法详解
**整体框架**  
SCoT 提示的工作流分为三步：  
1) **构造结构化提示**：在原始需求后加入“请先用 Sequence、Branch、Loop 描述你的解题思路”。  
2) **生成结构化思维链（SCoT）**：模型依据提示输出一段文字，每一行对应一种程序结构，明确每个结构内部的操作顺序。  
3) **基于 SCoT 生成代码**：模型再次接收同一需求和刚才的 SCoT，按照结构化描述逐段写出实际代码。

**关键模块拆解**  
- **提示模板**：作者设计了一个固定的模板，类似  
  ```
  需求：<用户描述>  
  请先用以下三种标签描述你的解题思路，每个标签对应一种代码结构：  
  [Sequence] …  
  [Branch] … (if … else …)  
  [Loop] … (for/while …)  
  最后请基于上述思路直接给出完整代码。  
  ```  
  这种模板把模型的注意力强制聚焦在“怎么组织代码”上，而不是仅仅在自然语言解释。  

- **结构化思维链的生成**：模型在看到标签后，会把需求拆解成若干子任务。例如，对于“计算列表中所有偶数的和”，模型可能输出：  
  ```
  [Sequence] 初始化 sum 为 0  
  [Loop] 遍历列表中的每个元素 x  
  [Branch] 如果 x 为偶数，则 sum += x  
  ```  
  这里每一步都对应代码的一个基本块，形成了类似伪代码的骨架。  

- **代码生成阶段**：模型把上面的 SCoT 当作“执行计划”，逐块转化为真实代码。因为每块已经明确了控制流，模型只需要填充具体的语法细节，如变量名、循环语法等。  

**最巧妙的设计**  
- **结构标签的强制约束**：相比让模型自行决定思考顺序，强制使用固定标签让模型的内部表征更接近编译器的抽象语法树（AST），从而自然产生符合语法的代码。  
- **两阶段解耦**：错误定位变得线性——先检查 SCoT 是否合理（人类可读），再检查代码实现是否忠实于 SCoT，这比一次性调试整段代码要省事得多。

### 实验与效果
- **测试数据集**：在三大公开基准上评估：HumanEval（Python 函数）、MBPP（Python 小项目）以及 MBCPP（C++/JavaScript 代码）。  
- **对比基线**：主要与传统 CoT 提示以及直接一次性生成的 Zero‑shot/ Few‑shot 方法比较。  
- **核心结果**：  
  - 在 Pass@1 指标上，SCoT 提示相较于 CoT 提示提升最高可达 **13.79%**（具体数值在不同数据集上略有差异）。  
  - 人类评审实验显示，开发者更倾向于接受 SCoT 生成的代码，认为其可读性和逻辑性更好。  
  - 在示例数量变化的鲁棒性实验中，SCoT 的表现波动幅度明显低于 CoT，说明结构化约束本身提供了足够的指导。  
- **消融实验**：作者分别去掉结构标签、去掉两阶段流程或只保留单一结构（如仅 Sequence），发现每一项都导致性能回落，验证了三大结构和两阶段设计的必要性。  
- **局限性**：论文未深入探讨 SCoT 在极其复杂的递归或面向对象设计中的表现，也没有提供针对不同编程语言的标签细化方案。  

### 影响与延伸思考
SCoT 的出现把“思维链”从纯自然语言推理拓展到代码结构层面，开启了 **结构化提示** 的新方向。随后的工作开始尝试把更丰富的抽象语法树（AST）信息直接注入提示，或让模型输出中间的伪代码再交给专门的代码合成器。对于想进一步探索的读者，可以关注以下几个方向：  
- **AST‑guided Prompting**：把编译器的树形结构直接作为提示的一部分。  
- **多语言结构化标签**：为不同语言设计专属的控制流标签（如 Java 的 try‑catch）。  
- **自动化标签生成**：利用小模型先对需求进行结构化分解，再交给大模型完成代码。  

这些方向都有望把提示工程提升到更接近程序语言本身的层次。

### 一句话记住它
**让大模型先用“顺序‑分支‑循环”三种结构写思维链，再把这张结构图直接翻译成代码，效果比普通思维链提升近 14%。**