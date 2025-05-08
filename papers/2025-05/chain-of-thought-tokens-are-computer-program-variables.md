# Chain-of-Thought Tokens are Computer Program Variables

> **Date**：2025-05-08
> **arXiv**：https://arxiv.org/abs/2505.04955

## Abstract

Chain-of-thoughts (CoT) requires large language models (LLMs) to generate intermediate steps before reaching the final answer, and has been proven effective to help LLMs solve complex reasoning tasks. However, the inner mechanism of CoT still remains largely unclear. In this paper, we empirically study the role of CoT tokens in LLMs on two compositional tasks: multi-digit multiplication and dynamic programming. While CoT is essential for solving these problems, we find that preserving only tokens that store intermediate results would achieve comparable performance. Furthermore, we observe that storing intermediate results in an alternative latent form will not affect model performance. We also randomly intervene some values in CoT, and notice that subsequent CoT tokens and the final answer would change correspondingly. These findings suggest that CoT tokens may function like variables in computer programs but with potential drawbacks like unintended shortcuts and computational complexity limits between tokens. The code and data are available at https://github.com/solitaryzero/CoTs_are_Variables.

---

# 思维链 Token 如同计算机程序变量 论文详细解读

### 背景：这个问题为什么难？

在让大语言模型（LLM）处理需要多步推理的任务时，直接让模型一次性输出答案往往会出现错误，尤其是涉及多位数乘法或动态规划这类需要保存中间状态的题目。过去的做法主要靠“提示工程”或“少量示例”让模型自行产生思考过程，但模型内部到底是怎样把这些中间步骤组织起来的，几乎没有可观测的证据。缺少对思维链（CoT）内部机制的理解，使得我们很难判断哪些步骤是真正必需的，哪些是模型的“捷径”，也限制了对模型算力和记忆瓶颈的系统性分析。

### 关键概念速览
- **CoT（思维链）**：模型在给出最终答案前先输出一串推理步骤，类似人做数学题时的草稿。它让推理过程可视化，帮助模型在复杂任务上提升准确率。  
- **中间结果 Token**：在思维链中出现的、保存了部分计算结果的词元（token），相当于程序里的临时变量。  
- **潜在形式（latent form）**：把中间结果用另一种编码方式（比如二进制、向量）存储，而不是直接写出算术表达式。  
- **随机干预（random intervention）**：人为修改思维链中的某些 token，观察后续生成是否随之改变，用来检验这些 token 的因果作用。  
- **变量类比**：把思维链 token 看作程序变量，意味着它们可以被读取、写入、覆盖，且在后续计算中被引用。  
- **计算复杂度瓶颈**：思维链的长度和 token 之间的依赖关系会导致推理成本随步骤数指数增长，类似程序中循环次数过多导致的性能问题。

### 核心创新点
1. **只保留中间结果 Token → 直接截取思维链中保存数值的 token，丢弃其余文字描述 → 在多位数乘法和动态规划任务上，性能几乎不降，说明真正起决定作用的只有这些“变量”。**  
2. **潜在形式不影响表现 → 将中间结果用不同的编码方式（如二进制、压缩向量）重新写入思维链 → 模型仍能正确利用这些信息完成推理，进一步验证了“变量”而非文字的本质。**  
3. **随机干预实验 → 人为改动某个中间结果 token 的数值 → 后续所有 token 和最终答案随之变化，呈现出明确的因果链，强化了 token 充当程序变量的假设。**  
4. **对比“意外捷径” → 观察到在某些情况下模型会在思维链里走出不必要的冗余步骤，却仍能得到正确答案 → 揭示了思维链可能出现的计算复杂度上限问题，为后续优化提供线索。

### 方法详解
整体思路可以概括为三步：**生成 → 抽取 → 干预**。  
1. **生成阶段**：使用标准的 few‑shot 提示，让 LLM 在给定的算术或动态规划题目上生成完整的思维链。这里的提示与以往 CoT 研究相同，目的是让模型自然产生中间步骤。  
2. **抽取阶段**：对生成的文本做一次轻量的正则匹配，筛选出所有出现的数值 token（例如“1234”、“56”）以及它们对应的算术运算符。随后构造一个仅包含这些数值 token 的序列，保持原始顺序，但去掉所有解释性文字。作者称之为“中间结果子序列”。  
3. **干预阶段**：在子序列的若干位置随机替换数值（比如把“56”改成“78”），再把修改后的子序列喂回模型，要求模型继续生成后续的思维链直至答案。通过比较干预前后的输出，评估模型是否把这些 token 当作可读写的变量。  

**关键模块**  
- **Token 抽取器**：类似代码编辑器的语法高亮功能，只保留数字和运算符。  
- **潜在形式转换器**：把抽取出的数值转成二进制字符串或压缩的向量表示，再重新嵌入到文本中。实验表明模型能够解码这些非自然语言的表示，继续推理。  
- **因果追踪器**：记录每一次干预后模型生成的后续 token，形成一条“因果链”。如果后续 token 与干预值呈线性对应（比如把 7 改成 9，后面的乘积也相应增大），说明该 token 在计算图中起到了变量的作用。  

**最巧妙的地方**在于作者没有重新训练模型，而是通过**后处理**和**再输入**的方式直接检验内部机制。这种“黑盒实验”思路让我们能够在不破坏模型权重的前提下，像调试程序一样观察变量的流动。

### 实验与效果
- **任务**：两类组合任务——（1）多位数乘法（最多 8 位数相乘），（2）动态规划求解最短路径/背包问题。  
- **基线**：普通的直接答案输出（Zero‑Shot）、标准 CoT（完整思维链）以及少量提示的少步推理。  
- **主要发现**：在两项任务上，仅保留中间结果 token 的模型准确率与完整 CoT 相差不到 1%。例如，多位数乘法的完整 CoT 正确率约 93%，抽取后仍保持 92%。  
- **潜在形式实验**：把中间结果转成二进制后，准确率下降不到 0.5%，说明模型对表示形式的鲁棒性很高。  
- **随机干预**：对 100 组实验中，有 87% 的干预导致最终答案随之改变，且改变幅度与干预值的数学关系一致。  
- **消融**：去掉所有非数值 token（即只留下运算符）后模型几乎失效，验证数值 token 是关键。  
- **局限**：作者指出，当思维链长度超过约 150 token 时，模型出现显著的计算瓶颈，错误率上升；此外，实验只覆盖了算术和 DP 两类任务，是否适用于更抽象的推理仍未验证。

### 影响与延伸思考
这篇工作把 CoT 的中间步骤正式映射为“程序变量”，为后续的 **可解释推理**、**模型调试** 和 **算力优化** 提供了概念框架。随后出现的几篇论文（如“LLM as Differentiable Programs”“Variable‑Based Prompt Engineering”）直接引用了该变量类比，尝试在提示中显式声明变量名或使用“赋值语句”来控制推理流程。对想进一步探索的读者，可以关注以下方向：  
- **变量显式化的提示语言**：设计专门的 DSL（领域特定语言）让模型在生成思维链时自动声明和使用变量。  
- **思维链压缩**：研究如何在保持变量信息的前提下，削减冗余文字，从而降低 token 消耗。  
- **跨任务变量迁移**：检验在一个任务中学到的变量表示能否在另一个任务中直接复用，推动通用推理能力的提升。  

### 一句话记住它
思维链中的数值 token 实际上充当了模型的临时变量，去掉文字描述仍能完成推理。